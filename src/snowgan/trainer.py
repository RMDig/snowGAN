import os, atexit, re, shutil, math, gc
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from glob import glob


def assert_spectral_norm_consistency(expected_sn, weights_path, which="Discriminator"):
    """Fail fast when a checkpoint's spectral-norm structure disagrees with the
    model being built.

    A mixed/stale ``save_dir`` (e.g. a config that says ``spectral_norm: true``
    sitting next to a plain-conv weights file) otherwise surfaces as the opaque
    Keras ``expected 2 variables, received 0`` traceback. This raises a clear,
    actionable error instead. No-op when the weights format can't be inspected.
    """
    actual_sn = weights_use_spectral_norm(weights_path)
    if actual_sn is None or actual_sn == bool(expected_sn):
        return
    raise ValueError(
        f"{which} checkpoint at {weights_path} "
        f"{'USES' if actual_sn else 'does NOT use'} spectral normalization, but the "
        f"model config has spectral_norm={bool(expected_sn)}. The saved weights and the "
        f"model architecture disagree — this usually means a mixed/stale save_dir. "
        f"Start fresh: move the directory aside, or pass --rebuild to ignore saved weights."
    )


def cosine_decayed_lr(base_lr, lr_min, post_fade_step, decay_steps):
    """Cosine-annealed learning rate, extracted as a pure function so the
    schedule can be unit-tested without constructing a Trainer.

    Anneals from ``base_lr`` at ``post_fade_step == 0`` down to ``lr_min`` at
    ``post_fade_step == decay_steps``, holding ``lr_min`` thereafter. The
    horizon ``decay_steps`` is a caller-supplied parameter (NOT a hard-coded
    200k): a too-short horizon was what silently floored both LRs ~70% into a
    long run (the post-250k destabilization).
    """
    decay_steps = max(1, int(decay_steps))
    post_fade_step = max(0, int(post_fade_step))
    progress = min(post_fade_step / decay_steps, 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (base_lr - lr_min) * cosine_factor


def _process_rss_mb() -> float:
    """Resident-set size of the current process in MiB. Linux /proc only."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0

from snowgan.checkpoint import resolve_weights_path, to_weights_path, weights_use_spectral_norm
from snowgan.losses import compute_gradient_penalty
from snowgan.generate import generate, make_movie
from snowgan.log import save_history, load_history
from snowgan.data.dataset import DataManager
from snowgan.utils import compute_fade_alpha
from snowgan.augment import augment as diff_augment
from snowgan.memtrace import maybe_create_memtrace

class Trainer:

    def __init__(self, generator, discriminator):

        # Generator and discriminator models
        self.gen = generator
        self.disc = discriminator

        print(f"Initializing trainer in cwd {os.getcwd()} with checkpoint {self.gen.config.checkpoint}...")

        # Construct DataManager early — it derives the canonical pair_depth
        # from config.modality, which is the single source of truth for how
        # the trainer should size its models. The merged-mode depth is
        # len(Modality)=2; single-modality is 1.
        self.dataset = DataManager(self.gen.config)
        target_depth = self.dataset.pair_depth

        # Sync model depth to the dataset's pair depth BEFORE loading weights.
        # If the configured depth doesn't match what the data path produces,
        # the models' input layers are the wrong shape and the train loop
        # used to silently rebuild them mid-batch (UPGRADES #2 + #40),
        # discarding any weights this init would later load. Rebuild here
        # while the models are still fresh and have no trained weights to
        # lose; from this point forward depth is invariant.
        if int(getattr(self.gen.config, "depth", 1)) != target_depth:
            self.gen.config.depth = target_depth
            self.gen = type(self.gen)(self.gen.config)
        if int(getattr(self.disc.config, "depth", 1)) != target_depth:
            self.disc.config.depth = target_depth
            self.disc = type(self.disc)(self.disc.config)
        self._built_depth = target_depth

        # Now load weights — model shapes are aligned with the dataset's
        # pair depth, so load_weights raises on a real shape mismatch
        # rather than masking it with a silent rebuild.
        #
        # --rebuild forces a clean from-scratch start: skip loading entirely so
        # an existing (possibly stale/mismatched) save_dir can't warm-start or
        # crash the run. This is the safe alternative to manually archiving the
        # directory before a breaking-architecture change.
        rebuild = bool(getattr(self.gen.config, "rebuild", False)) or bool(getattr(self.disc.config, "rebuild", False))
        if rebuild:
            print("REBUILD requested: ignoring any saved weights in save_dir, initializing fresh models")
            # One-shot: clear the flag so the first save persists rebuild=False.
            # Otherwise a config saved with rebuild=True would discard weights on
            # every subsequent resume — the inverse footgun.
            self.gen.config.rebuild = False
            self.disc.config.rebuild = False

        gen_weights_path = None if rebuild else resolve_weights_path(self.gen.config.checkpoint)
        if gen_weights_path is not None:
            self.gen.model.load_weights(gen_weights_path)
            print(f"Generator weights loaded from {gen_weights_path}")
            # Load fade endpoints weights (toRGB_prev) for mid-fade resume
            if getattr(self.gen, 'fade_endpoints', None) is not None:
                fade_weights_path = os.path.join(os.path.dirname(self.gen.config.checkpoint), "generator_fade_endpoints.weights.h5")
                if os.path.exists(fade_weights_path):
                    self.gen.fade_endpoints.load_weights(fade_weights_path)
                    print(f"Generator fade endpoints loaded from {fade_weights_path}")
        else:
            print("Generator saved weights not found, new model initialized")

        disc_weights_path = None if rebuild else resolve_weights_path(self.disc.config.checkpoint)
        if disc_weights_path is not None:
            # Fail fast with a clear message if the saved critic's spectral-norm
            # structure disagrees with the config (mixed/stale save_dir), instead
            # of the opaque Keras "expected 2 variables, received 0".
            assert_spectral_norm_consistency(
                getattr(self.disc.config, "spectral_norm", False), disc_weights_path, "Discriminator")
            self.disc.model.load_weights(disc_weights_path)
            print(f"Discriminator weights loaded from {disc_weights_path}")
        else:
            print("Discriminator saved weights not found, new model initialized")

        self.save_dir = self.gen.config.save_dir # Save dictory for the model and it's generated images

        # Diagnostic native-CPU-RAM attributor. No-op unless SNOWGAN_MEMTRACE
        # is set; writes a JSONL trace under save_dir for post-run analysis.
        self._memtrace = maybe_create_memtrace(self.save_dir)

        # Persistent loss-history figure, created lazily in plot_history and
        # reused across calls. Creating a fresh figure every call leaks CPU RAM
        # in the Agg backend even with plt.close(fig) (isolation test: ~1.8 GB
        # over 300 calls vs flat when reused), and plot_history runs every 10
        # batches over a long run.
        self._history_fig = None
        self._history_ax = None

        self.batch_size = self.gen.config.batch_size # Number of images to load in per training batch

        self.n_samples = self.gen.config.n_samples # Number of synthetic images to generate after training
        cleanup_raw = getattr(self.gen.config, "cleanup_milestone", 1000)
        try:
            cleanup_value = int(cleanup_raw)
        except (TypeError, ValueError):
            cleanup_value = 1000
        self.cleanup_milestone = max(0, cleanup_value)
        if hasattr(self.gen.config, "cleanup_milestone"):
            self.gen.config.cleanup_milestone = self.cleanup_milestone
        if hasattr(self.disc.config, "cleanup_milestone"):
            self.disc.config.cleanup_milestone = self.cleanup_milestone
        self._save_interval = self.cleanup_milestone if self.cleanup_milestone > 0 else 1000

        self.disc.config.cleanup_milestone = self.cleanup_milestone
        self._save_interval = self.cleanup_milestone if self.cleanup_milestone > 0 else 1000

        # Persist train/val/test split on first run so downstream consumers
        # (AvAI's evaluation against test_pool) see a stable group-level split
        # of (site, column, core) triples. Idempotent if already derived.
        # (DataManager itself was constructed earlier so its pair_depth could
        # drive the pre-weight-load depth sync above.)
        self._persist_initial_splits()
        # Keep seen profile tracking persistent across configs
        self.gen.config.seen_profiles = self.dataset.seen_profiles
        self.disc.config.seen_profiles = self.dataset.seen_profiles
        # Keep discriminator channel/volume settings aligned with generator at startup
        # (depth is already synced above via dataset.pair_depth).
        self.disc.config.channels = self.gen.config.channels
        self.train_ind = self.gen.config.train_ind
        self.trained_data = []


        self.loss = {'disc': [], 'gen': []}

        self.loss, self.trained_data = load_history(self.gen.config.save_dir) # Load any history we can find in save directory

        atexit.register(self.save_model)

        print("Trainer initialized...")
        # Global step for fade scheduling (persisted in config)
        gen_step = int(getattr(self.gen.config, 'fade_step', 0) or 0)
        disc_step = int(getattr(self.disc.config, 'fade_step', 0) or 0)
        self.global_step = max(gen_step, disc_step)
        self.fade_steps = max(1, int(getattr(self.gen.config, 'fade_steps', 1)))
        self.fade_complete = not (getattr(self.gen.config, 'fade', False) and getattr(self.gen, 'fade_endpoints', None) is not None)
        if self.global_step >= self.fade_steps:
            self.fade_complete = True
        # Keep generator/discriminator configs aligned without forcing a disk write on init
        self._sync_fade_progress(persist=False)

        # --- Post-progressive training improvements ---
        # When spectral norm is active, reduce gradient penalty weight since SN
        # already enforces the Lipschitz constraint — heavy GP on top over-constrains
        # the discriminator and causes color drift / mode collapse.
        use_sn = getattr(self.disc.config, 'spectral_norm', False)
        if use_sn and self.disc.config.lambda_gp > 1.0:
            old_gp = self.disc.config.lambda_gp
            self.disc.config.lambda_gp = 1.0
            print(f"Spectral norm active: reduced lambda_gp from {old_gp} to {self.disc.config.lambda_gp}")

        # Differentiable augmentation
        self.use_augment = getattr(self.gen.config, 'augment', False)

        # Learning rate decay (cosine annealing)
        self.lr_decay = getattr(self.gen.config, 'lr_decay', None)
        self.lr_min = getattr(self.gen.config, 'lr_min', 1e-7)
        self.lr_decay_steps = int(getattr(self.gen.config, 'lr_decay_steps', 0) or 0)
        self._lr_horizon_warned = False
        self.gen_lr_base = float(self.gen.config.learning_rate)
        self.disc_lr_base = float(self.disc.config.learning_rate)

        # Generator EMA (exponential moving average) shadow weights
        self.ema_decay = getattr(self.gen.config, 'ema_decay', 0.0)
        self.ema_weights = None
        if self.ema_decay > 0:
            self._init_ema()

        # FID-based checkpointing
        self.fid_interval = getattr(self.gen.config, 'fid_interval', 0)
        self.best_fid = float('inf')

        # Gradient clipping
        self.grad_clip_norm = getattr(self.gen.config, 'grad_clip_norm', 0.0)

        # Fixed seed for consistent visual tracking across batches
        self._tracking_seed = tf.random.normal([self.n_samples, self.gen.config.latent_dim])

        # Adaptive disc/gen step ratio
        self.adaptive_steps = getattr(self.gen.config, 'adaptive_steps', False)
        self._base_disc_steps = self.disc.config.training_steps
        self._disc_loss_ema = 0.0
        self._gen_loss_ema = 0.0

        # Adaptive augmentation (ADA) — adjust augment probability based on disc overfitting
        self.ada_target = getattr(self.gen.config, 'ada_target', 0.0)
        self.augment_p = 0.5 if self.use_augment else 0.0

        # Multi-scale discriminator
        self.multiscale_disc = getattr(self.disc.config, 'multiscale_disc', False)
        self.disc_lowres = None
        self.disc_lowres_optimizer = None
        if self.multiscale_disc:
            self._build_lowres_disc()
            lowres_dir = os.path.dirname(self.disc.config.checkpoint)
            lowres_declared = os.path.join(lowres_dir, "discriminator_lowres.weights.h5")
            lowres_resolved = resolve_weights_path(lowres_declared)
            if lowres_resolved is not None:
                self.disc_lowres.load_weights(lowres_resolved)
                print(f"Low-res discriminator loaded from {lowres_resolved}")

    def _init_ema(self):
        """Initialize EMA shadow weights as a copy of current generator weights on CPU to save VRAM."""
        with tf.device('/CPU:0'):
            self.ema_weights = [tf.Variable(w, trainable=False, name=f"ema/{w.name}")
                                for w in self.gen.model.trainable_variables]
        # Load persisted EMA weights if available
        ema_path = os.path.join(os.path.dirname(self.gen.config.checkpoint), "generator_ema.weights.h5")
        if os.path.exists(ema_path):
            try:
                # Build a temporary model clone to load weights
                tmp_model = keras.models.clone_model(self.gen.model)
                tmp_model.build(self.gen.model.input_shape)
                tmp_model.load_weights(ema_path)
                for ema_var, tmp_var in zip(self.ema_weights, tmp_model.trainable_variables):
                    ema_var.assign(tmp_var)
                del tmp_model
                print(f"EMA weights loaded from {ema_path}")
            except Exception as e:
                print(f"Warning: could not load EMA weights: {e}. Using current generator weights.")

    def _update_ema(self):
        """Update EMA shadow weights with current generator weights."""
        if self.ema_weights is None:
            return
        decay = self.ema_decay
        for ema_var, model_var in zip(self.ema_weights, self.gen.model.trainable_variables):
            ema_var.assign(decay * ema_var + (1.0 - decay) * model_var)

    def _apply_ema_to_generator(self):
        """Swap EMA weights into the generator (for generation/eval)."""
        if self.ema_weights is None:
            return None
        # Save current weights so we can restore later
        backup = [tf.identity(w) for w in self.gen.model.trainable_variables]
        for model_var, ema_var in zip(self.gen.model.trainable_variables, self.ema_weights):
            model_var.assign(ema_var)
        return backup

    def _restore_generator_weights(self, backup):
        """Restore generator weights from a backup (after EMA swap)."""
        if backup is None:
            return
        for model_var, bak in zip(self.gen.model.trainable_variables, backup):
            model_var.assign(bak)

    def _update_learning_rates(self):
        """Apply cosine annealing LR decay based on post-fade step count."""
        if self.lr_decay != "cosine":
            return
        # Only decay after fade completes; use steps since fade ended
        if not self.fade_complete:
            return
        post_fade_step = self.global_step - self.fade_steps
        if post_fade_step < 0:
            post_fade_step = 0
        # Horizon is config-driven. If unset, fall back to a long horizon (and
        # warn once) rather than the old hard-coded 200k, which floored both LRs
        # at lr_min ~70% into this run and froze learning (the post-250k drift).
        if self.lr_decay_steps > 0:
            decay_steps = self.lr_decay_steps
        else:
            decay_steps = 1_000_000
            if not self._lr_horizon_warned:
                print(
                    "WARNING: lr_decay='cosine' with lr_decay_steps unset; "
                    f"falling back to {decay_steps} steps. Set --lr_decay_steps "
                    "to the planned run length so the schedule anneals over the "
                    "real horizon."
                )
                self._lr_horizon_warned = True

        gen_lr = cosine_decayed_lr(self.gen_lr_base, self.lr_min, post_fade_step, decay_steps)
        disc_lr = cosine_decayed_lr(self.disc_lr_base, self.lr_min, post_fade_step, decay_steps)

        self.gen.optimizer.learning_rate.assign(gen_lr)
        self.disc.optimizer.learning_rate.assign(disc_lr)

    def _build_lowres_disc(self):
        """Build a lightweight discriminator for 256x256 input (multi-scale feedback)."""
        use_sn = getattr(self.disc.config, 'spectral_norm', False)
        inputs = keras.Input(shape=(256, 256, 3))
        x = inputs
        for filters in [64, 128, 256]:
            conv = keras.layers.Conv2D(filters, 3, strides=2, padding='same')
            x = keras.layers.SpectralNormalization(conv)(x) if use_sn else conv(x)
            x = keras.layers.LeakyReLU(negative_slope=0.25)(x)
        x = keras.layers.Flatten()(x)
        # Pin scalar critic output to float32 under mixed precision (UPGRADES #15).
        dense = keras.layers.Dense(1, dtype="float32")
        outputs = keras.layers.SpectralNormalization(dense)(x) if use_sn else dense(x)
        self.disc_lowres = keras.Model(inputs, outputs, name="DiscriminatorLowRes")
        self.disc_lowres_optimizer = keras.optimizers.Adam(
            learning_rate=self.disc.config.learning_rate,
            beta_1=self.disc.config.beta_1,
            beta_2=self.disc.config.beta_2
        )
        # Wrap in LossScaleOptimizer under mixed_float16 so the lowres
        # discriminator's fp16 gradients don't underflow (UPGRADES #15).
        if keras.mixed_precision.global_policy().name == "mixed_float16":
            self.disc_lowres_optimizer = keras.mixed_precision.LossScaleOptimizer(self.disc_lowres_optimizer)
        print(f"Multi-scale discriminator built (256x256, {self.disc_lowres.count_params()} params)")

    def _update_adaptive_steps(self, disc_loss, gen_loss):
        """Adjust disc training steps based on loss balance."""
        if not self.adaptive_steps:
            return
        # Exponential moving average of losses
        alpha = 0.05
        self._disc_loss_ema = alpha * abs(float(disc_loss)) + (1 - alpha) * self._disc_loss_ema
        self._gen_loss_ema = alpha * abs(float(gen_loss)) + (1 - alpha) * self._gen_loss_ema
        total = self._disc_loss_ema + self._gen_loss_ema + 1e-8
        disc_ratio = self._disc_loss_ema / total

        # Adjust every 100 steps
        if self.global_step % 100 == 0 and self.global_step > 0:
            max_steps = self._base_disc_steps * 2
            if disc_ratio < 0.3:
                # Disc too weak — give it more steps
                self.disc.config.training_steps = min(self.disc.config.training_steps + 1, max_steps)
                print(f"Adaptive steps: disc weak (ratio={disc_ratio:.2f}), disc_steps -> {self.disc.config.training_steps}")
            elif disc_ratio > 0.7:
                # Disc too strong — reduce its steps
                self.disc.config.training_steps = max(self.disc.config.training_steps - 1, 1)
                print(f"Adaptive steps: disc strong (ratio={disc_ratio:.2f}), disc_steps -> {self.disc.config.training_steps}")

    def _update_ada(self, real_scores):
        """Adjust augmentation probability based on discriminator overfitting (ADA)."""
        if self.ada_target <= 0:
            return
        # rt = fraction of real images the disc correctly identifies (score > 0 in WGAN)
        rt = float(tf.reduce_mean(tf.cast(real_scores > 0, tf.float32)))
        # Adjust p every 50 steps
        if self.global_step % 50 == 0 and self.global_step > 0:
            if rt > self.ada_target:
                # Disc overfitting — increase augmentation
                self.augment_p = min(self.augment_p + 0.01, 0.8)
            else:
                # Disc not overfitting — decrease augmentation
                self.augment_p = max(self.augment_p - 0.01, 0.0)

    def _compute_fid(self, num_samples=64):
        """
        Compute a lightweight FID approximation using InceptionV3 features.
        Compares real dataset samples against generated samples.
        Returns FID score or None if computation fails.
        """
        try:
            from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

            # Load inception model on demand — do NOT cache in VRAM permanently
            inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

            def _flatten_depth(x):
                # Inception expects 4-D (N, H, W, 3); collapse the depth axis
                # into batch so each modality slice scores independently.
                if len(x.shape) == 5:
                    s = tf.shape(x)
                    x = tf.reshape(x, [s[0] * s[1], s[2], s[3], s[4]])
                return x

            def get_features_batched(images, batch_size=4):
                """Extract features in small batches to avoid OOM at high resolutions."""
                all_features = []
                for i in range(0, len(images), batch_size):
                    chunk = images[i:i + batch_size]
                    chunk = _flatten_depth(chunk)
                    chunk = tf.image.resize(chunk, (299, 299))
                    chunk = (chunk + 1.0) * 127.5
                    chunk = preprocess_input(chunk)
                    feats = inception_model(chunk, training=False)
                    all_features.append(feats.numpy())
                return np.concatenate(all_features, axis=0)

            # Generate fake samples and extract features per-chunk to avoid
            # holding all full-resolution images in memory simultaneously.
            backup = self._apply_ema_to_generator()
            fake_features_list = []
            gen_batch = 4
            for i in range(0, num_samples, gen_batch):
                n = min(gen_batch, num_samples - i)
                noise = tf.random.normal([n, self.gen.config.latent_dim])
                chunk = self.gen.model(noise, training=False)
                chunk = _flatten_depth(chunk)
                chunk = tf.image.resize(chunk, (299, 299))
                chunk = (chunk + 1.0) * 127.5
                chunk = preprocess_input(chunk)
                fake_features_list.append(inception_model(chunk, training=False).numpy())
            self._restore_generator_weights(backup)

            # Get real samples WITHOUT advancing the training pointer
            saved_ind = self.gen.config.train_ind
            real_images = self.dataset.batch(num_samples, 'magnified_profile')
            self.gen.config.train_ind = saved_ind  # Restore pointer
            if real_images is None:
                return None

            fake_features = np.concatenate(fake_features_list, axis=0)
            real_features = get_features_batched(real_images)

            # Compute FID: ||mu_r - mu_f||^2 + Tr(C_r + C_f - 2*sqrt(C_r @ C_f))
            mu_r, mu_f = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
            sigma_r = np.cov(real_features, rowvar=False)
            sigma_f = np.cov(fake_features, rowvar=False)

            diff = mu_r - mu_f
            # Stable matrix sqrt via eigendecomposition
            product = sigma_r @ sigma_f
            eigenvalues, _ = np.linalg.eigh(product)
            eigenvalues = np.maximum(eigenvalues, 0)
            sqrt_trace = np.sum(np.sqrt(eigenvalues))

            fid = float(np.dot(diff, diff) + np.trace(sigma_r) + np.trace(sigma_f) - 2 * sqrt_trace)

            # Free InceptionV3 from VRAM — it's only needed during FID computation
            del inception_model

            return fid
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")
            return None

    def train(self, batch_size = 8, epochs = 1):
        """
        Initializes training the discriminator and generator based on requested
        runtime behavioral. The model is saved after every training batch to preserve
        training history. 
        
        Function arguments:
            batches (int or None) - Number of training batches to learn from before stopping
            batch_size (int) - Number of images to load into each training batch
            epochs (int or None) - Number of epochs to train per batch, defaults to class default
        """

        # Update hyperparameters if passed in before training
        if batch_size: self.batch_size = batch_size

        start_epoch = int(getattr(self.gen.config, "current_epoch", 0) or 0)
        # Iterate through requested training batches
        for epoch in range(start_epoch, start_epoch + epochs):
            # New epoch restart: allow profiles to be reused next pass
            self.dataset.reset_seen_profiles()
            self.gen.config.current_epoch = epoch
            self.disc.config.current_epoch = epoch
            batch = 1

            # Resume the batch counter from the highest existing snapshot dir
            # so a mid-epoch crash doesn't restart at batch 1 and overwrite
            # earlier snapshots. (UPGRADES #36 plans to persist this in
            # config; until then, snapshot dirs are the durable source.)
            for snapshot in glob(f"{self.gen.config.save_dir}/batch_*"):
                if not os.path.isdir(snapshot):
                    continue
                snapshot_n = self._extract_batch_number(snapshot)
                if snapshot_n is not None and snapshot_n >= batch:
                    batch = snapshot_n + 1

            trainable_data = True
            while trainable_data:
                # Compute the memtrace cadence up front so the data load gets
                # bracketed too: uord marks span every phase of the real loop
                # (dataload -> train phases -> plot), and the reliable live-arena
                # delta per phase tells us which one leaks.
                _mt_due = self._memtrace is not None and self._memtrace.due(batch)
                if _mt_due:
                    self._memtrace.note("dataload")
                # Load a new batch of subjects via the modality-aware
                # dispatcher. config.modality determines whether this returns
                # depth-1 single-modality samples or depth-2 merged stacks.
                print(f"Grab a batch of data")
                x = self.dataset.next_batch(batch_size)
                if x is None:
                    self.gen.config.train_ind = 0
                    x = self.dataset.next_batch(batch_size)

                print(f"Data batched")
                if x is None:
                    print(f"Training Epoch Complete")
                    trainable_data = False
                    continue

                # Sync discriminator settings with any update the dataset applied to generator
                self.disc.config.channels = self.gen.config.channels

                # Ensure models match the incoming depth — should always pass
                # since pair_depth is fixed at construction; mismatch indicates
                # an upstream contract violation (UPGRADES #2 + #40).
                self._ensure_depth_alignment(x.shape[1])

                print(f"Training on batch {batch}...")

                # train_step lays disc_loop / gen_loop / poststep / end marks.
                if _mt_due:
                    self._memtrace.note("trainstep")
                self.train_step(x, trace=_mt_due) # Train on batch of images
                # Drop the input batch reference before printing RSS so the
                # number reflects steady-state memory between batches.
                del x
                # Force a GC sweep to break any reference cycles closures may
                # create. Cheap (microseconds) and isolates "creeping" leaks
                # caused by cycles vs. true unfreed allocations.
                if batch % 10 == 0:
                    gc.collect()
                rss_mb = _process_rss_mb()
                print(f'Epoch {epoch} | Batch {batch} | Generator loss: {round(float(self.loss["gen"][-1]), 3)} | Discrimintator loss: {round(float(self.loss["disc"][-1]), 3)} | RSS: {rss_mb:.0f} MiB')


                # Plot training curves on a fixed batch cadence to avoid I/O storm
                if batch % 10 == 0:
                    self.plot_history()
                # Native-heap snapshot taken AFTER plot_history so its allocation
                # is captured in this tick. Mark the plot phase, then drain all
                # phase marks (dataload / train phases / plot) into the snapshot.
                if _mt_due:
                    self._memtrace.note("plot")
                    self._memtrace.tick(batch)

                # FID-based best model checkpointing
                if self.fid_interval > 0 and self.global_step % self.fid_interval == 0:
                    fid = self._compute_fid()
                    if fid is not None:
                        print(f"FID @ step {self.global_step}: {fid:.2f} (best: {self.best_fid:.2f})")
                        if fid < self.best_fid:
                            self.best_fid = fid
                            best_dir = os.path.join(self.save_dir, "best_fid/")
                            self.save_model(best_dir)
                            print(f"New best FID! Model saved to {best_dir}")

                # Save the models state
                if batch % self._save_interval == 0:
                    # Log the live LR so a floored/over-decayed schedule is
                    # visible in the run log instead of silently freezing
                    # learning (the post-250k destabilization had no such signal).
                    if self.lr_decay == "cosine":
                        gen_lr = float(self.gen.optimizer.learning_rate.numpy())
                        disc_lr = float(self.disc.optimizer.learning_rate.numpy())
                        print(f"LR @ step {self.global_step}: gen={gen_lr:.2e} disc={disc_lr:.2e}")
                    self.save_model(f"{self.save_dir}/batch_{batch}/")
                    if self.cleanup_milestone > 0 and batch % self.cleanup_milestone == 0:
                        self._cleanup_saved_batches(100)

                # Per-batch seeded preview emission. Off by default
                # (sample_batch_interval=0); when enabled it mirrors the
                # epoch-end block below so a run that never closes an
                # epoch still leaves a visible trail in synthetic_images/.
                # Filename prefix matches the historical naming so
                # _cleanup_saved_batches' glob still trims them.
                batch_sample_interval = int(getattr(self.gen.config, "sample_batch_interval", 0) or 0)
                if self._should_emit_batch_sample(batch, batch_sample_interval, self.n_samples):
                    backup = self._apply_ema_to_generator()
                    try:
                        generate(
                            self.gen,
                            count=self.n_samples,
                            seed_size=self.gen.config.latent_dim,
                            save_dir=f"{self.save_dir}/synthetic_images/",
                            filename_prefix=f"batch_{batch}_synthetic",
                            seed=self._tracking_seed,
                        )
                    finally:
                        self._restore_generator_weights(backup)

                batch += 1
            # Mark epoch completion for persistence
            self.gen.config.current_epoch = epoch + 1
            self.disc.config.current_epoch = epoch + 1

            # Generate seeded preview samples at epoch end so progress is
            # visible across either modality. Cadence and N are config-driven
            # (sample_epoch_interval, n_samples). Skipped when the interval
            # is 0 or n_samples is 0.
            sample_interval = int(getattr(self.gen.config, "sample_epoch_interval", 1) or 0)
            if sample_interval > 0 and self.n_samples and (epoch + 1) % sample_interval == 0:
                backup = self._apply_ema_to_generator()
                try:
                    generate(
                        self.gen,
                        count=self.n_samples,
                        seed_size=self.gen.config.latent_dim,
                        save_dir=f"{self.save_dir}/synthetic_images/",
                        filename_prefix=f"epoch_{epoch + 1}_synthetic",
                        seed=self._tracking_seed,
                    )
                finally:
                    self._restore_generator_weights(backup)

        # Flush the diagnostic trace on graceful exit. (An OOM-kill won't reach
        # here, but the JSONL is line-buffered so no records are lost either way.)
        if self._memtrace is not None:
            self._memtrace.close()


    def train_step(self, images, disc_steps = None, gen_steps = None, trace = False):
        """
        This function trains the snowGAN on the passed in images, with variables 
        training length of the discriminator and generator. This function implements
        WGAN-GP unique EMD loss and gradient penalty regularization.
        
        Function arguments:
            images (4D numpy array) - numpy array of RGB images set to the class resolution (sample axis = 0)
            disc_steps (int) - Number N of training steps for the discriminator per epoch
            gen_steps (int) - Number M of training steps for the generator per epoch
        """
        # Update training parameters if passed in
        if disc_steps:
            self.disc.config.training_steps = disc_steps
        if gen_steps:
            self.gen.config.training_steps = gen_steps

        # Set batch size to the number of images passed in
        batch_size = tf.shape(images)[0] 

        # Generate noise for generator input
        noise = tf.random.normal([batch_size, self.gen.config.latent_dim])
        print(f"Noise shape: {noise.shape}")
        
        # Current augmentation probability (ADA adjusts this dynamically)
        aug_p = self.augment_p

        # Train the discriminator N times. Accumulate per-iteration losses
        # so the recorded train_step datapoint reflects the mean across all
        # inner updates rather than only the final iteration's value
        # (UPGRADES #33). Empty-list guard handles training_steps=0.
        if trace and self._memtrace is not None:
            self._memtrace.note("disc_loop")
        disc_losses: list[float] = []
        real_scores = None
        for _disc_i in range(self.disc.config.training_steps):
            # Finer uord marks on the FIRST disc iteration only (avoids 3x
            # overwrite); bisects which sub-op of the disc loop leaks.
            _mt0 = self._memtrace if (trace and self._memtrace is not None and _disc_i == 0) else None
            # Generate synthetic images outside the tape — no need to track generator ops for disc training
            synthetic_images = tf.stop_gradient(self._generate_with_fade(noise, training=True))
            if _mt0: _mt0.note("d_synth")

            # Apply augmentation outside the tape to avoid storing intermediate tensors in memory
            disc_real = diff_augment(images, p=aug_p) if self.use_augment else images
            disc_fake = diff_augment(synthetic_images, p=aug_p) if self.use_augment else synthetic_images
            if _mt0: _mt0.note("d_augment")

            # Main discriminator tape — separate from lowres to reduce peak VRAM
            with tf.GradientTape() as tape:
                # Forward propogate real & synethic images to the discriminator
                output = self.disc.model(disc_real, training=True)
                synthetic_output = self.disc.model(disc_fake, training=True)
                real_scores = output  # Track for ADA

                # Gradient penalty. Skipped entirely when lambda_gp == 0: with
                # spectral norm enforcing the Lipschitz constraint structurally,
                # a GP on top double-constrains the critic (the SN+GP over-
                # regularization the audit flagged). When active, compute it on
                # the SAME augmented tensors the critic is scored on, so the
                # 1-Lipschitz constraint is enforced on the distribution the
                # critic actually sees (DiffAugment recipe), not the raw manifold.
                lambda_gp = self.disc.config.lambda_gp or 0.0
                if lambda_gp > 0:
                    gp = compute_gradient_penalty(self.disc, disc_real, disc_fake)
                else:
                    gp = tf.constant(0.0, dtype=tf.float32)

                # Calculate EMD/loss for the discriminators outputs
                disc_loss = self.disc.get_loss(output, synthetic_output, gp, lambda_gp)

            if _mt0: _mt0.note("d_forward")
            # Backpropogate main discriminator
            disc_gradients = tape.gradient(disc_loss, self.disc.model.trainable_variables)
            if _mt0: _mt0.note("d_grad")
            if self.grad_clip_norm > 0:
                disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, self.grad_clip_norm)
            self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))
            if _mt0: _mt0.note("d_apply")

            # Free main disc tape activations before lowres pass
            del tape, disc_gradients

            # Multi-scale discriminator in its own tape (separate VRAM peak)
            if self.disc_lowres is not None:
                # disc_lowres takes 4-D (N, 256, 256, 3); collapse the
                # depth axis into the batch so each modality slice scores
                # independently against the low-res critic.
                def _to_lowres(x):
                    if len(x.shape) == 5:
                        s = tf.shape(x)
                        x = tf.reshape(x, [s[0] * s[1], s[2], s[3], s[4]])
                    return tf.image.resize(x, (256, 256))
                real_lowres = _to_lowres(images)
                fake_lowres = _to_lowres(synthetic_images)
                with tf.GradientTape() as lr_tape:
                    real_out_lr = self.disc_lowres(real_lowres, training=True)
                    fake_out_lr = self.disc_lowres(fake_lowres, training=True)
                    disc_loss_lr = tf.reduce_mean(tf.cast(fake_out_lr, tf.float32)) - tf.reduce_mean(tf.cast(real_out_lr, tf.float32))
                lowres_grads = lr_tape.gradient(disc_loss_lr, self.disc_lowres.trainable_variables)
                if self.grad_clip_norm > 0:
                    lowres_grads, _ = tf.clip_by_global_norm(lowres_grads, self.grad_clip_norm)
                self.disc_lowres_optimizer.apply_gradients(zip(lowres_grads, self.disc_lowres.trainable_variables))
                disc_loss = disc_loss + 0.5 * disc_loss_lr  # For logging
                del lr_tape, lowres_grads, real_lowres, fake_lowres

            disc_losses.append(float(disc_loss))

            # Free augmented copies and per-iter tensors before next iter / gen training
            del disc_real, disc_fake, output, synthetic_output, gp, disc_loss, synthetic_images

        # Train the generator M times. Same accumulation pattern as the
        # discriminator loop above (UPGRADES #33).
        if trace and self._memtrace is not None:
            self._memtrace.note("gen_loop")
        gen_losses: list[float] = []
        for _ in range(self.gen.config.training_steps):
            # Generate random noise to pass into the generator
            noise = tf.random.normal([batch_size, self.gen.config.latent_dim])
            # Initialize automtic differentiation during forward propogation
            with tf.GradientTape() as tape:
                # Forward pass noise through the generator to create synthetic images
                use_fade = self._use_fade()
                synthetic_images = self._generate_with_fade(noise, training=True)
                # Forward pass generator outputs through the discriminator for loss calculation
                synthetic_output = self.disc.model(synthetic_images, training=True)
                # Calculate generator loss for backpropogation by calculating mean of disc output
                gen_loss = self.gen.get_loss(synthetic_output)
            # Backpropogate to calculate gradient of trainable parameters given loss
            var_list = list(self.gen.model.trainable_variables)
            # If fading, also update fade endpoint layers (e.g., toRGB_prev)
            if use_fade and hasattr(self.gen, 'fade_endpoints') and self.gen.fade_endpoints is not None:
                # Extend with unique vars from fade_endpoints by variable name (TF variables lack ref())
                existing_names = {v.name for v in var_list}
                fade_vars = [v for v in self.gen.fade_endpoints.trainable_variables if v.name not in existing_names]
                var_list.extend(fade_vars)

            gen_gradients = tape.gradient(gen_loss, var_list)
            # Apply gradient clipping if configured
            if self.grad_clip_norm > 0:
                gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, self.grad_clip_norm)
            # Apply the gradients and update trainable parameters (w, b, etc.)
            self.gen.optimizer.apply_gradients(zip(gen_gradients, var_list))

            gen_losses.append(float(gen_loss))
            # Drop per-iter tensors so refs don't pile up across the inner loop.
            del tape, gen_gradients, var_list, synthetic_images, synthetic_output, gen_loss, noise

        if trace and self._memtrace is not None:
            self._memtrace.note("poststep")
        # Record mean loss across the inner update loops. Using means makes
        # the curve faithful to all training_steps iterations and stabilizes
        # the EMA in _update_adaptive_steps below.
        mean_disc_loss = self._mean_loss(disc_losses)
        mean_gen_loss = self._mean_loss(gen_losses)
        self.loss['gen'].append(mean_gen_loss)
        self.loss['disc'].append(mean_disc_loss)
        # Increment global step after a full train step
        self.global_step += 1
        self._update_fade_completion()
        # Persist fade progress across generator/discriminator configs for resume support
        # Throttle disk writes to every 50 steps (atexit save_model covers final state)
        self._sync_fade_progress(persist=(self.global_step % 50 == 0))

        # Post-step improvements
        self._update_ema()
        self._update_learning_rates()
        self._update_adaptive_steps(mean_disc_loss, mean_gen_loss)
        if real_scores is not None:
            self._update_ada(real_scores)
        if trace and self._memtrace is not None:
            self._memtrace.note("end")

    def _use_fade(self):
        return getattr(self.gen.config, 'fade', False) and getattr(self.gen, 'fade_endpoints', None) is not None and not self.fade_complete

    def _generate_with_fade(self, noise, training=True):
        if self._use_fade():
            fade_total = self.fade_steps
            fade_progress = min(self.global_step, fade_total)
            alpha = compute_fade_alpha(fade_progress, fade_total)
            prev_up, curr_img = self.gen.fade_endpoints(noise, training=training)
            alpha = tf.cast(alpha, curr_img.dtype)
            return (1.0 - alpha) * prev_up + alpha * curr_img
        return self.gen.model(noise, training=training)

    def _update_fade_completion(self):
        if not self.fade_complete and self.global_step >= self.fade_steps:
            self.fade_complete = True

    def _ensure_depth_alignment(self, depth):
        """Hard-assert batch depth matches the depth the trainer was built at.

        The previous behavior was to silently rebuild gen/disc on mismatch,
        which discarded any trained weights (UPGRADES #2) and could fire
        on every batch under a mixed-depth manifest (UPGRADES #40). The
        rebuild has moved to ``__init__`` (one-time, before weight load);
        from this point on, the dataset's ``DataManager.PAIR_DEPTH`` is
        the single source of truth and any per-batch deviation indicates
        an upstream contract violation we should surface immediately.
        """
        if depth is None:
            return
        desired = int(depth)
        if desired != self._built_depth:
            raise RuntimeError(
                f"Batch depth {desired} does not match the trainer's built "
                f"depth {self._built_depth}. The dataset/model depth contract "
                f"was violated upstream of train_step — check that "
                f"DataManager.PAIR_DEPTH matches the data path's actual "
                f"output depth."
            )

    def save_model(self, path = None):
        """
        Save the currently loaded generator and discriminator to a given path

        Function arguments:
            path (str) - Folder to save the generator and discriminator .keras files
        """
        # Update path if none passed in
        if path == None:
            path = self.save_dir
        
        os.makedirs(path, exist_ok = True)
        # Persist configs alongside weights for reproducible snapshots. The
        # config JSONs are the architecture sidecar — downstream consumers
        # (e.g. AvAI) reconstruct the model via Generator(config) /
        # Discriminator(config) and then load_weights from the *.weights.h5
        # files written below. See docs/UPGRADES.md #17.
        self._save_configs(path)

        # Save weights-only (no architecture, no Lambda layer serialization).
        # Weights are the most important artifact and must not be blocked by
        # a plotting failure.
        self.gen.model.save_weights(f"{path}/generator.weights.h5")
        self.disc.model.save_weights(f"{path}/discriminator.weights.h5")

        # Log and plot history (non-critical — failures here won't lose weights)
        try:
            save_history(self.gen.config.save_dir, self.loss, self.trained_data)
            self.plot_history()
        except Exception as e:
            print(f"Warning: failed to save/plot history: {e}")
        # Save fade endpoints weights so mid-fade resume preserves toRGB_prev
        if getattr(self.gen, 'fade_endpoints', None) is not None:
            self.gen.fade_endpoints.save_weights(f"{path}/generator_fade_endpoints.weights.h5")
        # Save multi-scale discriminator weights
        if self.disc_lowres is not None:
            self.disc_lowres.save_weights(f"{path}/discriminator_lowres.weights.h5")
        # Save EMA shadow weights for resume
        if self.ema_weights is not None:
            backup = self._apply_ema_to_generator()
            self.gen.model.save_weights(f"{path}/generator_ema.weights.h5")
            self._restore_generator_weights(backup)
        print(f"Models saved in {path}...")

    def plot_history(self):
        # Check if save folder exists yet
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir, exist_ok = True)
        # Reuse ONE persistent figure across calls. Creating a fresh figure
        # every call leaks CPU RAM in the Agg backend even with plt.close(fig)
        # — an isolation test showed RSS climbing 61->1867 MiB over 300 calls
        # with per-call figures, but flat at 189 MiB when the figure is reused
        # and the axes are cleared. Since plot_history runs every 10 batches
        # over a long run, that per-figure leak was a multi-GB CPU-RAM source.
        # The loss lists are already plain Python floats (from load_history /
        # _mean_loss), so passing them directly avoids a large copy.
        if self._history_fig is None:
            self._history_fig, self._history_ax = plt.subplots()
        ax = self._history_ax
        ax.cla()
        ax.plot(self.loss['gen'], label='Generator loss')
        ax.plot(self.loss['disc'], label='Discriminator loss')
        ax.set_title("GAN History")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        self._history_fig.savefig(os.path.join(self.save_dir, 'history.png'))

    @staticmethod
    def _should_emit_batch_sample(batch: int, interval: int, n_samples: int) -> bool:
        """Predicate for the per-batch seeded-preview emission.

        Pulled out of the train loop so it has a single, testable definition.
        Off when interval<=0 or n_samples<=0; otherwise fires on multiples of
        ``interval``. ``batch`` is 1-indexed in the train loop, so batch=1 with
        interval=1 emits on the first batch (matches the existing checkpoint
        cadence semantics).
        """
        if interval <= 0:
            return False
        if not n_samples or n_samples <= 0:
            return False
        return batch % interval == 0

    @staticmethod
    def _mean_loss(losses: list[float]) -> float:
        """Mean of a list of per-iteration losses, or 0.0 if empty.

        UPGRADES #33: train_step records one curve point per call. Each
        point should reflect the mean across all inner update iterations
        rather than only the final iteration's value, so curves stay
        faithful when training_steps > 1. Empty-list case (training_steps=0)
        returns 0.0 so the appended value is always defined.
        """
        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    @staticmethod
    def _extract_batch_number(path: str):
        basename = os.path.basename(os.path.normpath(path))
        match = re.search(r"batch_(\d+)", basename)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _cleanup_saved_batches(self, keep_every: int):
        """
        Remove checkpoint directories for batches that do not align with the cleanup milestone
        and trim only the seven most recent synthetic samples for those batches.
        """
        if keep_every <= 0:
            return

        removed_checkpoints = 0
        trimmed_images = 0
        synthetic_dir = os.path.join(self.save_dir, "synthetic_images")

        for checkpoint_path in glob(os.path.join(self.save_dir, "batch_*")):
            if not os.path.isdir(checkpoint_path):
                continue
            batch_number = self._extract_batch_number(checkpoint_path)
            if batch_number is None or batch_number % keep_every == 0:
                continue

            shutil.rmtree(checkpoint_path, ignore_errors=True)
            removed_checkpoints += 1

            if os.path.isdir(synthetic_dir):
                pattern = os.path.join(synthetic_dir, f"batch_{batch_number}_synthetic_*.png")
                indexed_images = []
                for image_path in glob(pattern):
                    stem = os.path.splitext(os.path.basename(image_path))[0]
                    try:
                        image_index = int(stem.split("_")[-1])
                    except (ValueError, IndexError):
                        continue
                    indexed_images.append((image_index, image_path))
                indexed_images.sort(key=lambda item: item[0])
                for _, image_path in indexed_images[-7:]:
                    try:
                        os.remove(image_path)
                        trimmed_images += 1
                    except (FileNotFoundError, ValueError):
                        continue

        if removed_checkpoints or trimmed_images:
            print(
                f"Cleanup milestone reached (keep every {keep_every} batches): "
                f"removed {removed_checkpoints} checkpoint directories, trimmed {trimmed_images} synthetic images."
            )

    def _save_configs(self, path: str):
        """
        Save generator and discriminator configs to the given snapshot path without
        permanently changing their primary config file targets.
        """
        gen_cfg_path = os.path.join(path, "generator_config.json")
        disc_cfg_path = os.path.join(path, "discriminator_config.json")

        # Preserve original config destinations so subsequent saves still hit the main paths
        gen_orig = getattr(self.gen.config, "config_filepath", None)
        disc_orig = getattr(self.disc.config, "config_filepath", None)
        try:
            self.gen.config.save_config(gen_cfg_path)
            self.disc.config.save_config(disc_cfg_path)
        except Exception as e:
            print(f"Warning: failed to save configs in {path}: {e}")
        finally:
            if gen_orig is not None:
                self.gen.config.config_filepath = gen_orig
            if disc_orig is not None:
                self.disc.config.config_filepath = disc_orig

    def _persist_initial_splits(self):
        """Derive train/val/test splits and persist them to both configs.

        ``DataManager.derive_splits`` only mutates the config it was constructed
        with (``self.gen.config``). AvAI's evaluation reads
        ``discriminator_config.json`` for the test pool, so the three pools must
        be mirrored onto ``self.disc.config`` explicitly — otherwise the
        periodic save in ``_sync_fade_progress`` keeps dumping the still-null
        fields on the discriminator side and Phase 4b stays blocked despite the
        splits being derived correctly upstream.

        ``previously_split`` requires *both* configs to already have a populated
        ``test_pool``; if either side is null (the torn state caused by the
        original PR #9 bug), the immediate save fires to repair the file.
        """
        previously_split = (
            self.gen.config.test_pool is not None
            and self.disc.config.test_pool is not None
        )
        self.dataset.derive_splits()
        self.disc.config.trained_pool = self.gen.config.trained_pool
        self.disc.config.validation_pool = self.gen.config.validation_pool
        self.disc.config.test_pool = self.gen.config.test_pool
        if not previously_split:
            try:
                self.gen.config.save_config()
                self.disc.config.save_config()
            except Exception as e:
                print(f"Warning: failed to persist derived data splits: {e}")

    def _sync_fade_progress(self, persist=True):
        """
        Mirror fade progress/targets to both generator and discriminator configs so training can resume seamlessly.
        """
        target_step = int(self.global_step)
        gen_cfg = getattr(self.gen, 'config', None)
        disc_cfg = getattr(self.disc, 'config', None)

        for cfg, label in ((gen_cfg, "generator"), (disc_cfg, "discriminator")):
            if cfg is None:
                continue
            try:
                cfg.fade_step = target_step
                if gen_cfg is not None and cfg is not gen_cfg and hasattr(cfg, 'fade_steps') and hasattr(gen_cfg, 'fade_steps'):
                    if cfg.fade_steps != gen_cfg.fade_steps:
                        cfg.fade_steps = gen_cfg.fade_steps
                if persist:
                    cfg.save_config()
            except Exception as e:
                print(f"Warning: failed to persist fade configuration for {label}: {e}")


