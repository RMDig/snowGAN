import os
import random
from functools import cached_property

from datasets import load_dataset
from PIL import Image
import tensorflow as tf
import numpy as np

from snowgan.modality import Modality


_SINGLE_MODALITIES = {"core", "profile", "magnified_profile", "crystal_card"}
_MERGED_MODALITY = "merged"

# Canonical datatype encoding. The rmdig manifest and the modality names share
# these strings; both normalize to the same ints, which are the representation
# the rest of the data layer (pair_index, batch, batch_merged) and downstream
# consumers (e.g. snowGradient reading dm.manifest) are written against.
DATATYPE_TO_INT = {
    "core": 0,
    "profile": 1,
    "magnified_profile": 2,
    "crystal_card": 3,
}
_VALID_DATATYPE_INTS = set(DATATYPE_TO_INT.values())


def normalize_datatype(value):
    """Map a datatype to its canonical int (core=0, profile=1,
    magnified_profile=2, crystal_card=3).

    Accepts the rmdig manifest string, the equivalent modality name, or an
    already-normalized int — the int case is idempotent so re-normalizing is
    safe (e.g. a caller passing ``2`` instead of ``"magnified_profile"``).

    Raises ``ValueError`` on any other value. An unknown datatype is a schema /
    contract error and must fail loudly — never be silently skipped. A silent
    skip is what made rmdig's 2026-08-17 int->string ``datatype`` change look
    like an empty ``pair_index`` when the manifest JSONL was read directly
    (bypassing HF's ClassLabel cast); via ``load_dataset`` the cast already
    yields ints, so this normalization is defensive, not a bug fix
    (docs/UPGRADES.md #50).
    """
    # bool is a subclass of int; reject it so True/False can't masquerade as 0/1.
    if isinstance(value, bool):
        raise ValueError(f"Invalid datatype {value!r}: bool is not a datatype.")
    if isinstance(value, (int, np.integer)):
        as_int = int(value)
        if as_int in _VALID_DATATYPE_INTS:
            return as_int
        raise ValueError(
            f"Unknown datatype int {as_int!r}; expected one of "
            f"{sorted(_VALID_DATATYPE_INTS)}."
        )
    if value in DATATYPE_TO_INT:
        return DATATYPE_TO_INT[value]
    raise ValueError(
        f"Unknown datatype {value!r}; expected one of {sorted(DATATYPE_TO_INT)} "
        f"or ints {sorted(_VALID_DATATYPE_INTS)}."
    )


# --- Blue measurement-board masking (core modality) ---------------------------
# Core photos are a snow sample on a blue ruler board; the board + ruler + printed
# "Centimeters" text dominate every frame and are irrelevant (confounding) for the
# downstream avalanche-risk task. A GAN trained on the raw frames collapses onto
# the board because it is the most consistent, learnable structure. The board is a
# chromatic blue at a lighting-stable hue (measured hue peak 168-172 on PIL's 0-255
# scale across the whole split, spread 4), while snow is achromatic (low
# saturation). Lighting moves brightness, not hue, so an HSV rule generalizes where
# an RGB threshold would not. TF's rgb_to_hsv returns H,S,V in [0,1]; the PIL-scale
# band [150,185]/255 and saturation floor 50/255 map to the constants below.
_BOARD_HUE_LO = 150.0 / 255.0
_BOARD_HUE_HI = 185.0 / 255.0
_BOARD_SAT_MIN = 50.0 / 255.0
# Fill masked board pixels with neutral grey (127.5 in 0-255 -> 0.0 after the
# /127.5 - 1 rescale, i.e. the centre of tanh's linear region). This removes the
# confound WITHOUT handing the generator a large flat region at a tanh rail (which
# black, -> -1, would). UNCERTAIN: grey is the reasoned default but unproven for
# feature transfer; if masked runs still degrade, black is the documented
# fallback. See docs/UPGRADES.md #49 and the memory note.
_BOARD_FILL_255 = 127.5


def mask_blue_board(image_255):
    """Zero out the blue measurement board, filling it with neutral grey.

    Args:
        image_255: RGB tensor in [0, 255], shape (H, W, 3).

    Returns:
        Same shape/range with board pixels set to neutral grey. Non-RGB inputs
        (rank != 3 or channels != 3) are returned unchanged — the hue test needs
        colour, so grayscale/other modalities are a no-op.
    """
    image_255 = tf.cast(image_255, tf.float32)
    if image_255.shape.rank != 3 or image_255.shape[-1] != 3:
        return image_255
    hsv = tf.image.rgb_to_hsv(image_255 / 255.0)
    hue, sat = hsv[..., 0], hsv[..., 1]
    board = (hue >= _BOARD_HUE_LO) & (hue <= _BOARD_HUE_HI) & (sat >= _BOARD_SAT_MIN)
    fill = tf.fill(tf.shape(image_255), tf.constant(_BOARD_FILL_255, tf.float32))
    return tf.where(board[..., None], fill, image_255)


def pair_depth_for_modality(modality: str) -> int:
    """Return the depth axis size that ``DataManager`` produces for a modality.

    ``"merged"`` stacks the modalities defined in ``snowgan.modality.Modality``
    (PROFILE + CORE today, depth=2). Any single-modality choice yields depth=1.
    Used by the trainer to size models before loading weights, so the choice
    is the single source of truth for the depth contract.
    """
    if modality == _MERGED_MODALITY:
        return len(Modality)
    if modality in _SINGLE_MODALITIES:
        return 1
    raise ValueError(
        f"Unknown modality {modality!r}. Expected one of "
        f"{sorted(_SINGLE_MODALITIES | {_MERGED_MODALITY})}."
    )


class DataManager:

    def __init__(self, config):
        """
        Initialize the DataManager with a configuration object.

        Function arguments:
            config (Config) - Configuration object containing dataset parameters

        This class handles loading the dataset and preparing batches of data for training.
        """
        dataset_name = getattr(config, "dataset", None) or "rmdig/rocky_mountain_snowpack"
        self.dataset = load_dataset(dataset_name)
        manifest_df = self.dataset["train"].to_pandas().drop(columns=["image", "audio"], errors="ignore")

        # Normalize datatype to canonical ints at the load boundary, so every
        # consumer — pair_index/batch/batch_merged here, and external readers of
        # dm.manifest (snowGradient) — sees one representation. Defensive: HF's
        # ClassLabel cast in load_dataset already yields ints 0-3, so this is not
        # required today; it guarantees the datatype=0/1/2/3 contract regardless
        # of encoding (e.g. a consumer reading the JSONL directly, or if the
        # ClassLabel schema is ever dropped) and is a single source of truth for
        # the mapping. Idempotent; unknown values raise rather than skip silently.
        if "datatype" in manifest_df.columns:
            manifest_df["datatype"] = manifest_df["datatype"].map(normalize_datatype)

        # Record which manifest this run actually saw. snowGradient had to infer
        # this from pool contents to establish that the v0.1.0 backbones predate
        # sites 3-6; a recorded revision makes it a lookup.
        try:
            revision = getattr(self.dataset["train"].info, "version", None)
            fingerprint = getattr(self.dataset["train"], "_fingerprint", None)
            resolved = str(revision) if revision else None
            if fingerprint:
                resolved = f"{resolved or 'v?'}+fp:{fingerprint}"
            if resolved:
                config.dataset_revision = resolved
        except Exception:
            pass

        self.manifest_columns = manifest_df.columns.tolist()
        self.manifest = manifest_df.values.tolist()

        self.config = config

        # Single source of the datatype mapping (copied so callers can't mutate
        # the module constant). Datatype normalization goes through
        # normalize_datatype, which this mirrors.
        self.translator = dict(DATATYPE_TO_INT)

        # Stack depth this DataManager will produce for the trainer. Derived
        # once at construction from config.modality so the trainer can read
        # `dataset.pair_depth` as the canonical value when sizing its models.
        self.pair_depth = pair_depth_for_modality(getattr(config, "modality", "magnified_profile"))

        # Local image root (optional). The HF manifest's `image` column is
        # URL-backed — `dataset['train'][i]['image']` issues an HTTP GET for a
        # ~16 MB PNG on EVERY access, i.e. once per image per epoch. Measured on
        # this dataset: 2.03 s/image over HTTP vs 0.007 s/image decoding the
        # same row from a local 512 px cache — a 290x difference that makes the
        # data pipeline ~97% of a 1024 px train step and completely hides the
        # GPU. (UPGRADES #51 noted the HTTP cost in passing; this is the fix.)
        #
        # When `image_root` is set, images load from
        # `<image_root>/<manifest file_path>` and the network is never touched.
        # Rows missing from the root fall back to the HF column, so a partial
        # mirror degrades in speed, not correctness.
        raw_root = getattr(config, "image_root", None) or None
        # expanduser: a root of "~/rmdig-cache-512" arriving from a config JSON or a
        # quoted CLI arg would otherwise never match any file, giving a 100% miss
        # rate that is invisible except as a run that is mysteriously 19x slower.
        self.image_root = os.path.expanduser(raw_root) if raw_root else None
        self._local_hits = 0
        self._local_misses = 0
        if self.image_root:
            if "file_path" not in self.manifest_columns:
                raise ValueError(
                    "image_root was set but the manifest has no 'file_path' column, so "
                    "local files cannot be resolved. Either unset image_root or use a "
                    "dataset revision that carries file_path."
                )
            # Fail loudly rather than silently falling back to the URL-backed
            # column: the whole point of this setting is throughput, and a typo'd
            # or unmounted root costs ~290x per image with no other signal.
            if not os.path.isdir(self.image_root):
                raise ValueError(
                    f"image_root {self.image_root!r} is not a directory. Training would "
                    f"silently fall back to the URL-backed image column (~2 s of HTTP per "
                    f"image, ~19x slower end to end). Build a mirror with "
                    f"scripts/build_image_cache.py, or pass --image_root '' to opt out."
                )
            print(f"Local image root: {self.image_root}")

        # Track seen profiles across runs/epochs and keep in sync with config for persistence
        self.seen_profiles = set(getattr(self.config, "seen_profiles", []) or [])
        self.config.seen_profiles = self.seen_profiles
        self.seen_cores = set()

    @cached_property
    def pair_index(self):
        """Cross-pair index of the manifest, grouped by (site, column, core).

        Returns:
            dict[tuple, list[tuple[int, int]]] — keys are ``(site, column, core)``
            tuples for every group that has at least one core (datatype=0) and at
            least one magnified profile (datatype=2). Values are the full Cartesian
            product ``[(core_idx, profile_idx), ...]`` of every core row index ×
            every magnified-profile row index that share the group key.

        Intended for downstream consumers (e.g. AvAI transfer learning) that split
        at the group level to prevent leakage and want every cross-pair per group
        as a combinatorial augmentation. The GAN trainer's ``batch_merged`` does
        not use this — it has different (per-epoch, no-reuse) pairing semantics.

        Read-only and side-effect-free: does not touch ``self.config``,
        ``self.seen_profiles``, ``self.seen_cores``, or ``self.config.train_ind``.
        Cached on the instance after the first access.
        """
        cores: dict[tuple, list[int]] = {}
        profiles: dict[tuple, list[int]] = {}
        for idx in range(len(self.manifest)):
            meta = self._get_manifest_entry(idx)
            if meta is None:
                continue
            datatype = meta.get("datatype")
            if datatype != 0 and datatype != 2:
                continue
            key = (meta.get("site"), meta.get("column"), meta.get("core"))
            bucket = cores if datatype == 0 else profiles
            bucket.setdefault(key, []).append(idx)

        return {
            key: [(c, p) for c in cores[key] for p in profiles[key]]
            for key in cores.keys() & profiles.keys()
        }

    @property
    def held_out_keys(self):
        """``(site, column, core)`` groups the training stream must not see.

        ``derive_splits`` has always partitioned the groups 80/10/10 and
        persisted the pools, and ``Trainer`` mirrors them onto both configs so
        AvAI can evaluate against ``test_pool`` — but no batch path ever
        consulted them. ``batch()`` filtered on ``datatype`` alone, so the GAN
        trained on its own held-out groups and every downstream transfer probe
        against these backbones was contaminated. This is the set that
        ``batch`` / ``batch_merged`` now skip when ``config.honor_splits``.

        Built from validation+test rather than *from* ``trained_pool`` on
        purpose: the pools derive from ``pair_index``, which only contains
        groups having **both** a core and a magnified profile. Filtering to
        ``trained_pool`` would silently discard every unpaired group — a large
        slice of the magnified_profile rows — which is data loss, not a
        leakage fix. Excluding the held-out groups gives the same isolation and
        keeps everything that was never in any pool.

        Pools round-trip through JSON as ``list[list]``; keys are re-tupled
        here so membership matches the manifest's tuple keys.

        Memoized on the pool sizes rather than ``cached_property``, on purpose.
        ``Trainer.__init__`` constructs the DataManager *before* it calls
        ``derive_splits``, so a plain cache populated by any earlier access
        would freeze an empty set and silently restore the leak — with no
        error, which is the failure mode this method exists to close.
        """
        validation = getattr(self.config, "validation_pool", None) or []
        test = getattr(self.config, "test_pool", None) or []
        signature = (len(validation), len(test))
        if getattr(self, "_held_out_signature", None) != signature:
            self._held_out_keys = {tuple(entry) for entry in validation}
            self._held_out_keys.update(tuple(entry) for entry in test)
            self._held_out_signature = signature
        return self._held_out_keys

    def load_image(self, index, meta=None):
        """Return the PIL/array image for a manifest row.

        Prefers a local file under ``image_root`` (keyed by the manifest's
        ``file_path``) and falls back to the HF ``image`` column — which is
        URL-backed and pays an HTTP round trip per access.

        Returns ``None`` when neither source yields an image, so callers can
        skip the row rather than crash a long run on one bad file.
        """
        # getattr, not attribute access: the test suite constructs DataManager
        # via __new__ with a synthetic manifest (test_dataset / test_splits /
        # test_modality_modes), so __init__-assigned state may not exist.
        image_root = getattr(self, "image_root", None)
        if image_root:
            if meta is None:
                meta = self._get_manifest_entry(index)
            relative = (meta or {}).get("file_path")
            if relative:
                local = os.path.join(image_root, str(relative))
                if os.path.exists(local):
                    try:
                        with Image.open(local) as handle:
                            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
                        self._local_hits = getattr(self, "_local_hits", 0) + 1
                        return image
                    except (OSError, ValueError) as error:
                        # A truncated/corrupt local file must not be silently
                        # swapped for a slow network fetch without saying so.
                        print(f"Warning: local image {local} unreadable ({error}); "
                              f"falling back to the remote column.")
            misses = getattr(self, "_local_misses", 0) + 1
            self._local_misses = misses
            # A plain miss is otherwise the silent case: no exception, just a
            # ~2 s HTTP GET instead of 7 ms. Say so the first time and then on a
            # widening cadence, so an incomplete mirror is visible in the log
            # without flooding it.
            if misses in (1, 10, 100, 1000) or misses % 5000 == 0:
                print(f"NOTE: {misses} image(s) missing from image_root "
                      f"{image_root}; those rows fall back to the URL-backed "
                      f"column (~2 s each). Mirror may be incomplete.")

        return self.dataset['train'][index]['image']

    def _is_held_out(self, meta):
        """True when this manifest row belongs to a validation/test group."""
        if not getattr(self.config, "honor_splits", True):
            return False
        if not self.held_out_keys:
            return False
        return (meta.get("site"), meta.get("column"), meta.get("core")) in self.held_out_keys

    def _get_manifest_entry(self, index):
        if index < 0 or index >= len(self.manifest):
            return None
        row = self.manifest[index]
        return {key: row[i] for i, key in enumerate(self.manifest_columns)}

    def reset_seen_profiles(self):
        self.seen_profiles.clear()
        self.config.seen_profiles = self.seen_profiles

    def derive_splits(self, train_frac: float = 0.8, val_frac: float = 0.1):
        """Populate ``config.trained_pool / validation_pool / test_pool`` with a
        deterministic 80/10/10 split of ``pair_index`` keys.

        Splits are taken at the group level (``(site, column, core)`` tuples) so
        every profile of a given core lands in exactly one split — preventing
        leakage where downstream consumers (AvAI's transfer-learning eval) would
        otherwise see profiles of cores the GAN was trained on.

        Idempotent: if all three pools are already populated on the config, this
        method returns without rederiving. That preserves a previously chosen
        split across resumes — a torn write that loses a pool will deterministically
        regenerate the same split on next call (same seed, same pair_index keys).

        Persistence is the caller's responsibility — this method only mutates
        ``self.config`` in memory.

        Pools are persisted as ``list[list]`` (JSON-friendly). Group keys round-
        trip through JSON as lists; consumers that need tuple-keyed lookup against
        ``pair_index`` should ``tuple(...)`` each entry on read.
        """
        if (
            self.config.trained_pool is not None
            and self.config.validation_pool is not None
            and self.config.test_pool is not None
        ):
            return

        keys = sorted(self.pair_index.keys())
        rng = random.Random(int(getattr(self.config, "seed", 42)))
        rng.shuffle(keys)

        n = len(keys)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        self.config.trained_pool = [list(k) for k in keys[:n_train]]
        self.config.validation_pool = [list(k) for k in keys[n_train:n_train + n_val]]
        self.config.test_pool = [list(k) for k in keys[n_train + n_val:]]

    def next_batch(self, batch_size, config=None):
        """Dispatch to the right batch fetch path based on ``config.modality``.

        Returns a numpy array of shape ``(B, pair_depth, H, W, C)``. Single
        modalities produce depth=1; ``"merged"`` produces depth=2. The trainer
        calls this exclusively so a modality change at config time is the only
        thing that needs to switch behavior.
        """
        modality = getattr(self.config, "modality", "magnified_profile")
        if modality == _MERGED_MODALITY:
            return self.batch_merged(batch_size, config=config)
        return self.batch(batch_size, datatype=modality, config=config)

    def batch(self, batch_size, datatype = "magnified_profile", config = None):
        """
        Prepare a batch of data from the dataset.
        
        Function arguments:
            batch_size (int) - Number of samples to include in the batch
            datatype (int) - Type of data to load (0: core, 1:
                        profile, 2: magnified_profile, 3: crystal_card)
            config (Config) - Optional configuration object to override current settings
        
        Returns:
            np.ndarray - Batch of images ready for training"""
        count = 0
        batch = []

        if config:
            self.config = config

        # Normalize the requested datatype to its canonical int. Idempotent, so
        # callers may pass a modality name ("magnified_profile") or an int (2);
        # the old `self.translator[datatype]` KeyError'd on an already-int arg.
        # Compared below against the manifest's now-normalized int datatypes.
        datatype = normalize_datatype(datatype)

        print(f"Collecting a batch...")
        
        while count < batch_size and self.config.train_ind < len(self.manifest):
            meta = self._get_manifest_entry(self.config.train_ind)
            if meta is None:
                break
            sample_datatype = meta.get("datatype")

            print(f"Checking sample at index {self.config.train_ind} - {sample_datatype} - {datatype}")

            if sample_datatype == datatype and not self._is_held_out(meta):
                image = self.load_image(self.config.train_ind, meta)
                if image is None:
                    self.config.train_ind += 1
                    continue

                scaled_image = self.preprocess_image(image)

                # Add a depth dimension for single-view samples
                scaled_image = tf.expand_dims(scaled_image, axis=0)

                batch.append(scaled_image)
                print(f"Image {self.config.train_ind} added")

                # HF datasets accumulates PIL/decoded buffers in process RAM
                # on repeated indexed access (huggingface/datasets #4883, #7180).
                # Drop references eagerly so they're collectable before the
                # next iteration grabs another row. (`sample` is gone — the row
                # is no longer materialized here; load_image owns that.)
                del image, scaled_image

                count += 1
            self.config.train_ind += 1

        if len(batch) > 0:
            return np.stack(batch)
        
        else:
            print("All images have been assessed and no available images could be found")
            return None
        
    def batch_merged(self, batch_size, config = None):
        """
        Prepare a batch of data from the dataset.
        
        Function arguments:
            batch_size (int) - Number of samples to include in the batch
            datatype (int) - Type of data to load (0: core, 1:
                        profile, 2: magnified_profile, 3: crystal_card)
            config (Config) - Optional configuration object to override current settings
        
        Returns:
            np.ndarray - Batch of images ready for training"""
        count = 0
        batch = []

        if config:
            self.config = config

        print(f"Collecting a batch...")
        
        while count < batch_size and self.config.train_ind < len(self.manifest):
            meta = self._get_manifest_entry(self.config.train_ind)
            if meta is None:
                break
            sample_datatype = meta.get("datatype")

            print(f"Checking sample at index {self.config.train_ind} - {sample_datatype}")

            # If we've found a core sample in a group we're allowed to train on
            if sample_datatype == 0 and not self._is_held_out(meta):

                # NB: no `self.dataset['train'][...]` here. Materializing an HF
                # row decodes every column including the URL-backed `image`, so
                # the old fetch paid a full ~16 MB HTTP GET per core candidate
                # purely to read site/column/core -- values `meta` already has.
                # That single line cancelled the image_root speedup on this path.
                core_raw = self.load_image(self.config.train_ind, meta)
                if core_raw is None:
                    self.config.train_ind += 1
                    continue
                core_image = self.preprocess_image(core_raw)
                self.seen_cores.add(self.config.train_ind)

                profile_ind = self.config.train_ind
                profile_image = None

                # Look for a magnified profile to pair with
                seen_profile = False
                while profile_image == None and (profile_ind + 1) < len(self.manifest):
                    profile_ind += 1
                    profile_meta = self._get_manifest_entry(profile_ind)

                    if profile_ind >= len(self.manifest):
                        break
                    if profile_meta is None:
                        break

                    # If we've found a profile to use
                    if profile_meta.get("datatype") == 2:
                        
                        seen_profile = True

                        # Check that it matches our core
                        if profile_ind in self.seen_profiles:
                            continue

                        if profile_meta.get('site') != meta.get('site'):
                            continue

                        if profile_meta.get('column') != meta.get('column'):
                            continue

                        if profile_meta.get('core') != meta.get('core'):
                            continue

                        # Preprocessing image
                        profile_raw = self.load_image(profile_ind, profile_meta)
                        if profile_raw is None:
                            continue
                        profile_image = self.preprocess_image(profile_raw)
                        self.seen_profiles.add(profile_ind)
                        self.config.seen_profiles = self.seen_profiles

                    # If all profiles have been viewed
                    if seen_profile and profile_meta.get("datatype") == 0:
                        
                        break


                if profile_image == None:
                    self.config.train_ind += 1
                    
                    print("Profile image not found, skipping...")
                    continue

                merged_image = self.merge_images(core_image, profile_image)

                batch.append(merged_image)
                print(f"Image add with core {self.config.train_ind} and profile {profile_ind}")

                # HF datasets accumulates PIL buffers in process RAM on
                # repeated indexed access (#4883, #7180). Drop references
                # eagerly so they're collectable before the next iteration.
                # (`profile_sample` is gone — load_image owns row materialization
                # now, and the per-row `segment` lookup that used to be inlined in
                # the log line above was a second full row fetch per pair, i.e. a
                # second HTTP round trip on the URL-backed image column.)
                del core_raw, core_image, profile_image, merged_image

                count += 1
            self.config.train_ind += 1

        if len(batch) > 0:
            return np.stack(batch)
        
        else:
            print("All images have been assessed and no available images could be found")
            return None
        
    def preprocess_image(self, image):
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(np.array(image))  # Convert from PIL to tensor
        
        image = tf.image.resize(image, self.config.resolution)
        # NOTE: no per-image debug print here. `tf.reduce_max(...).numpy()` forces
        # a device sync on every image (~0.5 ms/img measured on an RTX 5080), so
        # the GPU cannot pipeline across images — plus it floods stdout. This is
        # the hottest loop in the data pipeline; keep it sync-free. (UPGRADES #51;
        # CLAUDE.md §6: print is legacy.)

        # Mask the blue measurement board BEFORE the [-1,1] rescale, while pixels
        # are still in [0,255] where the HSV thresholds are defined. Same op must
        # run in the on-device phone pipeline (it is per-pixel arithmetic, no
        # accelerator needed), so training and inference see identical inputs.
        if getattr(self.config, "mask_board", False):
            image = mask_blue_board(image)

        if image.shape.rank == 2:  # grayscale image
            image = tf.expand_dims(image, -1)

        scaled_image = (tf.cast(image, tf.float32) / 127.5) - 1.0
        return scaled_image

    def merge_images(self, core, profile):
        """Stack core and profile along a new depth axis per ``Modality``.

        ``Modality.PROFILE`` lands at depth index 0 and ``Modality.CORE`` at
        depth index 1. Profile defines the spatial resolution; the core image
        is resized to match before stacking.
        """
        core_resized = tf.image.resize(core, (profile.shape[0], profile.shape[1]))
        images = {Modality.PROFILE: profile, Modality.CORE: core_resized}
        return tf.stack([images[Modality.PROFILE], images[Modality.CORE]], axis=0)
