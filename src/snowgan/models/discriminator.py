import os
import tensorflow as tf
import keras

from snowgan.config import build

class Discriminator(keras.Model):
    def __init__(self, config):
        """
        Wasserstein GAN Discriminator: distinguishes real vs. synthetic snow images.
        
        Args:
            config
        """
        super(Discriminator, self).__init__()
        self.config = config

        # Check if first is larger than last filter (i.e. discriminator setup) switch
        if self.config.filter_counts[0] > self.config.filter_counts[-1]:
            print(f"WARNING: Inverse filter counts detected, inverting filter counts - {self.config.filter_counts}")
            self.config.filter_counts = self.config.filter_counts[::-1]

        self.resolution = self.config.resolution
        self.model = self._build_model()
        self.optimizer = self.get_optimizer()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = keras.Input(shape=(self.config.depth, self.config.resolution[0], self.config.resolution[1], self.config.channels))
        x = inputs
        use_sn = getattr(self.config, 'spectral_norm', False)

        ksize = (1, self.config.kernel_size[0], self.config.kernel_size[1])
        kstride = (1, self.config.kernel_stride[0], self.config.kernel_stride[1])

        for filters in self.config.filter_counts:
            conv = keras.layers.Conv3D(filters, ksize, strides=kstride, padding=self.config.padding)
            x = keras.layers.SpectralNormalization(conv)(x) if use_sn else conv(x)
            x = keras.layers.LeakyReLU(negative_slope=self.config.negative_slope)(x)


        # name="features": cross-repo contract with AvAI Phase 3 transfer
        # learning, which calls model.get_layer("features") on the loaded
        # discriminator. See docs/UPGRADES.md §3.
        # Capacity campaign K1 (capacity_plan H1): a nonlinear head. The
        # default linear head is a position-specific linear readout over the
        # 32x32 grid of 63-px patch features: it can weigh *where* a cue is
        # (e.g. vignette darkness) but cannot model interactions between
        # patches (layout, geometry, placement). "mlp" adds a 1x1 conv down to
        # 64 channels and a 128-unit hidden layer, so the score is a nonlinear
        # function of the whole grid. It changes the `features` tensor that
        # AvAI reads (32x32x64 instead of 32x32x1024), so a K1 critic is not a
        # drop-in transfer backbone.
        disc_head = getattr(self.config, "disc_head", "linear") or "linear"
        if disc_head == "mlp":
            squeeze = keras.layers.Conv3D(64, (1, 1, 1), strides=1, padding="same", name="head_squeeze")
            x = keras.layers.SpectralNormalization(squeeze)(x) if use_sn else squeeze(x)
            x = keras.layers.LeakyReLU(negative_slope=self.config.negative_slope)(x)
        elif disc_head != "linear":
            raise ValueError(f"disc_head must be 'linear' or 'mlp', got {disc_head!r}")
        x = keras.layers.Flatten(name="features")(x)
        if disc_head == "mlp":
            hidden = keras.layers.Dense(128, name="head_hidden")
            x = keras.layers.SpectralNormalization(hidden)(x) if use_sn else hidden(x)
            x = keras.layers.LeakyReLU(negative_slope=self.config.negative_slope)(x)
        # Pin the critic's scalar output head to float32 under any mixed
        # precision policy. Wasserstein scores are unbounded, so fp16 can
        # overflow during the WGAN-GP gradient penalty's interpolation pass
        # and corrupt the entire loss. UPGRADES #15.
        # No activation: a WGAN critic outputs an unbounded linear score.
        # config.final_activation (e.g. "tanh") is intentionally IGNORED here —
        # it applies only to the generator's RGB head. Do not wire it into this
        # head: a bounded critic score would break the Wasserstein estimate and
        # the gradient penalty.
        dense = keras.layers.Dense(1, dtype="float32")
        outputs = keras.layers.SpectralNormalization(dense)(x) if use_sn else dense(x)

        return keras.Model(inputs, outputs, name="Discriminator")

    def get_optimizer(self):
        optimizer = keras.optimizers.Adam(learning_rate = self.config.learning_rate, beta_1 = self.config.beta_1, beta_2 = self.config.beta_2)
        # Wrap in LossScaleOptimizer under mixed_float16 so fp16 gradients
        # don't silently underflow to zero (UPGRADES #15).
        if keras.mixed_precision.global_policy().name == "mixed_float16":
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
        return optimizer

    def get_loss(self, real_output, synthetic_output, gp, lambda_gp):
        """
        Wasserstein loss with gradient penalty for the Discriminator.

        Args:
            real_output (tf.Tensor): Discriminator output for real images
            synthetic_output (tf.Tensor): Discriminator output for fake images
            gp (tf.Tensor): Gradient penalty term
            lambda_gp (float): Weight for gradient penalty

        Returns:
            tf.Tensor: Total discriminator loss
        """
        # Cast to float32 for loss computation (mixed precision outputs float16)
        wasserstein = tf.reduce_mean(tf.cast(synthetic_output, tf.float32)) - tf.reduce_mean(tf.cast(real_output, tf.float32))
        return wasserstein + lambda_gp * tf.cast(gp, tf.float32)
    
def load_discriminator(checkpoint, config = None):
    
    if not config:
        split = checkpoint.split("/")
        config = build("/".join(split[:-1]) + "/discriminator.keras")
        config.checkpoint = split.pop() # Get the model filename from the path
        config.save_dir = "/".join(split) + "/"

    # Load the discriminator
    discriminator = Discriminator(config)
    discriminator.model.build((None, config.depth, config.resolution[0], config.resolution[1], config.channels))
    return discriminator
