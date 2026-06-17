import os
import tensorflow as tf
import keras

from snowgan.config import build


class PixelNorm(keras.layers.Layer):
    """Per-pixel feature-vector normalization (ProGAN, Karras et al. 2018).

    Normalizes each spatial position's channel vector to unit RMS. Unlike
    BatchNorm it uses no batch statistics, so it is safe under WGAN-GP (the
    gradient penalty is a per-sample quantity that BatchNorm would couple
    across the batch) and gives the generator the activation-scale control its
    deep transposed/resize-conv stack otherwise lacks — without which the
    final tanh saturates toward a single hue (the v0.1 monochrome blobs).
    """

    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        scale = tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)
        return inputs * scale

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class Generator(keras.Model):
    def __init__(self, config):
        """
        Wasserstein GAN Generator that converts latent vectors into synthetic snow images.
        
        Args:
            resolution (tuple): Target image resolution, e.g. (512, 512)
            latent_dim (int): Dimension of the input latent vector
        """
        
        super(Generator, self).__init__()
        self.config = config

        # Check if first is smaller than last filter (i.e. discriminator setup) switch
        if self.config.filter_counts[0] < self.config.filter_counts[-1]:
            print(f"Inverse filter counts detected, inverting filter counts (Normal if using default config for generator) - {self.config.filter_counts}")
            self.config.filter_counts = self.config.filter_counts[::-1]

        self.model, self.fade_endpoints = self._build_model()
        self.optimizer = self.get_optimizer()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = keras.Input(shape=(self.config.latent_dim,))

        start_depth = max(1, int(getattr(self.config, "depth", 1) or 1))
        x = keras.layers.Dense(start_depth * 16 * 16 * self.config.filter_counts[0], use_bias=False)(inputs)
        x = keras.layers.Reshape((start_depth, 16, 16, self.config.filter_counts[0]))(x)

        feats = []
        ksize = (1, self.config.kernel_size[0], self.config.kernel_size[1])
        # The depth axis is never upsampled (paired modalities live there); the
        # spatial upsampling factor comes from kernel_stride.
        up_size = (1, self.config.kernel_stride[0], self.config.kernel_stride[1])
        gen_norm = getattr(self.config, "gen_norm", None) or ("batch" if self.config.batch_norm else "none")
        # Upsampler: "resize" = UpSampling(nearest) + stride-1 conv (no
        # checkerboard, but LOW-PASS — it smooths away high-frequency texture and
        # the generator collapses to blobs). "transpose" = learned
        # Conv3DTranspose (recovers fine detail like the magnified_profile model;
        # use a kernel divisible by the stride, e.g. 4/2, to keep checkerboard
        # mild). See docs/UPGRADES.md.
        upsampler = getattr(self.config, "gen_upsampler", "resize") or "resize"

        def norm_act(x):
            if gen_norm == "batch":
                x = keras.layers.BatchNormalization()(x)
            elif gen_norm == "pixel":
                x = PixelNorm()(x)
            return keras.layers.LeakyReLU(self.config.negative_slope)(x)

        def conv_block(x, filters):
            """Stride-1 conv + normalization + activation."""
            x = keras.layers.Conv3D(filters, ksize, strides=1, padding=self.config.padding)(x)
            return norm_act(x)

        for filters in self.config.filter_counts:
            if upsampler == "transpose":
                # Learned upsampling (Conv3DTranspose) so the generator can
                # synthesize texture, plus a stride-1 conv for capacity.
                x = keras.layers.Conv3DTranspose(filters, ksize, strides=up_size, padding=self.config.padding)(x)
                x = norm_act(x)
                x = conv_block(x, filters)
            else:
                # Resize-convolution (Odena et al. 2016): nearest upsample + two
                # stride-1 convs. Avoids checkerboard but is low-pass.
                x = keras.layers.UpSampling3D(size=up_size)(x)
                x = conv_block(x, filters)
                x = conv_block(x, filters)
            feats.append(x)

        # toRGB does the final doubling (the "+1" in the
        # 16*2^(len(filter_counts)+1) resolution coupling). Pin the head to
        # float32 even under a mixed_float16 policy: the tanh saturates near ±1
        # and fp16 underflows the asymptotes / overflows the upstream gradient
        # (UPGRADES #15).
        if upsampler == "transpose":
            curr_img = keras.layers.Conv3DTranspose(self.config.channels, ksize, strides=up_size, padding=self.config.padding, activation=self.config.final_activation, use_bias=False, name="toRGB_curr", dtype="float32")(x)
        else:
            x_rgb = keras.layers.UpSampling3D(size=up_size, name="toRGB_upsample")(x)
            curr_img = keras.layers.Conv3D(self.config.channels, kernel_size=(1, 1, 1), strides=1, padding='same', activation=self.config.final_activation, use_bias=False, name="toRGB_curr", dtype="float32")(x_rgb)

        base_model = keras.Model(inputs, curr_img, name="Generator")

        # Build fade endpoints model (prev_up_rgb, curr_rgb) if we have >= 2 blocks
        fade_model = None
        if len(feats) >= 2:
            prev_feat = feats[-2]
            # Use bias so the prev path can avoid collapsing to near-zero early
            prev_rgb = keras.layers.Conv3D(self.config.channels, kernel_size=(1,1,1), padding='same', activation=self.config.final_activation, use_bias=True, name="toRGB_prev")(prev_feat)
            # Dynamically resize previous RGB to match current output spatial dims (depth preserved)
            def resize_spatial(xs):
                src, tgt = xs
                tgt_shape = tf.shape(tgt)
                b, d, h, w, c = tgt_shape[0], tgt_shape[1], tgt_shape[2], tgt_shape[3], tgt_shape[4]
                src_shape = tf.shape(src)
                src = tf.reshape(src, (b * src_shape[1], src_shape[2], src_shape[3], c))
                resized = tf.image.resize(src, size=(h, w), method='nearest')
                return tf.reshape(resized, (b, src_shape[1], h, w, c))

            prev_up = keras.layers.Lambda(
                resize_spatial,
                name="prev_resize_to_curr"
            )([prev_rgb, curr_img])

            fade_model = keras.Model(inputs, [prev_up, curr_img], name="GeneratorFadeEndpoints")

        return base_model, fade_model

    def get_optimizer(self):
        optimizer = keras.optimizers.Adam(learning_rate = self.config.learning_rate, beta_1 = self.config.beta_1, beta_2 = self.config.beta_2)
        # Wrap in LossScaleOptimizer under mixed_float16 so fp16 gradients
        # don't silently underflow to zero below ~1e-7 (UPGRADES #15).
        if keras.mixed_precision.global_policy().name == "mixed_float16":
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
        return optimizer

    def get_loss(self, synthetic_difference):
        """
        Generator loss for Wasserstein GAN loss

        Args:
            synthetic_output (tf.Tensor): Discriminator output for fake images

        Returns:
            tf.Tensor: Generator loss
        """
        return -tf.reduce_mean(tf.cast(synthetic_difference, tf.float32))


def load_generator(checkpoint, config = None):
    
    if not config:
        split = checkpoint.split("/")
        config = build("/".join(split[:-1]) + "/generator.keras")
        
        config.checkpoint = checkpoint # Get the model filename from the path
        config.save_dir = "/".join(split) + "/"


    # Load the discriminator
    generator = Generator(config)
    # Build generator with latent input shape
    generator.model.build((None, config.latent_dim))
    # Build fade endpoints as well if present (latent input shape)
    if generator.fade_endpoints is not None:
        generator.fade_endpoints.build((None, config.latent_dim))
    return generator
