"""Inference-only rank-4 export of the discriminator and generator.

TFLite runs CONV_3D on a slow CPU-only kernel with no int8 and no GPU / NNAPI /
Core ML delegate, so the training models cannot ship to a phone as they are.
They do not need to: every Conv3D / Conv3DTranspose here has kernel
``(1, kH, kW)`` and depth stride 1, so at ``depth == 1`` each one *is* a
Conv2D / Conv2DTranspose using ``kernel[0]``, and the rest of the stack
(LeakyReLU, PixelNorm, BatchNorm, UpSampling, Flatten, Dense, Reshape) acts per
depth slice. ``to_conv2d`` rebuilds the same chain at rank 4 with the same
weights; inputs become ``(B, H, W, 3)`` (discriminator) and outputs
``(B, H, W, 3)`` (generator).

What this deliberately refuses rather than approximates:

- ``depth != 1``. The critic's ``Flatten -> Dense`` head mixes depth slices, so
  a merged (core + profile) model has no single rank-4 equivalent.
- A generator mid-fade. ``Generator.model`` holds only the current-resolution
  endpoint; the trainer blends it with ``fade_endpoints`` until the fade ends.
- Any layer type not listed above, any non-chain topology, and any non-unit
  depth stride or dilation. A converter that skipped an unknown layer would
  produce a model that loads, runs, and is wrong.

SpectralNormalization needs no special handling: at inference
``SpectralNormalization.call(training=False)`` runs the wrapped layer, whose
stored kernel is already normalised, so the inner kernel is copied as-is.
"""

import argparse
import os
import tempfile

import keras
import numpy as np
import tensorflow as tf

from snowgan.models.generator import PixelNorm


def _keras_model(model):
    """Accept a Discriminator / Generator wrapper or a bare keras.Model."""
    config = getattr(model, "config", None)
    if config is not None and getattr(model, "fade_endpoints", None) is not None:
        if getattr(config, "fade", False) and int(getattr(config, "fade_step", 0)) < int(getattr(config, "fade_steps", 0)):
            raise ValueError(
                f"Generator is mid-fade (fade_step {config.fade_step} < fade_steps {config.fade_steps}); "
                "Generator.model is only the current-resolution endpoint of what the trainer runs. "
                "Export a checkpoint taken after the fade completed."
            )
    inner = getattr(model, "model", model)
    if not isinstance(inner, keras.Model):
        raise TypeError(f"Expected a snowgan Discriminator/Generator or keras.Model, got {type(model).__name__}.")
    return inner


def _chain(model):
    """The model's layers in order, after checking they form a single chain."""
    layers = model.layers
    if not isinstance(layers[0], keras.layers.InputLayer):
        raise ValueError(f"{model.name}: expected an InputLayer first, got {type(layers[0]).__name__}.")
    for prev, layer in zip(layers, layers[1:]):
        if layer.input is not prev.output:
            raise ValueError(f"{model.name}: layer {layer.name!r} does not consume {prev.name!r}; "
                             "only single-chain models are supported.")
    if layers[-1].output is not model.output:
        raise ValueError(f"{model.name}: last layer is not the model output.")
    return layers[1:]


def _bias(layer):
    return [layer.bias.numpy()] if layer.use_bias else []


def _conv(inner, name, transpose):
    kernel = inner.kernel.numpy()
    if kernel.shape[0] != 1 or inner.strides[0] != 1 or inner.dilation_rate[0] != 1:
        raise ValueError(f"{name}: kernel {kernel.shape[:3]}, strides {inner.strides}, dilation "
                         f"{inner.dilation_rate} mix the depth axis; not expressible at rank 4.")
    if getattr(inner, "groups", 1) != 1:
        raise ValueError(f"{name}: grouped convolution is not supported.")
    cls = keras.layers.Conv2DTranspose if transpose else keras.layers.Conv2D
    layer = cls(inner.filters, kernel.shape[1:3], strides=inner.strides[1:], padding=inner.padding,
                dilation_rate=inner.dilation_rate[1:], activation=inner.activation,
                use_bias=inner.use_bias, dtype=inner.dtype_policy, name=name)
    return layer, [kernel[0]] + _bias(inner)


def _convert(layer):
    """Return (rank-4 layer, weights or None) for one layer of the chain."""
    name = layer.name
    inner = layer.layer if isinstance(layer, keras.layers.SpectralNormalization) else layer
    # Order matters: Conv3DTranspose subclasses Conv3D.
    if isinstance(inner, keras.layers.Conv3DTranspose):
        return _conv(inner, name, transpose=True)
    if isinstance(inner, keras.layers.Conv3D):
        return _conv(inner, name, transpose=False)
    if isinstance(inner, keras.layers.Dense):
        return (keras.layers.Dense(inner.units, activation=inner.activation, use_bias=inner.use_bias,
                                   dtype=inner.dtype_policy, name=name),
                [inner.kernel.numpy()] + _bias(inner))
    if inner is not layer:
        raise TypeError(f"{name}: SpectralNormalization around {type(inner).__name__} is not supported.")
    if isinstance(layer, keras.layers.UpSampling3D):
        if layer.size[0] != 1:
            raise ValueError(f"{name}: upsamples the depth axis ({layer.size}).")
        return keras.layers.UpSampling2D(layer.size[1:], interpolation="nearest", name=name), None
    if isinstance(layer, keras.layers.BatchNormalization):
        axis = layer.axis if isinstance(layer.axis, int) else layer.axis[0]
        if axis not in (-1, 4):
            raise ValueError(f"{name}: BatchNormalization over axis {layer.axis}, not channels.")
        return (keras.layers.BatchNormalization(axis=-1, momentum=layer.momentum, epsilon=layer.epsilon,
                                                center=layer.center, scale=layer.scale, name=name),
                layer.get_weights())
    if isinstance(layer, PixelNorm):
        return PixelNorm(epsilon=layer.epsilon, name=name), None
    if isinstance(layer, keras.layers.LeakyReLU):
        return keras.layers.LeakyReLU(negative_slope=layer.negative_slope, name=name), None
    if isinstance(layer, keras.layers.Flatten):
        # Keeps name="features": the AvAI / snowGradient transfer tap.
        return keras.layers.Flatten(name=name), None
    if isinstance(layer, keras.layers.Reshape):
        if layer.target_shape[0] != 1:
            raise ValueError(f"{name}: reshapes to depth {layer.target_shape[0]}; only depth 1 is exportable.")
        return keras.layers.Reshape(layer.target_shape[1:], name=name), None
    raise TypeError(f"{name}: no rank-4 equivalent registered for {type(layer).__name__}. "
                    "Add one here (with an equivalence test) rather than skipping it.")


def to_conv2d(model):
    """Rebuild a depth-1 snowGAN model at rank 4 with identical weights.

    Args:
        model: a ``Discriminator`` / ``Generator`` (or its ``.model``).

    Returns:
        A ``keras.Model`` named ``<name>_conv2d``. Discriminator input
        ``(B, H, W, C)``; generator output ``(B, H, W, C)``. The discriminator's
        ``"features"`` layer is preserved.
    """
    m3 = _keras_model(model)
    in_shape = tuple(m3.input.shape[1:])
    if len(in_shape) == 4:
        if in_shape[0] != 1:
            raise ValueError(f"{m3.name}: input depth {in_shape[0]}; the Flatten->Dense head mixes depth "
                             "slices, so only depth 1 has an exact rank-4 equivalent.")
        in_shape = in_shape[1:]
    inputs = keras.Input(shape=in_shape, name="image" if len(in_shape) == 3 else "latent")
    x = inputs
    for layer in _chain(m3):
        new, weights = _convert(layer)
        x = new(x)
        if weights is not None:
            new.set_weights(weights)
    return keras.Model(inputs, x, name=f"{m3.name}_conv2d")


def export_tflite(model, path, *, representative_dataset=None):
    """Write a rank-4 model to a .tflite file with batch size 1.

    Goes Keras ExportArchive -> SavedModel -> ``from_saved_model``.
    ``TFLiteConverter.from_concrete_functions`` on a Keras 3 model leaves the
    weights as resource variables and writes a ~17 KB file that outputs NaN.

    Args:
        model: output of ``to_conv2d``.
        path: destination ``.tflite`` file (written atomically).
        representative_dataset: optional callable yielding ``[np.float32 batch
            of shape (1, ...)]``; enables int8 quantization with float I/O.

    Returns:
        ``path``.
    """
    spec = tf.TensorSpec((1,) + tuple(model.input.shape[1:]), tf.float32, name=model.input.name)
    with tempfile.TemporaryDirectory() as saved_model_dir:
        archive = keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint(name="serve", fn=lambda x: model(x, training=False), input_signature=[spec])
        archive.write_out(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if representative_dataset is not None:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
        flatbuffer = converter.convert()
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(flatbuffer)
    os.replace(tmp, path)
    return path


def _load(kind, sidecar, weights):
    from snowgan.config import build
    from snowgan.models.discriminator import Discriminator
    from snowgan.models.generator import Generator

    config = build(sidecar)  # read-only: autosave stays off
    model = (Generator if kind == "generator" else Discriminator)(config)
    model.model.load_weights(weights)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m snowgan.export", description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", required=True, choices=["discriminator", "generator"],
                        help="Which model the sidecar and weights describe.")
    parser.add_argument("--sidecar", required=True, help="Path to the model's *_config.json; it is never rewritten.")
    parser.add_argument("--weights", required=True, help="Path to the .weights.h5 file to load.")
    parser.add_argument("--out", required=True, help="Output path: .tflite writes TFLite, .keras writes a Keras model.")
    parser.add_argument("--check", type=int, default=4,
                        help="Random inputs to compare the export against the Conv3D model on; 0 skips the check.")
    args = parser.parse_args(argv)

    model = _load(args.kind, args.sidecar, args.weights)
    m2 = to_conv2d(model)
    if args.check:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, (args.check,) + tuple(m2.input.shape[1:])).astype(np.float32)
        x3 = x[:, None] if args.kind == "discriminator" else x
        y3 = keras.ops.convert_to_numpy(model.model(x3, training=False))
        y3 = y3[:, 0] if args.kind == "generator" else y3
        diff = float(np.max(np.abs(y3 - keras.ops.convert_to_numpy(m2(x, training=False)))))
        print(f"max |conv3d - conv2d| over {args.check} inputs: {diff:.3e}")
        if diff > 1e-5:
            raise SystemExit(f"Export does not match the Conv3D model (max diff {diff:.3e} > 1e-5); not writing it.")
    if args.out.endswith(".tflite"):
        export_tflite(m2, args.out)
    else:
        m2.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
