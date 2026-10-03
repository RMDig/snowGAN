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

Every rebuilt layer is float32 whatever the global Keras dtype policy is: this
is an inference artifact, and a mixed_float16 layer makes the TFLite converter
fail on fp16 ops (or, for BatchNorm, drift ~7e-4 from the original).

A bare ``keras.Model`` carries no config, so the mid-fade check only runs when
you pass the ``Generator`` wrapper.
"""

import argparse
import os
import tempfile

import keras
import numpy as np
import tensorflow as tf

from snowgan.models.generator import PixelNorm

_F32 = "float32"

# Gates for the CLI's self-check, relative to the output's scale. Keras
# Conv3D vs Conv2D differs only by summation order: 0 on most inputs, up to
# 2.4e-4 absolute (1e-7 relative) on the core critic, whose scores are ~2300.
# TFLite adds kernel differences (XNNPACK): measured 4.3e-5 relative on the
# magnified-profile critic at 1024^2.
KERAS_RTOL = 1e-5
TFLITE_RTOL = 1e-3


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
                use_bias=inner.use_bias, dtype=_F32, name=name)
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
                                   dtype=_F32, name=name),
                [inner.kernel.numpy()] + _bias(inner))
    if inner is not layer:
        raise TypeError(f"{name}: SpectralNormalization around {type(inner).__name__} is not supported.")
    if isinstance(layer, keras.layers.UpSampling3D):
        if layer.size[0] != 1:
            raise ValueError(f"{name}: upsamples the depth axis ({layer.size}).")
        return keras.layers.UpSampling2D(layer.size[1:], interpolation="nearest", dtype=_F32, name=name), None
    if isinstance(layer, keras.layers.BatchNormalization):
        axis = layer.axis if isinstance(layer.axis, int) else layer.axis[0]
        if axis not in (-1, 4):
            raise ValueError(f"{name}: BatchNormalization over axis {layer.axis}, not channels.")
        return (keras.layers.BatchNormalization(axis=-1, momentum=layer.momentum, epsilon=layer.epsilon,
                                                center=layer.center, scale=layer.scale, dtype=_F32, name=name),
                layer.get_weights())
    if isinstance(layer, PixelNorm):
        return PixelNorm(epsilon=layer.epsilon, dtype=_F32, name=name), None
    if isinstance(layer, keras.layers.LeakyReLU):
        return keras.layers.LeakyReLU(negative_slope=layer.negative_slope, dtype=_F32, name=name), None
    if isinstance(layer, keras.layers.Flatten):
        # Keeps name="features": the AvAI / snowGradient transfer tap.
        return keras.layers.Flatten(dtype=_F32, name=name), None
    if isinstance(layer, keras.layers.Reshape):
        if layer.target_shape[0] != 1:
            raise ValueError(f"{name}: reshapes to depth {layer.target_shape[0]}; only depth 1 is exportable.")
        return keras.layers.Reshape(layer.target_shape[1:], dtype=_F32, name=name), None
    raise TypeError(f"{name}: no rank-4 equivalent registered for {type(layer).__name__}. "
                    "Add one here (with an equivalence test) rather than skipping it.")


def to_conv2d(model, *, features_only=False, input_hw=None):
    """Rebuild a depth-1 snowGAN model at rank 4 with identical weights.

    Args:
        model: a ``Discriminator`` / ``Generator`` (or its ``.model``).
        features_only: discriminator only -- stop at the ``"features"`` layer,
            so the model outputs the backbone features rather than the critic
            score. This is what a downstream head attaches to.
        input_hw: discriminator with ``features_only`` only -- build at a
            different (height, width), e.g. (256, 256) for a phone. The conv
            weights are resolution-independent; the critic's Dense head is not,
            which is why this needs ``features_only``.

    Returns:
        A float32 ``keras.Model`` named ``<name>_conv2d``. Discriminator input
        ``(B, H, W, C)``; generator output ``(B, H, W, C)``. The
        discriminator's ``"features"`` layer is preserved.
    """
    m3 = _keras_model(model)
    in_shape = tuple(m3.input.shape[1:])
    is_image_input = len(in_shape) in (3, 4)
    if (features_only or input_hw is not None) and not is_image_input:
        raise ValueError("features_only / input_hw apply to the discriminator only.")
    if input_hw is not None and not features_only:
        raise ValueError("input_hw needs features_only=True: the critic's Dense head is sized for one resolution.")
    if len(in_shape) == 4:
        if in_shape[0] != 1:
            raise ValueError(f"{m3.name}: input depth {in_shape[0]}; the Flatten->Dense head mixes depth "
                             "slices, so only depth 1 has an exact rank-4 equivalent.")
        in_shape = in_shape[1:]
    if input_hw is not None:
        in_shape = (int(input_hw[0]), int(input_hw[1])) + in_shape[2:]
    inputs = keras.Input(shape=in_shape, dtype=_F32, name="image" if is_image_input else "latent")
    x = inputs
    reached_features = False
    for layer in _chain(m3):
        new, weights = _convert(layer)
        x = new(x)
        if weights is not None:
            new.set_weights(weights)
        if features_only and layer.name == "features":
            reached_features = True
            break
    if features_only and not reached_features:
        raise ValueError(f"{m3.name}: no layer named 'features' to stop at.")
    suffix = "_features_conv2d" if features_only else "_conv2d"
    return keras.Model(inputs, x, name=f"{m3.name}{suffix}")


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


def run_tflite(path, x):
    """Run a .tflite file on one batch-1 input; returns the first output."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], np.asarray(x, np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])


def _rel_diff(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if not np.all(np.isfinite(candidate)):
        return float("inf")
    return float(np.max(np.abs(reference - candidate)) / max(1.0, float(np.max(np.abs(reference)))))


def _load(kind, sidecar, weights):
    from snowgan.config import build
    from snowgan.models.discriminator import Discriminator
    from snowgan.models.generator import Generator

    # build() falls back to the default template when the path is missing,
    # which would export a randomly configured model; refuse instead.
    if not os.path.isfile(sidecar):
        raise SystemExit(f"Sidecar not found: {sidecar}")
    config = build(sidecar)  # read-only: autosave stays off
    model = (Generator if kind == "generator" else Discriminator)(config)
    model.model.load_weights(weights)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m snowgan.export", description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", required=True, choices=["discriminator", "generator"],
                        help="Which model the sidecar and weights describe.")
    parser.add_argument("--sidecar", required=True, help="Path to the model's *_config.json; it is never rewritten.")
    parser.add_argument("--weights", required=True,
                        help="Path to the .weights.h5 file to load. For a generator, a release's "
                             "generator_ema.weights.h5 (EMA shadow) is usually the one to ship.")
    parser.add_argument("--out", required=True, help="Output path: .tflite writes TFLite, .keras writes a Keras model.")
    parser.add_argument("--features", action="store_true",
                        help="Discriminator only: export the backbone up to the 'features' layer instead "
                             "of the critic score.")
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="With --features: build the backbone at this input size (e.g. 256 256) "
                             "instead of the sidecar's.")
    parser.add_argument("--check", type=int, default=4,
                        help="Random inputs (each run at batch 1, the deployment size) to compare the export "
                             "against the Conv3D model on; a .tflite output is also run on one. 0 skips.")
    args = parser.parse_args(argv)
    if args.resolution and not args.features:
        parser.error("--resolution requires --features")

    model = _load(args.kind, args.sidecar, args.weights)
    m2 = to_conv2d(model, features_only=args.features, input_hw=args.resolution)

    # The Conv3D reference for the check: the same weights at the export's resolution.
    reference = model.model
    if args.features:
        m3 = keras.Model(model.model.input, model.model.get_layer("features").output)
        if args.resolution:
            reference = None  # no Conv3D model at this size; checked via the native-size export
            native = to_conv2d(model, features_only=True)
        else:
            reference = m3

    xs = []
    if args.check:
        rng = np.random.default_rng(0)
        shape = (1,) + tuple(m2.input.shape[1:])
        for _ in range(args.check):
            x = (rng.uniform(-1, 1, shape) if args.kind == "discriminator" else rng.standard_normal(shape))
            xs.append(x.astype(np.float32))
        if reference is not None:
            worst = 0.0
            for x in xs:
                x3 = x[:, None] if args.kind == "discriminator" else x
                y3 = keras.ops.convert_to_numpy(reference(x3, training=False))
                y3 = y3[:, 0] if args.kind == "generator" else y3
                worst = max(worst, _rel_diff(y3, keras.ops.convert_to_numpy(m2(x, training=False))))
            print(f"max relative |conv3d - conv2d| over {len(xs)} batch-1 inputs: {worst:.3e}")
            if worst > KERAS_RTOL:
                raise SystemExit(f"Export does not match the Conv3D model ({worst:.3e} > {KERAS_RTOL}); not writing it.")
        else:
            # Same weights, same layers as the native-size features export; check the rebuild
            # at native size, then that the resized model shares those weights exactly.
            for a_w, b_w in zip(native.get_weights(), m2.get_weights()):
                if a_w.shape != b_w.shape or not np.array_equal(a_w, b_w):
                    raise SystemExit("Resized features export does not carry the native weights.")
            print(f"features at {tuple(args.resolution)}: weights identical to the native-size export")

    if args.out.endswith(".tflite"):
        export_tflite(m2, args.out)
        if xs:
            y_keras = keras.ops.convert_to_numpy(m2(xs[0], training=False))
            diff = _rel_diff(y_keras, run_tflite(args.out, xs[0]))
            print(f"max relative |keras - tflite| on 1 input: {diff:.3e}")
            if diff > TFLITE_RTOL:
                os.remove(args.out)
                raise SystemExit(f"TFLite output does not match ({diff:.3e} > {TFLITE_RTOL}); removed {args.out}.")
    else:
        m2.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
