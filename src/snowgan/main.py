import os
import sys

from snowgan.trainer import Trainer
from snowgan.generate import generate
from snowgan.models.generator import load_generator
from snowgan.models.discriminator import load_discriminator
from snowgan.config import configure_disc, configure_gen, build
from snowgan.checkpoint import resolve_weights_path
from snowgan.utils import parse_args, configure_device, set_seed


_INFER_MOVED_MESSAGE = (
    "Inference has moved out of snowGAN. The transfer-learning pipeline "
    "(avalanche / wind-loading heads on top of the trained discriminator "
    "backbone) now lives in the AvAI library: "
    "https://github.com/dennys246/AvAI"
)


def main():
    args = parse_args()
    if args.mode == "infer":
        print(_INFER_MOVED_MESSAGE, file=sys.stderr)
        raise SystemExit(1)

    # Configure TensorFlow runtime (GPU/CPU, XLA, mixed precision)
    try:
        configure_device(args)
    except Exception as e:
        print(f"Warning: device configuration failed: {e}")

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    gen_config_path = os.path.join(args.save_dir, "generator_config.json")
    # autosave=True: this process owns these sidecars and is expected to write
    # them back. Readers (snowGradient, AvAI) get the default autosave=False.
    gen_config = build(gen_config_path, autosave=(args.mode == "train"))
    gen_config = configure_gen(gen_config, args)
    # Persist configured generator settings (including fade/fade_steps) immediately
    gen_config.save_config(gen_config_path)

    disc_config_path = os.path.join(args.save_dir, "discriminator_config.json")
    disc_config = build(disc_config_path, autosave=(args.mode == "train"))
    disc_config = configure_disc(disc_config, args)
     # Persist configured discriminator settings immediately
    disc_config.save_config(disc_config_path)

    # Seed RNGs before any model construction so weight init, noise
    # sampling, and dataset shuffling are reproducible across runs with
    # matching configs (UPGRADES #12).
    set_seed(gen_config.seed)
    print(f"Seed set to {gen_config.seed}")

    # Load the generator when training or generating samples
    generator = load_generator(gen_config.checkpoint, gen_config)

    if args.mode == "train":
        # Load the discriminator
        discriminator = load_discriminator(disc_config.checkpoint, disc_config)

        # Call to trainer
        trainer = Trainer(generator, discriminator)
        trainer.train(batch_size = args.batch_size, epochs = args.epochs)

    elif args.mode == "generate":
        # `load_generator` builds the architecture but does not load
        # weights — without this, `--mode generate` would emit noise from
        # a fresh init. Mirror Trainer.__init__'s resolve+load_weights so
        # `--mode generate --save_dir keras/snowgan/batch_<N>/` actually
        # produces samples reflecting the trained generator. Prefer EMA
        # shadow weights when present (eval-grade); fall back to the
        # primary checkpoint.
        ema_path = os.path.join(os.path.dirname(gen_config.checkpoint),
                                "generator_ema.weights.h5")
        weights_loaded = False
        if os.path.exists(ema_path):
            try:
                generator.model.load_weights(ema_path)
                print(f"Generator EMA weights loaded from {ema_path}")
                weights_loaded = True
            except Exception as e:
                print(f"Warning: failed to load EMA weights ({e}); falling back to primary checkpoint.")
        if not weights_loaded:
            gen_weights_path = resolve_weights_path(gen_config.checkpoint)
            if gen_weights_path is not None:
                generator.model.load_weights(gen_weights_path)
                print(f"Generator weights loaded from {gen_weights_path}")
            else:
                print(
                    f"Warning: no generator weights found near {gen_config.checkpoint}. "
                    f"Output will reflect random init — point --save_dir at a snapshot dir "
                    f"(e.g. keras/snowgan/batch_<N>/) that contains generator.weights.h5."
                )

        synthetic_dir = os.path.join(gen_config.save_dir, "synthetic_images")
        _ = generate(generator,
                     count=args.n_samples,
                     seed_size=gen_config.latent_dim,
                     save_dir=f"{synthetic_dir}/")

if __name__ == "__main__":

    main()

