import argparse
import os
import random
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
import tensorflow as tf
from pathlib import Path

from snowgan.config import build as configuration


def set_seed(seed: int) -> None:
    """Pin RNG state across Python, NumPy, and TensorFlow for reproducibility.

    UPGRADES #12: training runs were not replayable across same-host
    restarts because no seed was applied to any RNG. This function fixes
    the library-level seeding (``random``, ``numpy``, ``tf.random``) and
    sets ``PYTHONHASHSEED`` / ``TF_DETERMINISTIC_OPS`` for completeness.

    The two env vars only take *full* effect when set before the Python
    interpreter and TensorFlow are started — Python's hash randomization
    is initialized at interpreter startup, and TF caches the determinism
    flag at import time. Setting them inside an already-running process
    is best-effort; UPGRADES #4 tracks the bootstrap module that would
    set these before any TF import.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def upscale_layer(layer, scale=2, method="nearest"):
    """ 
    Create an upscaled layer for fading in
    """
    if isinstance(scale, int):
        scale_h, scale_w = scale, scale
    else:
        scale_h, scale_w = scale

    shape = tf.shape(layer)
    h = shape[1]
    w = shape[2]

    new_h = tf.cast(h * scale_h, tf.int32)
    new_w = tf.cast(w * scale_w, tf.int32)
    new_size = tf.stack([new_h, new_w])

    resized = tf.image.resize(layer, size=new_size, method=method)
    return tf.cast(resized, layer.dtype)

def merge_last_layer(list_of_layers, alpha, scale=2):
    """
    Blend the last two layers in a progressive GAN:
    - Upscales the second-to-last layer
    - Blends it with the last layer by alpha ∈ [0,1]
    """
    upscaled = upscale_layer(list_of_layers[-2], scale)
    alpha_t = tf.convert_to_tensor(alpha, dtype=tf.float32)
    one_minus = tf.cast(1.0, tf.float32) - alpha_t
    
    upscaled = tf.cast(upscaled, tf.float32)
    curr = tf.cast(list_of_layers[-1], tf.float32)
    blended = one_minus * upscaled + alpha_t * curr
    return tf.cast(blended, curr.dtype)

def compute_fade_alpha(current_step, fade_steps):
    """
    Compute alpha in [0, 1] for progressive fade‑in.

    Args:
        current_step: integer step within the current resolution phase.
        fade_steps: number of steps to ramp alpha from 0 -> 1.
 
    Returns:
        tf.float32 alpha in [0, 1].
    """
    current = tf.cast(current_step, tf.float32)
    total = tf.cast(tf.maximum(1, fade_steps), tf.float32)
    return tf.clip_by_value(current / total, 0.0, 1.0)

def configure_device(args):
    # Disable XLA unless explicitly requested
    if not getattr(args, "xla", False):
        os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0")

    # Configure tensorflow
    if hasattr(args, "device") and args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Enable memory growth for all GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    # Do not force-disable GPUs unconditionally; respect the selected device
    if args.xla == True: # Use XLA computation for faster runtime operations
        tf.config.optimizer.set_jit(True)
    else:
        tf.config.optimizer.set_jit(False)  # Explicitly keep XLA off unless requested
    if args.mixed_precision == True : # Use mixed precision for faster training
        set_global_policy("mixed_float16")


def parse_args():
    # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The snowGAN model is used to train a GAN on a dataset of snow samples magnified on a crystal card. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained snowGAN to accomplish transfer learning on rebuild GAN tasks!")

    # Add command-line arguments
    parser.add_argument('--mode', type = str, choices = ["train", "generate", "infer"], required = True, help = "Mode to run the model in: train the GAN or generate synthetics. ('infer' is retained as a redirect — inference moved to the AvAI library.)")
    parser.add_argument('--dataset_dir', type = str, default = 'rmdig/rocky_mountain_snowpack', help = "Path to the Rocky Mountain Snowpack dataset, if none provided it will download directly from HF remote repository")
    parser.add_argument('--save_dir', type = str, default = "keras/snowgan/", help = "Path to save results where a pre-trained model may be found (defaults to keras/snowgan/)")
    
    parser.add_argument('--rebuild', action='store_true', default=None, help='Initialize fresh models and IGNORE any saved weights in save_dir (start from scratch over an existing checkpoint dir without archiving it). Defaults to off.')
    
    parser.add_argument('--device', type = str, choices = ["cpu", "gpu"], default = "gpu", help = 'Device to run the model on (defaults to gpu)')
    parser.add_argument('--xla', action='store_true', default = False, help = 'Whether to use accelerated linear algebra (XLA) (defaults to False)')
    parser.add_argument('--mixed_precision', action='store_true', default = False, help = 'Use mixed_float16 precision to halve activation VRAM (safe to enable mid-training)')

    parser.add_argument('--resolution', type = set, help = 'Resolution to downsample images too (Default set to (1024, 1024))')
    parser.add_argument('--n_samples', type = int, default = 10, help = "Number of synthetic images to generate (defaults to 10)")
    parser.add_argument('--batch_size', type = int, default = 4, help = 'Batch size (Defaults to 8)')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Epochs to train on (Defaults to 10)')
    parser.add_argument('--latent_dim', type = float, default = 100, help = 'Latent dimension size (Defaults to 100)')
    parser.add_argument('--cleanup_milestone', type = int, default = 1000, help = 'Frequency (in batches) to save checkpoints and cleanup older batches (Defaults to 1000)')
    # Use None defaults so resume can rely on persisted config unless explicitly overridden
    parser.add_argument('--fade', type = bool, default = None, help = 'Enable progressive fade-in between resolutions (set True/False to override config)')
    parser.add_argument('--fade_steps', type = int, default = None, help = 'Steps to ramp alpha from 0 to 1 during fade-in (override config if set)')

    parser.add_argument('--gen_checkpoint', type = str, help = 'Path to a pre-trained generator model to load')
    parser.add_argument('--gen_kernel', type = str, help = 'Generator kernel size (Defaults to [5, 5])')
    parser.add_argument('--gen_stride', type = str, help = 'Generator kernel stride (Defaults to [2, 2])')
    parser.add_argument('--gen_lr', type = float, help = 'Generators optimizer learning rate (Defaults to 0.001)')
    parser.add_argument('--gen_beta_1', type = float, help = 'Generators optimizer adam beta one (Defaults to 0.5)')
    parser.add_argument('--gen_beta_2', type = float, help = 'Generators optimizer adam beta two (Defaults to 0.9)')
    parser.add_argument('--gen_negative_slope', type = float, help = 'Generators negative slope for leaky relu (Defaults to 0.25)')
    parser.add_argument('--gen_steps', type = int, help = 'Training steps the generator takes per batch (Defaults to 5)')
    parser.add_argument('--gen_filters', type = str, help = 'Generators filters per convolution layer (Defaults to [1024, 512, 256, 128, 64])')
    parser.add_argument('--gen_norm', type = str, default = None, choices = ['pixel', 'batch', 'none'], help = "Generator normalization: 'pixel' (PixelNorm, GP-safe, recommended for WGAN), 'batch' (BatchNorm), or 'none'. Default: none (legacy).")
    parser.add_argument('--gen_upsampler', type = str, default = None, choices = ['resize', 'transpose'], help = "Generator upsampler: 'resize' (UpSampling+conv, no checkerboard but low-pass/blob-prone) or 'transpose' (learned Conv3DTranspose, recovers fine texture; use --gen_kernel 4 4 to keep checkerboard mild). Default: resize.")

    parser.add_argument('--disc_checkpoint', type = str, help = "Path to a pre-trained discriminator model to load")
    parser.add_argument('--disc_kernel', type = str, help = 'Discriminator kernel size (Defaults to [5, 5])')
    parser.add_argument('--disc_stride', type = str, help = 'Discriminator kernel stride (Defaults to [2, 2])')
    parser.add_argument('--disc_lr', type = float, help = 'Discriminators learning rate (Defaults to 0.0001)')
    parser.add_argument('--disc_beta_1', type = float, help = 'Discriminators adam beta one (Defaults to 0.5)')
    parser.add_argument('--disc_beta_2', type = float, help = 'Discriminators dam beta two (Defaults to 0.9)')
    parser.add_argument('--disc_negative_slope', type = float, help = 'Discriminators negative slope for leaky relu (Defaults to 0.25)')
    parser.add_argument('--disc_lambda_gp', type = float, default = None, help = 'Discriminator gradient-penalty weight. 0 disables GP (spectral-norm-only critic). Defaults to 10.0')
    parser.add_argument('--disc_steps', type = int, help = 'Steps the discriminator takes per batch (Defaults to 1)')
    parser.add_argument('--disc_filters', type = str, help = 'Discriminator filters per convolution layer (Defaults to [64, 128, 256, 512, 1024])')

    # Post-progressive training improvements
    parser.add_argument('--spectral_norm', action='store_true', default=None, help='Enable spectral normalization on the discriminator')
    parser.add_argument('--augment', action='store_true', default=None, help='Enable differentiable augmentation during training')
    parser.add_argument('--lr_decay', type=str, default=None, choices=['cosine'], help='Learning rate decay schedule (e.g. "cosine")')
    parser.add_argument('--lr_min', type=float, default=None, help='Minimum learning rate for LR decay (Defaults to 1e-7)')
    parser.add_argument('--lr_decay_steps', type=int, default=None, help='Cosine decay horizon in steps. Set to the planned run length so LRs reach lr_min at end-of-training, not partway through (0/unset = long-horizon fallback)')
    parser.add_argument('--ema_decay', type=float, default=None, help='EMA decay for generator shadow weights (e.g. 0.999, 0 to disable)')
    parser.add_argument('--fid_interval', type=int, default=None, help='Steps between FID evaluations (0 to disable)')
    parser.add_argument('--multiscale_disc', action='store_true', default=None, help='Enable multi-scale discriminator (adds 256x256 head)')
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='Global gradient norm clipping (0 to disable)')
    parser.add_argument('--max_rss_mb', type=float, default=None, help='Process RSS ceiling in MiB. When exceeded, the trainer saves a fresh checkpoint and exits with code 75 so a restart wrapper relaunches a clean process (workaround for the native CPU-RAM leak that OOM-kills long runs). 0/unset disables.')
    parser.add_argument('--ada_target', type=float, default=None, help='ADA target disc accuracy on reals (e.g. 0.6, 0 to disable)')
    # BooleanOptionalAction, not store_true: `adaptive_steps` persists into the
    # config, and a store_true flag can only ever set it. Omitting the flag left
    # args.adaptive_steps=None, which configure_generic reads as "no override" —
    # so a persisted True could not be cleared from the CLI at all, only by
    # hand-editing the config JSON. --no-adaptive_steps is the off switch.
    parser.add_argument('--adaptive_steps', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable adaptive disc/gen training step ratio (--no-adaptive_steps to disable; '
                             'omit to keep whatever the saved config has). Off by default on a fresh run.')

    # Modality selection: which depth-axis arrangement the trainer feeds the
    # GAN. "magnified_profile" / "core" / "profile" / "crystal_card" produce
    # depth-1 batches of that single modality; "merged" stacks core+profile
    # at depth=2 (the modality-blending experiment). Defaults to single
    # modality so each model is focused on generative quality of one view.
    parser.add_argument('--modality', type=str, default=None,
                        choices=['magnified_profile', 'core', 'profile', 'crystal_card', 'merged'],
                        help='Which modality the trainer feeds the GAN. Default: magnified_profile.')
    parser.add_argument('--sample_epoch_interval', type=int, default=None,
                        help='Generate seeded preview samples every N epochs (0 to disable). Default: 1.')
    parser.add_argument('--sample_batch_interval', type=int, default=None,
                        help='Generate seeded preview samples every N batches during an epoch (0 to disable). '
                             'Useful when one epoch spans many checkpoints and the epoch-end preview never fires. '
                             'Default: 0.')

    # Parse the arguments
    return parser.parse_args()

def get_repo_root(start: str = ".") -> Path:
    path = Path(start).resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("No .git directory found")
