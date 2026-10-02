"""Per-image preprocessing: the op the on-device pipeline must reproduce exactly.

TensorFlow + NumPy only, so a consumer (the AvApp's golden-image parity test)
can call the canonical reference without the HF dataset stack or a DataManager.
``DataManager.preprocess_image`` delegates here; there is one implementation.
"""

import numpy as np
import tensorflow as tf


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


def _to_rgb(image):
    """Coerce an image to 3 channels BEFORE resize.

    Order matters: ``tf.image.resize`` rejects rank-2 input, so a grayscale
    frame must gain its channel axis first. The models are built for
    ``channels=3``; a 1- or 4-channel tensor reaching them is a shape error at
    best and a silently different input at worst.
    """
    # PIL: let PIL do the mode conversion (handles L, LA, P, RGBA, CMYK ...).
    # RGBA -> RGB drops alpha without compositing, matching the tensor path.
    if hasattr(image, "mode") and hasattr(image, "convert"):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return tf.convert_to_tensor(np.array(image))

    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.asarray(image))
    rank = image.shape.rank
    if rank == 2:
        image = image[..., None]
    elif rank != 3:
        raise ValueError(f"preprocess_image expects one (H, W) or (H, W, C) image, got shape {image.shape}.")
    channels = image.shape[-1]
    if channels == 3:
        return image
    if channels == 1:
        return tf.image.grayscale_to_rgb(image)
    if channels == 4:
        return image[..., :3]
    raise ValueError(f"preprocess_image expects 1, 3 or 4 channels, got shape {image.shape}.")


def preprocess_image(image, resolution, *, mask_board=False):
    """Resize the whole frame, optionally mask the board, scale to [-1, 1].

    Args:
        image: one image in [0, 255] -- a PIL image, array or tensor of shape
            (H, W), (H, W, 1), (H, W, 3) or (H, W, 4).
        resolution: (height, width) to resize to.
        mask_board: replace the blue measurement board with neutral grey. Runs
            after the resize and BEFORE the rescale, while pixels are still in
            [0, 255] where the HSV thresholds are defined.

    Returns:
        float32 tensor of shape (height, width, 3) in [-1, 1].
    """
    image = _to_rgb(image)
    image = tf.image.resize(image, resolution)
    # NOTE: no per-image debug print here. `tf.reduce_max(...).numpy()` forces
    # a device sync on every image (~0.5 ms/img measured on an RTX 5080), so
    # the GPU cannot pipeline across images -- plus it floods stdout. This is
    # the hottest loop in the data pipeline; keep it sync-free. (UPGRADES #51;
    # CLAUDE.md §6: print is legacy.)
    if mask_board:
        image = mask_blue_board(image)
    return (tf.cast(image, tf.float32) / 127.5) - 1.0
