import os
import re
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from PIL import Image

from snowgan.modality import Modality

import ctypes
import ctypes.util


def _release_native_heap():
    """Return freed glibc arenas to the OS via malloc_trim.

    The per-image PIL/PNG-encoder buffers allocated while saving previews are
    freed by Python but glibc keeps the arenas, so process RSS ratchets up
    ~15 MiB on every preview (invisible to tracemalloc; gc.collect does not
    reclaim it — it is native, not a Python reference). malloc_trim(0) returns
    them to the OS. Linux/glibc only; a harmless no-op elsewhere.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def generate(generator, count = 1, seed_size = 100, save_dir = None, filename_prefix = 'synthetic', seed = None, gen_batch = 1):
    """
    Generate synthetic images using the currently loaded generator and save
    the images to the model's synthetic images folder.

    Function arguments:
        model (keras model) - GAN model to generate synthetic images from
        count (int) - Number of synthetic images to create
        seed_size (size) - Size of the input noise seed to the generator
        save_dir (str) - Folder path to save images, if none provided images are
        filename_prefix (str) - prefix to give the filename
        seed (tf.Tensor) - Optional fixed latent vector for consistent visualization
        gen_batch (int) - Generator forward sub-batch size. Previews run while
            the full training graph is resident in VRAM, so forwarding all
            `count` images at once spikes device memory with `count` — at 1024²
            a 10-wide forward materializes multi-GB upsample intermediates and
            OOMs. Chunking caps peak VRAM at `gen_batch` images regardless of
            `count`. Defaults to 1 (previews are infrequent; correctness over
            throughput).
    """

    # Check if the destimation exists
    if os.path.exists(save_dir) == False: # If folder doesn't exists
        print(f"Output folder doesn't exist, creating directory")
        os.makedirs(f"{save_dir}", exist_ok = True) # Create folder

    # Use provided seed or create a random one
    if seed is None:
        seed = tf.random.normal([count, seed_size])

    total = int(seed.shape[0])
    gen_batch = max(1, int(gen_batch))
    saved = 0

    # Forward the generator in sub-batches so peak device memory is bounded by
    # gen_batch, not by count (the n_samples-wide preview was the OOM driver).
    for start in range(0, total, gen_batch):
        synthetic_images = generator(seed[start:start + gen_batch], training = False)

        for ind in range(synthetic_images.shape[0]):
            # Construct image filename (1-based, running across all chunks)
            filepath = f"{save_dir}{filename_prefix}_{saved + 1}.png"

            image_arr = synthetic_images[ind].numpy()
            image_arr = (image_arr + 1.0) * 127.5
            image_arr = np.clip(image_arr, 0, 255).astype(np.uint8)

            # If depth dimension present, save each slice separately. Single-
            # modality runs (depth=1) collapse to one file with no per-modality
            # suffix — there's only one modality represented, so the suffix
            # would be redundant noise on the filesystem.
            if image_arr.ndim == 4:
                depth = image_arr.shape[0]
                if depth == 1:
                    image = Image.fromarray(image_arr[0])
                    save_image(image, filepath)
                else:
                    modality_by_index = {int(m): m.name.lower() for m in Modality}
                    for d in range(depth):
                        suffix = modality_by_index.get(d, f"view{d}")
                        arr = image_arr[d]
                        if suffix == Modality.CORE.name.lower():
                            arr = np.array(Image.fromarray(arr).resize((500, 300), Image.BICUBIC))
                        split_path = filepath.replace(".png", f"_{suffix}.png")
                        image = Image.fromarray(arr)
                        save_image(image, split_path)
            else:
                # Convert to PIL image
                image = Image.fromarray(image_arr)
                # Save image with parameters provided
                save_image(image, filepath)

            saved += 1

        # Drop the chunk's device tensor before the next forward so the BFC
        # pool can reuse it rather than growing to hold every chunk at once.
        del synthetic_images

    print(f"Synthetic images generated: {saved}")

    # Release the per-image PIL/numpy buffers, THEN return the freed native heap
    # to the OS. The PIL Image.fromarray + save path retains ~15 MiB of native
    # (libImaging/libpng/zlib) heap per preview that Python frees but glibc
    # keeps; on the preview cadence this ratchets process RSS up until long runs
    # OOM in CPU RAM. del-ing the last image/array first lets malloc_trim
    # reclaim them too. Both call sites (main.py and the trainer preview block)
    # discard the return value, so nothing needs these.
    # (Measured: preview RSS delta went from +9.4 MiB to net-negative.)
    try:
        del image, image_arr
    except (NameError, UnboundLocalError):
        pass
    _release_native_heap()

def save_image(image, filepath):
    # Ensure output dir exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert to image and save
    image.save(filepath)
    print(f"Saved image to {filepath}")

_PREVIEW_NAME = re.compile(r"(?:batch|step|epoch)_(\d+)_synthetic_(\d+)(?:_[a-z]+)?\.png$")


def preview_frames(save_dir, sample_index=None):
    """Preview images in training order, as ``[(step, path), ...]``.

    Parses the step out of the filename. The previous ``make_movie`` sorted on
    the LAST underscore segment, which is the *sample* index, not the step
    (``batch_74750_synthetic_3.png`` -> 3): every frame of a single-seed movie
    tied, and the tie fell back to string order, so ``batch_10000`` played
    before ``batch_2000`` and the training history ran scrambled.

    Args:
        save_dir: directory holding ``{batch|step}_N_synthetic_I.png`` files.
        sample_index: keep only seed ``I`` (the ``_3`` in ``..._synthetic_3``).
            ``None`` keeps all, ordered by step then sample.
    """
    frames = []
    for path in glob(os.path.join(save_dir, "*.png")):
        match = _PREVIEW_NAME.search(os.path.basename(path))
        if not match:
            continue
        step, sample = int(match.group(1)), int(match.group(2))
        if sample_index is None or sample == int(sample_index):
            frames.append((step, sample, path))
    frames.sort()
    return [(step, path) for step, _, path in frames]


def make_movie(save_dir = "outputs", videoname = "snowgan_synthetics.mp4", framerate = 15,
               filepath_pattern = None, sample_index = None, label_steps = False):
    """
    Create a .mp4 movie of the training history of synthetic images, to show
    the progression of what the snowGAN generator learned.

    Function arguments:
        save_dir (str) - Folder holding the preview images
        videoname (str) - Output filename, written inside save_dir unless absolute
        framerate (int) - Frames per second
        filepath_pattern (str) - Deprecated; ignored in favour of sample_index.
            Kept so old calls do not break.
        sample_index (int) - Render a single seed (e.g. 3 for ``*_synthetic_3``),
            which is what makes a coherent history: the tracking seed is fixed,
            so one index is one latent followed across training.
        label_steps (bool) - Burn the training step into each frame's corner.

    Returns:
        str: path of the written video.
    """
    # Lazy import: cv2 is only needed for video rendering, not for generate()
    # or for `import snowgan` in downstream consumers that never call make_movie.
    # Keeps opencv-python an optional install (see UPGRADES #29).
    import cv2

    # os.path.join, not f"{save_dir}{videoname}": without a trailing slash on
    # save_dir that glued the two into one filename next to the directory.
    video_path = videoname if os.path.isabs(videoname) else os.path.join(save_dir, videoname)

    frames = preview_frames(save_dir, sample_index)
    if not frames:
        raise FileNotFoundError(
            f"no preview images matching sample_index={sample_index} in {save_dir}")

    first = cv2.imread(frames[0][1])
    if first is None:
        raise IOError(f"could not read {frames[0][1]}")
    height, width = first.shape[:2]

    # H.264 plays everywhere (browsers, Windows Photos); mp4v often does not.
    # Fall back if this OpenCV build lacks an avc1 encoder.
    video, used_codec = None, None
    for codec in ("avc1", "mp4v"):
        candidate = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*codec), framerate, (width, height))
        if candidate.isOpened():
            video, used_codec = candidate, codec
            break
        candidate.release()
    if video is None:
        raise RuntimeError(f"OpenCV could not open a video writer for {video_path}")

    written = 0
    try:
        for step, image_file in frames:
            image = cv2.imread(image_file)
            if image is None:
                continue  # a torn/partial PNG from a killed run; skip, don't abort
            # VideoWriter silently drops frames whose size differs from the
            # first, so a mid-run resolution change would vanish without error.
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if label_steps:
                text = f"step {step:,}"
                scale = max(0.6, width / 900)
                thick = max(1, int(scale * 2))
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                cv2.rectangle(image, (8, 8), (20 + tw, 20 + th + 6), (0, 0, 0), -1)
                cv2.putText(image, text, (14, 14 + th), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (255, 255, 255), thick, cv2.LINE_AA)
            video.write(image)
            written += 1
    finally:
        video.release()

    # Pip-installed OpenCV usually ships without an H.264 encoder, so the
    # writer falls back to mp4v -- large (~350 MB for 1,580 frames at 1024px)
    # and unplayable in browsers and Windows Photos. If an ffmpeg binary is
    # available (imageio-ffmpeg bundles one), transcode to H.264 in place.
    if used_codec != "avc1":
        _transcode_to_h264(video_path)

    print(f"Wrote {written} frames ({written / framerate:.1f}s at {framerate} fps) to {video_path}")
    return video_path


def _transcode_to_h264(video_path):
    """Re-encode to H.264 in place if an ffmpeg binary can be found; else leave it."""
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            print("Note: no ffmpeg found; leaving the mp4v file as written. "
                  "`pip install imageio-ffmpeg` gives a smaller, widely playable H.264 file.")
            return
    tmp_path = video_path + ".h264.mp4"
    result = subprocess.run(
        [exe, "-y", "-loglevel", "error", "-i", video_path, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", tmp_path],
        capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(tmp_path):
        print(f"Warning: H.264 transcode failed, keeping mp4v file: {result.stderr.strip()[:200]}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return
    os.replace(tmp_path, video_path)
