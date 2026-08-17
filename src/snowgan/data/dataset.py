import random
from functools import cached_property

from datasets import load_dataset
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

            if sample_datatype == datatype:
                sample = self.dataset['train'][self.config.train_ind]

                image = sample['image']

                scaled_image = self.preprocess_image(image)

                # Add a depth dimension for single-view samples
                scaled_image = tf.expand_dims(scaled_image, axis=0)

                batch.append(scaled_image)
                print(f"Image {self.config.train_ind} added")

                # HF datasets accumulates PIL/decoded buffers in process RAM
                # on repeated indexed access (huggingface/datasets #4883, #7180).
                # Drop references eagerly so they're collectable before the
                # next iteration grabs another row.
                del sample, image, scaled_image

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

            # If we've found a core sample
            if sample_datatype == 0: 

                sample = self.dataset['train'][self.config.train_ind]

                core_image = self.preprocess_image(sample['image'])
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

                        if profile_meta.get('site') != sample['site']:
                            continue

                        if profile_meta.get('column') != sample['column']:
                            continue

                        if profile_meta.get('core') != sample['core']:
                            continue

                        # Preprocessing image
                        profile_sample = self.dataset['train'][profile_ind]
                        profile_image = self.preprocess_image(profile_sample['image'])
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
                print(f"Image add with core {self.config.train_ind} and profile {profile_ind} from segment {self.dataset['train'][profile_ind]['segment']}")

                # HF datasets accumulates PIL buffers in process RAM on
                # repeated indexed access (#4883, #7180). Drop references
                # eagerly so they're collectable before the next iteration.
                del sample, profile_sample, core_image, profile_image, merged_image

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
        print(f"Max - {tf.reduce_max(image).numpy()} | Min {tf.reduce_min(image).numpy()} ")
        
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
