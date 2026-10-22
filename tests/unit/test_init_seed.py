"""init_seed: a same-split replicate (capacity_plan, E0b).

`seed` fixes the split (derive_splits) and the in-training KID latents
(seed + 2). Changing it changes what a run trains and is scored on, which is
why the seed-43 replicates could only be compared within a seed. `init_seed`
re-seeds only the global RNGs (weight init, training-time noise), so two runs
differing only in it share the split and the KID latents.
"""

import copy
import json
import random
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np

from snowgan.utils import training_seed


def test_defaults_to_seed():
    assert training_seed(SimpleNamespace(seed=42)) == 42
    assert training_seed(SimpleNamespace(seed=42, init_seed=None)) == 42


def test_init_seed_overrides_only_the_training_seed():
    assert training_seed(SimpleNamespace(seed=42, init_seed=7)) == 7


def test_split_rng_ignores_init_seed_and_global_state():
    """derive_splits seeds its own random.Random from config.seed; the global
    RNG state that init_seed changes cannot reach it."""
    def split_draws(global_seed):
        random.seed(global_seed)
        np.random.seed(global_seed)
        rng = random.Random(42)  # what derive_splits does with config.seed = 42
        return [rng.random() for _ in range(5)]
    assert split_draws(42) == split_draws(7)


def test_cli_round_trip_and_resume(tmp_path):
    from snowgan.config import build, config_template, configure_generic
    from snowgan.utils import parse_args

    def args(argv):
        with mock.patch.object(sys, "argv", ["snowgan", "--mode", "train"] + argv):
            return parse_args()

    path = tmp_path / "config.json"
    path.write_text(json.dumps(copy.deepcopy(config_template)))
    cfg = build(str(path))
    assert cfg.init_seed is None

    configure_generic(cfg, args(["--init_seed", "1042"]))
    path.write_text(json.dumps(cfg.dump()))
    cfg = build(str(path))
    configure_generic(cfg, args([]))
    assert (cfg.seed, cfg.init_seed, training_seed(cfg)) == (42, 1042, 1042)
