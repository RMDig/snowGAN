"""Focused tests: boolean CLI flags can express "off" (plan 0.1).

Every boolean flag was ``action='store_true', default=None``, so omitting it
preserved the persisted config (correct, and the resume contract) but there was
**no value that turned it off**. Once a save_dir had ``spectral_norm: true``,
no command line could say otherwise. That made half of docs/experiments.md's
increment queue unrunnable — its rank-1 entry is written ``--no-adaptive_steps``,
a flag that did not exist.

Three properties are pinned here:
  1. ``--no-X`` overrides a persisted ``true``.
  2. ``--X`` still overrides a persisted ``false``.
  3. Omitting X leaves the persisted value untouched (the resume contract —
     the restart wrapper replays argv on every relaunch, so a flag that
     defaulted to False instead of None would silently flip settings mid-run).
"""

import copy
import json

import pytest

from snowgan.config import build, config_template, configure_generic
from snowgan.utils import parse_args


BOOLEAN_FLAGS = [
    "spectral_norm",
    "augment",
    "multiscale_disc",
    "adaptive_steps",
    "rebuild",
    "fade",
    "honor_splits",
    "clamp_gp_under_sn",
]


def _args(argv):
    import sys
    from unittest import mock

    with mock.patch.object(sys, "argv", ["snowgan", "--mode", "train"] + argv):
        return parse_args()


def _config(tmp_path, **overrides):
    data = copy.deepcopy(config_template)
    data.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return build(str(path))


@pytest.mark.parametrize("flag", BOOLEAN_FLAGS)
def test_no_form_overrides_persisted_true(tmp_path, flag):
    cfg = _config(tmp_path, **{flag: True})
    assert getattr(cfg, flag) is True, "precondition: persisted value is True"

    configure_generic(cfg, _args([f"--no-{flag}"]))
    assert getattr(cfg, flag) is False, f"--no-{flag} must turn {flag} off"


@pytest.mark.parametrize("flag", BOOLEAN_FLAGS)
def test_bare_form_overrides_persisted_false(tmp_path, flag):
    cfg = _config(tmp_path, **{flag: False})
    configure_generic(cfg, _args([f"--{flag}"]))
    assert getattr(cfg, flag) is True, f"--{flag} must turn {flag} on"


@pytest.mark.parametrize("flag", BOOLEAN_FLAGS)
@pytest.mark.parametrize("persisted", [True, False])
def test_omitting_flag_preserves_persisted_value(tmp_path, flag, persisted):
    """The resume contract. scripts/train_with_restarts.sh replays the same
    argv on every RSS restart; if omission meant False, every relaunch would
    silently flip the run's settings."""
    cfg = _config(tmp_path, **{flag: persisted})
    configure_generic(cfg, _args([]))
    assert getattr(cfg, flag) is persisted


def test_seed_flag_reaches_config(tmp_path):
    """--seed did not exist at all: config.seed was reachable only by editing
    JSON, so the reproducibility story had no CLI surface."""
    cfg = _config(tmp_path, seed=42)
    configure_generic(cfg, _args(["--seed", "1234"]))
    assert cfg.seed == 1234


def test_latent_dim_stays_int(tmp_path):
    """--latent_dim was type=float, and configure_generic assigned the raw arg
    *after* configure()'s int() cast — so passing it left 100.0 on the config
    and crashed model build at keras.Input(shape=(100.0,)) (UPGRADES #7)."""
    cfg = _config(tmp_path, latent_dim=100)
    configure_generic(cfg, _args(["--latent_dim", "256"]))
    assert cfg.latent_dim == 256
    assert isinstance(cfg.latent_dim, int)


def test_max_steps_reaches_config(tmp_path):
    cfg = _config(tmp_path)
    configure_generic(cfg, _args(["--max_steps", "2000"]))
    assert cfg.max_steps == 2000


def test_lr_decay_none_disables_the_schedule(tmp_path):
    """--lr_decay had choices=['cosine'] only, so a persisted schedule could
    not be turned off from the command line — omitting the flag preserves it,
    which is a different thing. The Phase 1 control needs the schedule OFF."""
    cfg = _config(tmp_path, lr_decay="cosine")
    configure_generic(cfg, _args(["--lr_decay", "none"]))
    assert cfg.lr_decay is None

    cfg = _config(tmp_path, lr_decay="cosine")
    configure_generic(cfg, _args([]))
    assert cfg.lr_decay == "cosine", "omission must preserve the persisted schedule"


def test_new_fields_round_trip_through_dump(tmp_path):
    cfg = _config(tmp_path)
    configure_generic(cfg, _args(["--no-honor_splits", "--grad_probe_interval", "7"]))
    dumped = cfg.dump()
    assert dumped["honor_splits"] is False
    assert dumped["grad_probe_interval"] == 7
    assert "clamp_gp_under_sn" in dumped and "max_steps" in dumped


def test_legacy_config_without_new_fields_loads(tmp_path):
    """Existing save_dirs predate these fields; loading one must not KeyError."""
    data = copy.deepcopy(config_template)
    for key in ("clamp_gp_under_sn", "grad_probe_interval", "max_steps", "honor_splits"):
        data.pop(key)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))

    cfg = build(str(path))
    assert cfg.clamp_gp_under_sn is False
    assert cfg.grad_probe_interval == 50
    assert cfg.max_steps == 0
    # Validity fix: defaults ON even for legacy configs. Training on the
    # held-out pools invalidates the downstream probe, and the plan accepts
    # breaking comparability with runs that were already invalid.
    assert cfg.honor_splits is True
