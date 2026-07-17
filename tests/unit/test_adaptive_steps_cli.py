"""Focused test: --no-adaptive_steps can clear a persisted adaptive_steps=True.

`adaptive_steps` persists into the config, but the flag was `store_true` with
`default=None`. Omitting it left `args.adaptive_steps=None`, which
`configure_generic` treats as "no override" — so once True was written to the
config JSON there was no CLI path back to False, only hand-editing the file.
These pin the three-state contract: on / off / leave the saved value alone.
"""

import copy
import json

from snowgan.config import build, config_template, configure_generic
from snowgan.utils import parse_args


def _config(tmp_path, adaptive_steps):
    data = copy.deepcopy(config_template)
    data["adaptive_steps"] = adaptive_steps
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))
    return build(str(cfg_path))


def _args(monkeypatch, *flags):
    monkeypatch.setattr("sys.argv", ["snowgan", "--mode", "train", *flags])
    return parse_args()


def test_no_adaptive_steps_clears_persisted_true(tmp_path, monkeypatch):
    cfg = _config(tmp_path, adaptive_steps=True)
    configure_generic(cfg, _args(monkeypatch, "--no-adaptive_steps"))
    assert cfg.adaptive_steps is False


def test_adaptive_steps_sets_it(tmp_path, monkeypatch):
    cfg = _config(tmp_path, adaptive_steps=False)
    configure_generic(cfg, _args(monkeypatch, "--adaptive_steps"))
    assert cfg.adaptive_steps is True


def test_omitting_the_flag_keeps_the_saved_value(tmp_path, monkeypatch):
    """Neither flag means "don't override" — resume must not silently flip it."""
    cfg = _config(tmp_path, adaptive_steps=True)
    configure_generic(cfg, _args(monkeypatch))
    assert cfg.adaptive_steps is True

    cfg = _config(tmp_path, adaptive_steps=False)
    configure_generic(cfg, _args(monkeypatch))
    assert cfg.adaptive_steps is False
