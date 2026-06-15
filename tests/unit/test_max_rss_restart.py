"""Focused tests for the --max_rss_mb restart-exit plumbing.

The trainer self-exits with a sentinel code when RSS crosses --max_rss_mb so a
wrapper can relaunch a clean process (workaround for the native CPU-RAM leak).
These pin the sentinel code and that the config field round-trips.
"""

import copy
import json
import os

from snowgan.config import build, config_template
from snowgan.trainer import _RESTART_EXIT_CODE


def test_restart_exit_code_is_stable():
    # The wrapper (scripts/train_with_restarts.sh) hard-codes 75; keep in sync.
    assert _RESTART_EXIT_CODE == 75


def test_max_rss_mb_round_trips(tmp_path):
    data = copy.deepcopy(config_template)
    data["max_rss_mb"] = 24000
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))
    cfg = build(str(cfg_path))
    assert cfg.max_rss_mb == 24000.0
    assert cfg.dump()["max_rss_mb"] == 24000.0


def test_max_rss_mb_defaults_disabled(tmp_path):
    data = copy.deepcopy(config_template)
    data.pop("max_rss_mb", None)  # legacy config without the field
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))
    cfg = build(str(cfg_path))
    assert cfg.max_rss_mb == 0.0


def test_wrapper_script_present_and_references_code():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wrapper = os.path.join(root, "scripts", "train_with_restarts.sh")
    assert os.path.exists(wrapper)
    text = open(wrapper).read()
    assert "RESTART_CODE=75" in text  # must match _RESTART_EXIT_CODE
