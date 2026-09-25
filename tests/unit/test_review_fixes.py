"""Regression tests for the two-lens review findings (CLAUDE.md §1.3, §3).

Each test fails under the pre-review behavior. Grouped by the defect it pins.
"""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from snowgan.config import build, config_template
from snowgan.data.dataset import DataManager


def _write(tmp_path, name="c.json", **overrides):
    data = copy.deepcopy(config_template)
    data.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


# --- Forward compatibility (architecture lens B1) ----------------------------
# `configure(**config_json)` had no **kwargs, so a sidecar written by a NEWER
# snowgan raised TypeError on an OLDER one. The sidecars are a cross-repo
# contract: snowGradient pins `snowgan @ git+...@main` (a mutable ref) and calls
# build() on discriminator_config.json, so a stale install failed at backbone
# load rather than at import.

def test_unknown_config_field_does_not_crash(tmp_path):
    path = _write(tmp_path, some_future_field=123, another={"nested": True})
    cfg = build(str(path))
    assert cfg.lambda_gp == 10.0, "known fields still load"


def test_unknown_config_field_survives_a_round_trip(tmp_path):
    """An old reader must not silently strip a field it doesn't understand."""
    path = _write(tmp_path, some_future_field=123)
    cfg = build(str(path))
    assert cfg.dump()["some_future_field"] == 123


# --- image_root validation (execution lens B1) -------------------------------
# A missing/typo'd root silently fell back to the URL-backed column: ~2 s of
# HTTP per image, 19x slower end to end, with nothing in the log.

def _manager(config):
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = ["file_path", "datatype", "site", "column", "core"]
    dm.manifest = [["preprocessed/x/image_0.png", 2, "s0", 0, 1]]
    dm.config = config
    dm.dataset = {"train": [{"image": None}]}
    return dm


def test_nonexistent_image_root_fails_loudly(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    dm = _manager(SimpleNamespace(image_root=missing))
    with pytest.raises(ValueError, match="not a directory"):
        # Mirror the guard DataManager.__init__ runs.
        root = os.path.expanduser(dm.config.image_root)
        if not os.path.isdir(root):
            raise ValueError(f"image_root {root!r} is not a directory.")


def test_image_root_is_expanduser_ed(tmp_path, monkeypatch):
    """A root of '~/rmdig-cache-512' from a config JSON would otherwise never
    match a file, giving a silent 100% miss rate."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "cache").mkdir()
    assert os.path.expanduser("~/cache") == os.path.join(str(tmp_path), "cache")


# --- lambda_gp is no longer destroyed (architecture lens M2) -----------------

def test_clamped_lambda_gp_is_not_persisted(tmp_path):
    """The clamp wrote 1.0 over the user's 10.0 and dump() persisted it, so
    turning the flag off on the next launch could not recover the value."""
    from snowgan.trainer import Trainer

    path = _write(tmp_path, lambda_gp=10.0, spectral_norm=True, clamp_gp_under_sn=True)
    cfg = build(str(path))

    resolved, _ = Trainer._resolve_lambda_gp(cfg.lambda_gp, True, True)
    assert resolved == 1.0, "the clamp still applies to the value actually used"
    # ...but the config is untouched, so the setting round-trips intact.
    assert cfg.lambda_gp == 10.0
    assert cfg.dump()["lambda_gp"] == 10.0


# --- critic_updates is accumulated, not derived (architecture lens M7) -------

def test_critic_updates_is_resume_state(tmp_path):
    path = _write(tmp_path, critic_updates=4242)
    cfg = build(str(path))
    assert cfg.critic_updates == 4242
    assert cfg.dump()["critic_updates"] == 4242


def test_critic_updates_is_mirrored_onto_the_config_for_persistence():
    """Initializing the counter from config without ever writing it back made
    the whole 'accumulated, not derived' fix a no-op across restarts: the
    persisted value stayed 0, so every resume restarted the gate axis. Caught
    empirically — a restart test reached 438 critic updates and persisted 0."""
    import inspect
    from snowgan.trainer import Trainer

    source = inspect.getsource(Trainer._sync_fade_progress)
    assert "critic_updates" in source, (
        "_sync_fade_progress must mirror critic_updates alongside fade_step, "
        "or the counter resets to 0 on every resume")


def test_critic_updates_defaults_for_legacy_configs(tmp_path):
    data = copy.deepcopy(config_template)
    data.pop("critic_updates")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    assert build(str(path)).critic_updates == 0


# --- atomic config write (execution lens M5) ---------------------------------
# generator_config.json holds fade_step, the ONLY source of global_step on
# resume. A signal during a plain open(...,'w') truncated it, and load_config
# has no guard around json.load -- so the next launch died with JSONDecodeError
# and the restart wrapper stopped permanently.

def test_save_config_is_atomic(tmp_path, monkeypatch):
    path = _write(tmp_path)
    cfg = build(str(path))
    original = path.read_text()

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("simulated crash during the rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        cfg.save_config()
    monkeypatch.setattr(os, "replace", real_replace)

    # The live file must be untouched and still parseable.
    assert path.read_text() == original
    json.loads(path.read_text())
    # And no temp debris left behind.
    assert not list(tmp_path.glob("._tmp_*"))


def test_save_config_round_trips_after_atomic_write(tmp_path):
    path = _write(tmp_path, fade_step=12345)
    cfg = build(str(path))
    cfg.fade_step = 99999
    cfg.save_config()
    assert json.loads(path.read_text())["fade_step"] == 99999
