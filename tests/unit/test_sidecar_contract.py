"""The sidecar is a cross-repo contract. These tests pin the parts consumers rely on.

snowGradient loads released `discriminator_config.json` straight out of the
shared HuggingFace cache and calls `snowgan.config.build()` on it; AvAI does the
same. Everything here exists because a change on snowGAN's side surfaced (or
would surface) as a failure inside *their* loader.

Sources: snowGradient's 2026-09-25 alignment report after snowGAN PR #31.
"""

import atexit
import copy
import json

import pytest

from snowgan.config import build, config_template


def _write(tmp_path, name="discriminator_config.json", **overrides):
    data = copy.deepcopy(config_template)
    data.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


# --- autosave is opt-in ------------------------------------------------------
# `save_config` writes back to config_filepath. Registering that at exit for
# every build() meant a read-only consumer rewrote the sidecar it had just
# loaded. snowGradient sets resolution_override for its probes, so a released
# artifact in the SHARED HuggingFace cache was silently rewritten from
# [1024,1024] to [64,64] -- corrupting it for every other consumer on the box.

def test_reader_does_not_register_an_atexit_writeback(tmp_path):
    path = _write(tmp_path)
    cfg = build(str(path))

    assert cfg._autosave_registered is False
    # atexit.unregister is a no-op for something never registered; the real
    # assertion is that a plain reader leaves no hook behind at all.
    atexit.unregister(cfg.save_config)


def test_owner_can_opt_in(tmp_path):
    path = _write(tmp_path)
    cfg = build(str(path), autosave=True)
    try:
        assert cfg._autosave_registered is True
    finally:
        cfg.disable_autosave()
    assert cfg._autosave_registered is False


def test_enable_and_disable_autosave_are_idempotent(tmp_path):
    cfg = build(str(_write(tmp_path)))
    cfg.enable_autosave()
    cfg.enable_autosave()
    assert cfg._autosave_registered is True
    cfg.disable_autosave()
    cfg.disable_autosave()
    assert cfg._autosave_registered is False


def test_a_reader_cannot_corrupt_a_sidecar_by_mutating_resolution(tmp_path):
    """The exact snowGradient scenario: load a 1024 sidecar, override the
    resolution for a probe, exit. The file must be unchanged on disk."""
    path = _write(tmp_path, resolution=[1024, 1024])
    original = path.read_text()

    cfg = build(str(path))
    cfg.resolution = [64, 64]          # what resolution_override does
    # Simulate interpreter exit: run whatever this config registered.
    if cfg._autosave_registered:       # must be False
        cfg.save_config()

    assert json.loads(path.read_text())["resolution"] == [1024, 1024]
    assert path.read_text() == original


# --- the pools are load-bearing, not provenance decoration -------------------
# snowGradient's assert_sites_unseen reads all three pools and REFUSES to load a
# backbone whose sidecar records none, because unknown provenance must not read
# as "saw nothing". Dropping/renaming these keys breaks that check open.

@pytest.mark.parametrize("pool", ["trained_pool", "validation_pool", "test_pool"])
def test_all_three_pools_survive_a_round_trip(tmp_path, pool):
    groups = [["site0", 1, 1], ["site1", 1, 2]]
    cfg = build(str(_write(tmp_path, **{pool: groups})))
    assert cfg.dump()[pool] == groups


def test_pools_are_always_present_in_dump_even_when_unset(tmp_path):
    """Absent must be expressible as null, not as a missing key -- a consumer
    distinguishing "no pools recorded" from "empty pools" needs the key."""
    dumped = build(str(_write(tmp_path))).dump()
    for pool in ("trained_pool", "validation_pool", "test_pool"):
        assert pool in dumped


# --- provenance --------------------------------------------------------------
# Establishing which manifest the v0.1.0 backbones saw required inferring it
# from pool contents. A recorded revision makes it a lookup.

def test_dataset_revision_round_trips(tmp_path):
    cfg = build(str(_write(tmp_path, dataset_revision="abc123")))
    assert cfg.dataset_revision == "abc123"
    assert cfg.dump()["dataset_revision"] == "abc123"


def test_snowgan_version_is_stamped_by_the_writer(tmp_path):
    """Records who last WROTE the file, which is the question a schema
    mismatch asks -- so it is the running version, not the loaded one."""
    import snowgan

    cfg = build(str(_write(tmp_path, snowgan_version="0.0.1-ancient")))
    assert cfg.dump()["snowgan_version"] == snowgan.__version__


def test_version_is_exposed():
    import snowgan

    assert isinstance(snowgan.__version__, str) and snowgan.__version__


def test_legacy_sidecar_without_provenance_still_loads(tmp_path):
    data = copy.deepcopy(config_template)
    for key in ("dataset_revision", "snowgan_version", "critic_updates"):
        data.pop(key, None)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))

    cfg = build(str(path))
    assert cfg.dataset_revision is None
