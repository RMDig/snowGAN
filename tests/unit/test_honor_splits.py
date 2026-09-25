"""Focused tests: the training stream honors the persisted splits (plan 0.5a).

`derive_splits` has always partitioned groups 80/10/10 and `Trainer` mirrors
the pools onto both configs specifically so AvAI can evaluate against
`test_pool` — but no batch path ever consulted them. `batch()` filtered on
`datatype` alone, so the GAN trained on its own held-out groups and every
downstream transfer probe against these backbones was contaminated. CLAUDE.md
§9 calls that probe "the real metric".

Tests bypass DataManager.__init__ (which calls load_dataset) via __new__ and a
synthetic manifest, matching test_splits.py / test_dataset.py.
"""

from types import SimpleNamespace

from snowgan.data.dataset import DataManager


_COLUMNS = ["datatype", "site", "column", "core"]


def _make_manager(honor_splits=True, held_out=(), n_groups=6):
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = _COLUMNS
    dm.manifest = []
    for i in range(n_groups):
        dm.manifest.append([0, f"site{i}", i, 1])   # core
        dm.manifest.append([2, f"site{i}", i, 1])   # magnified profile
    dm.config = SimpleNamespace(
        trained_pool=None,
        validation_pool=[list(k) for k in held_out[:1]],
        test_pool=[list(k) for k in held_out[1:]],
        seed=42,
        honor_splits=honor_splits,
        train_ind=0,
    )
    dm.seen_profiles = set()
    dm.seen_cores = set()
    return dm


def test_held_out_keys_come_from_validation_and_test_pools():
    dm = _make_manager(held_out=[("site0", 0, 1), ("site1", 1, 1)])
    assert dm.held_out_keys == {("site0", 0, 1), ("site1", 1, 1)}


def test_held_out_rows_are_skipped():
    dm = _make_manager(held_out=[("site0", 0, 1), ("site1", 1, 1)])

    held = dm._get_manifest_entry(0)          # site0 core -> held out
    allowed = dm._get_manifest_entry(4)       # site2 core -> trainable

    assert dm._is_held_out(held) is True
    assert dm._is_held_out(allowed) is False


def test_no_honor_splits_reproduces_legacy_leaky_behavior():
    """--no-honor_splits must restore the old stream exactly, so a pre-2026-09
    run can be reproduced when comparability matters more than validity."""
    dm = _make_manager(honor_splits=False, held_out=[("site0", 0, 1), ("site1", 1, 1)])
    assert dm._is_held_out(dm._get_manifest_entry(0)) is False


def test_empty_pools_train_on_everything():
    """First run, before derive_splits has populated the pools: nothing is
    held out, and the filter must not silently starve the stream."""
    dm = _make_manager(held_out=[])
    assert dm.held_out_keys == set()
    assert dm._is_held_out(dm._get_manifest_entry(0)) is False


def test_pools_round_trip_from_json_lists():
    """Pools persist as list[list]; membership must still match the manifest's
    tuple keys after a JSON round trip."""
    dm = _make_manager(held_out=[("site3", 3, 1)])
    assert dm.config.validation_pool == [["site3", 3, 1]]
    assert ("site3", 3, 1) in dm.held_out_keys


def test_pools_derived_after_construction_are_picked_up():
    """The ordering trap. Trainer builds the DataManager BEFORE deriving
    splits, so anything that reads held_out_keys early must not freeze an
    empty set — that would silently restore the leak with no error."""
    dm = _make_manager(held_out=[])
    assert dm.held_out_keys == set()          # early read, pools still empty

    dm.config.validation_pool = [["site0", 0, 1]]
    dm.config.test_pool = [["site1", 1, 1]]

    assert dm.held_out_keys == {("site0", 0, 1), ("site1", 1, 1)}
    assert dm._is_held_out(dm._get_manifest_entry(0)) is True


def test_missing_honor_splits_attribute_defaults_to_honoring():
    """A config object predating the field must not silently leak."""
    dm = _make_manager(held_out=[("site0", 0, 1)])
    del dm.config.honor_splits
    assert dm._is_held_out(dm._get_manifest_entry(0)) is True
