"""Regression: rmdig `datatype` int->string schema change (2026-08-17).

rmdig re-uploaded `rocky_mountain_snowpack` with `datatype` as strings
('core'/'profile'/'magnified_profile') instead of ints (0/1/2). The old
`pair_index` filter `if datatype != 0 and datatype != 2` matched nothing against
strings, so it silently returned `{}` and every training batch came up empty.

The fix normalizes datatype to canonical ints at the load boundary
(`normalize_datatype`), so all consumers see one representation and unknown
values raise instead of being skipped. These tests exercise that path with a
synthetic string manifest — no dataset download, no images.
"""

import numpy as np
import pandas as pd
import pytest

from snowgan.data.dataset import DataManager, normalize_datatype, DATATYPE_TO_INT


# --- the boundary normalizer ------------------------------------------------

def test_normalize_maps_rmdig_strings():
    assert normalize_datatype("core") == 0
    assert normalize_datatype("profile") == 1
    assert normalize_datatype("magnified_profile") == 2
    assert normalize_datatype("crystal_card") == 3


def test_normalize_is_idempotent_on_ints():
    # A caller may pass an already-normalized int (e.g. 2), or rmdig may revert
    # to ints one day — both must pass through unchanged.
    for v in DATATYPE_TO_INT.values():
        assert normalize_datatype(v) == v
        assert normalize_datatype(np.int64(v)) == v


def test_normalize_raises_not_skips_on_unknown():
    # An unknown datatype is a schema error — loud, never silent.
    with pytest.raises(ValueError):
        normalize_datatype("loose_snow")
    with pytest.raises(ValueError):
        normalize_datatype(99)
    with pytest.raises(ValueError):  # bool is an int subclass; must not slip through
        normalize_datatype(True)


# --- pair_index over a normalized string manifest ---------------------------

def _manifest(rows):
    """Build (manifest, columns) the way DataManager.__init__ does: a DataFrame
    of rmdig-style rows with the datatype column normalized to ints."""
    df = pd.DataFrame(rows, columns=["datatype", "site", "column", "core"])
    df["datatype"] = df["datatype"].map(normalize_datatype)
    return df.values.tolist(), df.columns.tolist()


def _shell(manifest, columns):
    """A DataManager with only what pair_index reads, bypassing __init__ (which
    would call load_dataset)."""
    dm = object.__new__(DataManager)
    dm.manifest = manifest
    dm.manifest_columns = columns
    return dm


# rmdig-style STRING rows: two complete groups, plus decoys that must not pair.
_STRING_ROWS = [
    ("core", 0, 1, 1),               # 0  group (0,1,1)
    ("magnified_profile", 0, 1, 1),  # 1  group (0,1,1)
    ("profile", 0, 1, 1),            # 2  ignored (datatype 1)
    ("core", 1, 1, 1),               # 3  group (1,1,1)
    ("magnified_profile", 1, 1, 1),  # 4  group (1,1,1)
    ("magnified_profile", 1, 1, 1),  # 5  group (1,1,1) — 2nd profile
    ("core", 2, 1, 1),               # 6  lone core, no profile -> excluded
    ("magnified_profile", 3, 1, 1),  # 7  lone profile, no core -> excluded
    ("crystal_card", 0, 1, 1),       # 8  normalizes to 3, excluded from pairing
]


def test_pair_index_groups_string_datatype_manifest():
    dm = _shell(*_manifest(_STRING_ROWS))
    idx = dm.pair_index

    assert idx, "pair_index empty on string-datatype manifest (the regression)"
    assert set(idx) == {(0, 1, 1), (1, 1, 1)}, "wrong groups or decoys leaked in"
    assert idx[(0, 1, 1)] == [(0, 1)]                 # 1 core x 1 profile
    assert sorted(idx[(1, 1, 1)]) == [(3, 4), (3, 5)]  # 1 core x 2 profiles


def test_pair_index_empty_on_raw_unnormalized_strings():
    """Documents the bug: without boundary normalization the int filter matches
    nothing and pair_index collapses to {}. This is what the fix prevents."""
    dm = _shell(
        [["core", 0, 1, 1], ["magnified_profile", 0, 1, 1]],
        ["datatype", "site", "column", "core"],
    )
    assert dm.pair_index == {}


def test_pair_index_still_works_with_int_datatypes():
    """The pre-schema-change contract must survive: int manifests still group."""
    int_rows = [(DATATYPE_TO_INT[d], s, c, k) for (d, s, c, k) in _STRING_ROWS]
    df = pd.DataFrame(int_rows, columns=["datatype", "site", "column", "core"])
    dm = _shell(df.values.tolist(), df.columns.tolist())
    idx = dm.pair_index
    assert set(idx) == {(0, 1, 1), (1, 1, 1)}
    assert sorted(idx[(1, 1, 1)]) == [(3, 4), (3, 5)]


def test_manifest_normalization_raises_on_unknown_datatype():
    """The load-boundary .map surfaces an unknown datatype loudly."""
    df = pd.DataFrame([["glacier", 0, 1, 1]], columns=["datatype", "site", "column", "core"])
    with pytest.raises(ValueError):
        df["datatype"].map(normalize_datatype)
