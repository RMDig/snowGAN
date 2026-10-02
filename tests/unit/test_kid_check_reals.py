"""Real-set selection for scripts/kid_check.py.

A KID is only interpretable if (a) the reals were never trained on and (b) it is
read against a real-vs-real floor at the same sample size. The v0.1.0 releases
trained on their own validation/test pools (UPGRADES #54), so their only unseen
reals are sites added after release; these tests pin the guard that refuses a
site the sidecar shows was seen, and the group-disjoint split behind the floor.
"""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def kid_check():
    path = Path(__file__).resolve().parents[2] / "scripts" / "kid_check.py"
    spec = importlib.util.spec_from_file_location("kid_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release_config():
    # The released pools: sites 0-2 only (13 groups).
    return types.SimpleNamespace(
        trained_pool=[[2, 1, 3], [1, 1, 2]], validation_pool=[[0, 1, 1]], test_pool=[[1, 1, 1], [2, 1, 6]])


def test_unseen_sites_pass(kid_check):
    kid_check._check_sites_unseen(_release_config(), {3, 4, 5, 6})


@pytest.mark.parametrize("sites", [{2}, {0, 3}])
def test_seen_sites_are_refused(kid_check, sites):
    with pytest.raises(SystemExit, match="not unseen"):
        kid_check._check_sites_unseen(_release_config(), sites)


def test_no_pools_means_no_site_is_known_unseen(kid_check):
    cfg = types.SimpleNamespace(trained_pool=None, validation_pool=None, test_pool=None)
    with pytest.raises(SystemExit, match="no split pools"):
        kid_check._check_sites_unseen(cfg, {3})


def test_select_rows_by_site_and_datatype(kid_check):
    frame = pd.DataFrame({
        "datatype": ["magnified_profile", "core", "magnified_profile", "magnified_profile"],
        "site": [3, 3, 1, 5], "column": [1, 1, 1, 2], "core": [1, 1, 1, 4]})
    rows = kid_check._select_rows(frame, 2, sites={3, 5})
    assert rows == [(0, (3, 1, 1)), (3, (5, 2, 4))]


def test_halves_are_group_disjoint_and_cover_everything(kid_check):
    keys = [(3, 1, c) for c in range(5) for _ in range(4)]  # 5 groups x 4 images
    a, b = kid_check._group_halves(keys, seed=0)
    groups_a = {k for k, m in zip(keys, a) if m}
    groups_b = {k for k, m in zip(keys, b) if m}
    assert groups_a and groups_b and not (groups_a & groups_b)
    assert np.all(a ^ b)


def test_halves_need_two_groups(kid_check):
    with pytest.raises(SystemExit):
        kid_check._group_halves([(3, 1, 1)] * 4, seed=0)


# --- memorization_check corpus ---------------------------------------------
# build() defaults a missing honor_splits to True, but a sidecar without the key
# predates the fix and trained on its validation/test pools (UPGRADES #54). The
# corpus must come from the raw sidecar, or the releases' memorization is
# understated twice: pools the GAN saw are dropped, and sites it never saw
# (added later) are compared against.

@pytest.fixture(scope="module")
def memcheck():
    path = Path(__file__).resolve().parents[2] / "scripts" / "memorization_check.py"
    spec = importlib.util.spec_from_file_location("memorization_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_POOLS = {"trained_pool": [[2, 1, 3], [1, 1, 2]], "validation_pool": [[0, 1, 1]],
          "test_pool": [[1, 1, 1], [2, 1, 6]]}


def test_legacy_sidecar_corpus_includes_its_pools(memcheck):
    groups = memcheck._training_groups(dict(_POOLS))  # no honor_splits key
    assert groups == {(2, 1, 3), (1, 1, 2), (0, 1, 1), (1, 1, 1), (2, 1, 6)}


def test_honoured_split_corpus_is_the_trained_pool(memcheck):
    groups = memcheck._training_groups(dict(_POOLS, honor_splits=True))
    assert groups == {(2, 1, 3), (1, 1, 2)}


def test_no_trained_pool_falls_back(memcheck):
    assert memcheck._training_groups({"trained_pool": None}) is None
