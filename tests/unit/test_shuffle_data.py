"""shuffle_data: the training stream's order (capacity_plan, 2026-10-19).

The historical stream reads the manifest in (site, column, core) order. For the
seed-42 training pool that is 36 contiguous runs, so 94% of 4-image batches are
a single core and every pass sweeps the cores in a fixed cycle. shuffle_data
serves a fresh permutation per pass. Off must reproduce the old stream exactly.
"""

from types import SimpleNamespace

import numpy as np

from snowgan.data.dataset import DataManager

N_GROUPS, PER_GROUP = 6, 8  # 48 images, grouped contiguously like the manifest


def _manager(shuffle, train_ind=0, data_pass=0, seed=42):
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = ["datatype", "site", "column", "core"]
    dm.manifest = [[2, g, 1, 1] for g in range(N_GROUPS) for _ in range(PER_GROUP)]
    dm.config = SimpleNamespace(shuffle_data=shuffle, train_ind=train_ind, data_pass=data_pass,
                                seed=seed, honor_splits=False, exclude_sites=[])
    dm.load_image = lambda index, meta=None: np.full((2, 2, 3), index, np.float32)
    dm.preprocess_image = lambda image: image
    return dm


def _serve(dm, batches, size=4):
    """Rows served over `batches` fetches, wrapping like Trainer.train."""
    out = []
    for _ in range(batches):
        x = dm.batch(size)
        if x is None or len(x) < size:
            if x is not None:
                out.append([int(v[0, 0, 0, 0]) for v in x])
            dm.config.train_ind = 0
            continue
        out.append([int(v[0, 0, 0, 0]) for v in x])
    return out


def _group(row):
    return row // PER_GROUP


def test_off_is_the_historical_manifest_order():
    served = sum(_serve(_manager(False), 12), [])
    assert served == list(range(48))


def test_off_batches_are_single_group_like_the_real_stream():
    batches = _serve(_manager(False), 12)
    assert all(len({_group(r) for r in b}) == 1 for b in batches)


def test_on_covers_every_image_once_per_pass_and_mixes_groups():
    batches = _serve(_manager(True), 12)
    served = sum(batches, [])
    assert sorted(served) == list(range(48))
    single = sum(len({_group(r) for r in b}) == 1 for b in batches)
    assert single <= 2, f"{single}/12 batches still single-group"


def test_on_draws_a_fresh_permutation_each_pass():
    dm = _manager(True)
    first = sum(_serve(dm, 12), [])
    assert dm.config.data_pass == 0
    dm.config.train_ind = 0  # the caller wraps after the end: a new pass begins
    second = sum(_serve(dm, 12), [])
    assert dm.config.data_pass == 1
    assert sorted(second) == list(range(48)) and second != first


def test_resume_reproduces_the_stream():
    """train_ind and data_pass are persisted; a new process continues identically."""
    live = _manager(True)
    _serve(live, 5)
    ind, data_pass = live.config.train_ind, live.config.data_pass
    rest_live = _serve(live, 4)
    resumed = _manager(True, train_ind=ind, data_pass=data_pass)
    assert _serve(resumed, 4) == rest_live


def test_permutation_depends_on_the_seed():
    a = sum(_serve(_manager(True, seed=42), 12), [])
    b = sum(_serve(_manager(True, seed=43), 12), [])
    assert a != b
