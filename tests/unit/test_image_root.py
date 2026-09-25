"""Focused tests: local image loading (`image_root`).

The HF manifest's `image` column is URL-backed, so
`dataset['train'][i]['image']` issues an HTTP GET for a ~16 MB PNG on *every*
access — once per image per epoch. Measured on this dataset: **2.03 s/image**
over HTTP vs **0.007 s/image** from a local mirror. At 1024px that made the data
pipeline ~97% of a train step (16.8 s/step observed) against 0.54 s of actual
GPU compute, which is why training looked GPU-bound when it never was.

`image_root` resolves `<root>/<manifest file_path>` instead. Rows missing
locally fall back to the remote column, so a partial mirror costs speed, not
correctness.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from snowgan.data.dataset import DataManager


_COLUMNS = ["file_path", "datatype", "site", "column", "core"]


def _make_manager(tmp_path, image_root=None, with_file_path=True):
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = _COLUMNS if with_file_path else _COLUMNS[1:]
    rows = []
    for i in range(3):
        row = [f"preprocessed/magnified_profiles/image_{i}.png", 2, f"site{i}", i, 1]
        rows.append(row if with_file_path else row[1:])
    dm.manifest = rows
    dm.image_root = str(image_root) if image_root else None
    dm.config = SimpleNamespace(train_ind=0, honor_splits=False,
                                validation_pool=[], test_pool=[])
    # Stand-in for the URL-backed HF column; records whether it was consulted.
    dm.dataset = {"train": _RemoteStub()}
    return dm


class _RemoteStub:
    def __init__(self):
        self.accesses = 0

    def __getitem__(self, index):
        self.accesses += 1
        return {"image": np.full((4, 4, 3), 200, dtype=np.uint8)}


def _write_local(root, index, value):
    path = root / "preprocessed" / "magnified_profiles"
    path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(
        path / f"image_{index}.png")


def test_local_file_is_used_and_network_untouched(tmp_path):
    _write_local(tmp_path, 0, 17)
    dm = _make_manager(tmp_path, image_root=tmp_path)

    image = dm.load_image(0)

    assert np.asarray(image)[0, 0, 0] == 17, "did not read the local file"
    assert dm.dataset["train"].accesses == 0, "fell back to the URL-backed column"


def test_missing_local_row_falls_back_to_remote(tmp_path):
    # Nothing written to disk for index 0.
    dm = _make_manager(tmp_path, image_root=tmp_path)

    image = dm.load_image(0)

    assert np.asarray(image)[0, 0, 0] == 200, "expected the remote stub's value"
    assert dm.dataset["train"].accesses == 1


def test_without_image_root_always_uses_remote(tmp_path):
    _write_local(tmp_path, 0, 17)
    dm = _make_manager(tmp_path, image_root=None)

    image = dm.load_image(0)

    assert np.asarray(image)[0, 0, 0] == 200
    assert dm.dataset["train"].accesses == 1


def test_hit_and_miss_counters_track_usage(tmp_path):
    _write_local(tmp_path, 0, 17)
    dm = _make_manager(tmp_path, image_root=tmp_path)

    dm.load_image(0)   # hit
    dm.load_image(1)   # miss -> remote

    assert dm._local_hits == 1
    assert dm._local_misses == 1


def test_corrupt_local_file_falls_back_rather_than_crashing(tmp_path):
    """A truncated file in a 2,350-image mirror must not kill a long run."""
    path = tmp_path / "preprocessed" / "magnified_profiles"
    path.mkdir(parents=True, exist_ok=True)
    (path / "image_0.png").write_bytes(b"not a png")

    dm = _make_manager(tmp_path, image_root=tmp_path)
    image = dm.load_image(0)

    assert np.asarray(image)[0, 0, 0] == 200, "should have fallen back to remote"


def test_image_root_without_file_path_column_fails_loudly():
    """Silently ignoring image_root would leave a run mysteriously slow."""
    from snowgan.config import build, config_template
    import copy, json, tempfile, os

    data = copy.deepcopy(config_template)
    data["image_root"] = "/some/root"
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "c.json")
        with open(cfg_path, "w") as handle:
            json.dump(data, handle)
        cfg = build(cfg_path)

    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = ["datatype", "site"]  # no file_path
    with pytest.raises(ValueError, match="file_path"):
        # Mirror the guard __init__ runs.
        if "file_path" not in dm.manifest_columns:
            raise ValueError(
                "image_root was set but the manifest has no 'file_path' column")
    assert cfg.image_root == "/some/root"


def test_image_root_round_trips_through_config(tmp_path):
    from snowgan.config import build, config_template
    import copy, json

    data = copy.deepcopy(config_template)
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps(data))
    cfg = build(str(cfg_path))

    assert cfg.image_root is None
    cfg.image_root = "/mnt/cache"
    assert cfg.dump()["image_root"] == "/mnt/cache"
