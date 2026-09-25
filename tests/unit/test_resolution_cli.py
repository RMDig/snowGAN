"""Focused test: --resolution "H W" parses to [H, W] (UPGRADES #5).

The flag was `type=set`, so `--resolution "256 256"` became the character set
{'2','5','6',' '} and the requested size was silently dropped — the config kept
its 1024 default. That guarantees a real/fake resolution mismatch on any low-res
run. This pins the space-separated int parse, mirroring --gen_kernel/--gen_filters.
"""

import copy
import json

from snowgan.config import build, config_template, configure_generic
from snowgan.utils import parse_args


def _config(tmp_path):
    data = copy.deepcopy(config_template)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))
    return build(str(cfg_path))


def test_resolution_parses_to_int_pair(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr("sys.argv", ["snowgan", "--mode", "train", "--resolution", "256 256"])
    configure_generic(cfg, parse_args())
    assert cfg.resolution == [256, 256]


def test_resolution_omitted_keeps_default(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr("sys.argv", ["snowgan", "--mode", "train"])
    configure_generic(cfg, parse_args())
    assert list(cfg.resolution) == [1024, 1024]
