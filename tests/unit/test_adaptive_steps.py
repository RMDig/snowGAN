"""Focused regression: the adaptive disc-step count must not ratchet across resumes.

`_update_adaptive_steps` used to write the evolved count back into
`disc.config.training_steps`, which `Config.dump()` persists. Every restart then
re-read that evolved value as `_base_disc_steps` and set `max_steps = base * 2`,
so the ceiling doubled on each resume rather than staying anchored to the launch
value. The --max_rss_mb leak workaround turned that into a feedback loop: more
disc steps meant more leak per batch, which meant more restarts, which ratcheted
faster (observed: disc_steps 1 -> 6 -> 10 -> 16 -> 37 across the core run).

These pin the split that makes it impossible: `config.training_steps` is the
launch parameter and is never written during training; the live count lives on
the trainer and dies with the process.
"""

import copy
import json
import types

from snowgan.config import build, config_template
from snowgan.trainer import Trainer

# Loss pair yielding disc_ratio = |d| / (|d| + |g|) = 1/11 ≈ 0.09, i.e. below the
# 0.3 "disc too weak" threshold, so every call tries to add a step. Both EMAs
# start at 0.0 and scale together, so the ratio is stable across calls.
_DISC_WEAK = dict(disc_loss=1.0, gen_loss=10.0)


def _config(tmp_path, training_steps=5):
    data = copy.deepcopy(config_template)
    data["architecture"] = "discriminator"
    data["training_steps"] = training_steps
    data["adaptive_steps"] = True
    cfg_path = tmp_path / "discriminator_config.json"
    cfg_path.write_text(json.dumps(data))
    return cfg_path, build(str(cfg_path))


def _trainer_for(cfg):
    """A Trainer with only the state `_update_adaptive_steps` reads.

    Bypasses __init__ (which builds models and pulls the HF dataset) to keep this
    sub-second. `disc.config` is a real Config so the persistence half of the
    invariant is exercised, not stubbed.
    """
    trainer = object.__new__(Trainer)
    trainer.adaptive_steps = True
    trainer.global_step = 100  # the % 100 == 0 adjust tick
    trainer._base_disc_steps = cfg.training_steps
    trainer._disc_steps = cfg.training_steps
    trainer._disc_loss_ema = 0.0
    trainer._gen_loss_ema = 0.0
    trainer.disc = types.SimpleNamespace(config=cfg)
    return trainer


def test_adaptive_evolution_leaves_launch_config_untouched(tmp_path):
    _, cfg = _config(tmp_path, training_steps=5)
    trainer = _trainer_for(cfg)

    for _ in range(50):
        Trainer._update_adaptive_steps(trainer, **_DISC_WEAK)

    # The live count evolves and saturates at 2x the launch value.
    assert trainer._disc_steps == 10
    # The launch parameter does not move, in memory or on the way to disk.
    assert cfg.training_steps == 5
    assert cfg.dump()["training_steps"] == 5


def test_ceiling_does_not_ratchet_across_resumes(tmp_path):
    """Three save/resume cycles must leave the ceiling where it started."""
    cfg_path, _ = _config(tmp_path, training_steps=5)

    for _ in range(3):
        cfg = build(str(cfg_path))  # resume from what the previous cycle persisted
        trainer = _trainer_for(cfg)

        # Every cycle anchors to the launch value, not to where the last one ended.
        assert trainer._base_disc_steps == 5

        for _ in range(50):
            Trainer._update_adaptive_steps(trainer, **_DISC_WEAK)

        assert trainer._disc_steps <= 10
        cfg_path.write_text(json.dumps(cfg.dump()))  # checkpoint, as train() does

    assert build(str(cfg_path)).training_steps == 5


def test_adaptive_steps_disabled_is_a_no_op(tmp_path):
    _, cfg = _config(tmp_path, training_steps=5)
    trainer = _trainer_for(cfg)
    trainer.adaptive_steps = False

    for _ in range(50):
        Trainer._update_adaptive_steps(trainer, **_DISC_WEAK)

    assert trainer._disc_steps == 5
    assert cfg.training_steps == 5
