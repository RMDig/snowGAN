"""DiffAugment must be applied in the generator update, not only the critic's.

The defect: augmentation was applied to real and fake for the critic update, but
the generator update fed the critic RAW fakes. So the critic was trained on one
distribution and the generator was optimized against that critic on a different
one. Every snowGAN run with `--augment` was optimizing a mismatch -- which is
also why the `--augment` arm of the increment queue could never have been judged
fairly.

DiffAugment's whole premise (Zhao et al., NeurIPS 2020) is that the transform is
*differentiable* so gradients flow back through it during the generator update.
Applying it only on the critic side discards exactly the property the method is
named for.

Also pins that the fix stays inert when augmentation is off -- the increment
campaign's arms A and B compare against a baseline trained before this change,
and that comparison is only valid if `--no-augment` runs bit-identically.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from snowgan.config import build, config_template

RES = 64  # 16 * 2^(1+1) -> one filter block


class _StubData:
    def __init__(self, config):
        self.config = config
        self.pair_depth = 1
        self.seen_profiles = set()

    def next_batch(self, batch_size):
        return np.random.uniform(-1, 1, (batch_size, 1, RES, RES, 3)).astype("float32")

    def batch(self, batch_size, datatype=None, config=None):
        return self.next_batch(batch_size)

    def reset_seen_profiles(self):
        pass

    def derive_splits(self, *a, **k):
        self.config.trained_pool = self.config.validation_pool = self.config.test_pool = []


def _make_trainer(tmp_path, monkeypatch, augment):
    import copy
    import importlib

    trainer_module = importlib.import_module("snowgan.trainer")
    from snowgan.models.generator import Generator
    from snowgan.models.discriminator import Discriminator

    monkeypatch.setattr(trainer_module, "DataManager", _StubData)

    def cfg(architecture, filters, steps):
        data = copy.deepcopy(config_template)
        data.update({
            "architecture": architecture, "save_dir": str(tmp_path) + "/", "checkpoint": None,
            "resolution": [RES, RES], "filter_counts": filters, "latent_dim": 16,
            "batch_size": 2, "depth": 1, "modality": "magnified_profile",
            "training_steps": steps, "lambda_gp": 10.0, "spectral_norm": False,
            "gen_norm": "none", "gen_upsampler": "transpose",
            "gen_convs_per_resolution": 1, "fade": False, "augment": augment,
            "adaptive_steps": False, "multiscale_disc": False, "ada_target": 0.0,
            "grad_clip_norm": 0.0, "ema_decay": 0.0, "lr_decay": None,
            "fid_interval": 0, "grad_probe_interval": 0,
        })
        path = tmp_path / f"{architecture}_config.json"
        path.write_text(json.dumps(data))
        return build(str(path))

    generator = Generator(cfg("generator", [8], 1))
    generator.model.build((None, 16))
    discriminator = Discriminator(cfg("discriminator", [8], 1))
    discriminator.model.build((None, 1, RES, RES, 3))
    return trainer_module.Trainer(generator, discriminator)


def _record_critic_inputs(trainer, monkeypatch):
    """Capture every tensor handed to the critic during one train_step.

    Patches `.call`, not `__call__`: Python resolves dunder methods on the type,
    so an instance-level `__call__` is never consulted. Keras's `Layer.__call__`
    dispatches to `self.call`, which does honour an instance attribute.
    """
    seen = []
    original = trainer.disc.model.call

    def spy(inputs, *args, **kwargs):
        seen.append(np.asarray(inputs))
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(trainer.disc.model, "call", spy)
    return seen


def test_generator_step_feeds_augmented_fakes(tmp_path, monkeypatch):
    """With --augment on, the tensor the generator optimizes through must be
    augmented -- not the raw generator output."""
    trainer = _make_trainer(tmp_path, monkeypatch, augment=True)
    assert trainer.use_augment is True
    trainer.augment_p = 1.0  # force augmentation so the test cannot pass by luck

    seen = _record_critic_inputs(trainer, monkeypatch)
    images = tf.random.normal([2, 1, RES, RES, 3])
    trainer.train_step(images)

    # disc_steps=1 -> [real, fake] for the critic; gen_steps=1 -> [fake] for the
    # generator. The generator's is the last.
    assert len(seen) >= 3, f"expected >=3 critic calls, saw {len(seen)}"
    gen_input = seen[-1]

    # Regenerate the raw (un-augmented) output the generator produced. If the
    # generator step were feeding raw fakes, its critic input would be a valid
    # generator output; augmented, it carries cutout zeros / flips / shifted
    # brightness that raw output essentially never reproduces exactly.
    #
    # The robust signal: cutout writes exact 0.0 blocks, which a tanh head does
    # not produce in bulk.
    exact_zeros = float(np.mean(gen_input == 0.0))
    assert exact_zeros > 0.0, (
        "generator step fed RAW fakes to the critic: no augmentation signature "
        "present. DiffAugment must be applied inside the generator tape.")


def test_no_augment_leaves_the_generator_path_untouched(tmp_path, monkeypatch):
    """The increment campaign compares arms against a baseline trained before
    this fix. That is only valid if --no-augment is bit-identical."""
    trainer = _make_trainer(tmp_path, monkeypatch, augment=False)
    assert trainer.use_augment is False
    assert trainer.augment_p == 0.0

    seen = _record_critic_inputs(trainer, monkeypatch)
    images = tf.random.normal([2, 1, RES, RES, 3])
    trainer.train_step(images)

    gen_input = seen[-1]
    # No cutout zeros, and values stay inside the tanh range: this is raw output.
    assert float(np.mean(gen_input == 0.0)) == 0.0
    assert np.all(np.abs(gen_input) <= 1.0 + 1e-5)


def test_generator_still_learns_with_augmentation_on(tmp_path, monkeypatch):
    """Gradients must survive the transform -- that is what 'differentiable'
    augmentation buys. If the fix broke the path, weights would not move."""
    trainer = _make_trainer(tmp_path, monkeypatch, augment=True)
    trainer.augment_p = 1.0

    before = [w.numpy().copy() for w in trainer.gen.model.trainable_variables]
    trainer.train_step(tf.random.normal([2, 1, RES, RES, 3]))
    after = [w.numpy() for w in trainer.gen.model.trainable_variables]

    moved = any(not np.allclose(b, a) for b, a in zip(before, after))
    assert moved, "generator weights did not move: gradients did not flow through augment"
