"""Snapshot and preview retention (UPGRADES #66, #67).

The old policy was a no-op: `_cleanup_saved_batches(100)` was called while
snapshots only existed at multiples of 1000, so nothing was ever removed --
~70 GB per 100k-step run (34 GB of snapshots + 35 GB of previews). Its preview
trim was also inverted: `indexed_images[-7:]` deleted the 7 NEWEST images.

The replacement keeps exactly what the campaign's analysis reads, and is a pure
function so the policy can be tested without touching disk.
"""

from snowgan.trainer import Trainer

plan = Trainer._retention_plan

SNAPSHOTS = list(range(1000, 100001, 1000))  # a 100k run, every 1k


def test_old_policy_was_a_no_op():
    """Regression anchor: keep_every=100 against 1k-spaced snapshots deletes
    nothing. This is the bug, stated as arithmetic."""
    assert all(n % 100 == 0 for n in SNAPSHOTS)


def test_keeps_every_10k_and_the_newest_window():
    delete, _ = plan(SNAPSHOTS, [], keep_every=10000, keep_recent=10)
    kept = set(SNAPSHOTS) - delete

    assert {10000 * k for k in range(1, 11)} <= kept       # trajectory points
    assert set(range(91000, 100001, 1000)) <= kept          # KID window
    assert len(kept) == 10 + 9                              # 100000 is in both
    assert 55000 in delete and 99000 not in delete


def test_the_kid_window_survives_mid_run():
    """At any point mid-run, the newest 10 must still be there -- that is the
    window kid_check.py --sweep scores."""
    so_far = list(range(1000, 47001, 1000))
    delete, _ = plan(so_far, [], keep_every=10000, keep_recent=10)
    assert set(range(38000, 47001, 1000)).isdisjoint(delete)


def test_previews_keep_the_newest_not_the_oldest():
    """The old trim deleted the newest images. Recent previews must survive."""
    previews = list(range(50, 100001, 50))
    _, delete = plan(SNAPSHOTS, previews, keep_every=10000, keep_recent=10,
                     preview_keep_every=5000)

    assert set(range(91000, 100001, 50)).isdisjoint(delete)   # recent window kept
    assert 5000 not in delete and 50000 not in delete          # coarse cadence kept
    assert 50 in delete and 51050 in delete                    # the rest go


def test_disabling_both_rules_disables_pruning_rather_than_deleting_everything():
    delete, previews = plan(SNAPSHOTS, [100, 200], keep_every=0, keep_recent=0)
    assert delete == set() and previews == set()


def test_empty_run_is_a_no_op():
    assert plan([], [50, 100]) == (set(), set())


def test_trainer_does_not_prune_previews_by_default(tmp_path):
    """Previews are the training-history record make_movie renders; keep them."""
    import os
    from types import SimpleNamespace

    syn = tmp_path / "synthetic_images"
    syn.mkdir()
    for n in range(1000, 30001, 1000):
        (tmp_path / f"batch_{n}").mkdir()
        (syn / f"step_{n}_synthetic_1.png").write_bytes(b"")

    fake = SimpleNamespace(save_dir=str(tmp_path),
                           _extract_batch_number=Trainer._extract_batch_number,
                           _retention_plan=Trainer._retention_plan)
    Trainer._cleanup_saved_batches(fake, keep_every=10000, keep_recent=3)

    assert len(os.listdir(syn)) == 30                       # every preview kept
    assert not (tmp_path / "batch_5000").exists()           # snapshots still pruned
    assert (tmp_path / "batch_30000").exists()
