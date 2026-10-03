"""make_movie must play the training history in step order.

The old sort key was the last underscore segment of the filename -- the sample
index, not the step -- so a single-seed movie tied on every frame and fell back
to string order: batch_10000 played before batch_2000.
"""

from snowgan.generate import preview_frames


def _touch(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_bytes(b"")


def test_frames_are_ordered_by_step_not_string(tmp_path):
    _touch(tmp_path, "batch_10000_synthetic_3.png", "batch_2000_synthetic_3.png",
           "batch_1000_synthetic_3.png", "batch_950_synthetic_3.png")
    steps = [step for step, _ in preview_frames(str(tmp_path), sample_index=3)]
    assert steps == [950, 1000, 2000, 10000]


def test_sample_index_selects_one_seed(tmp_path):
    _touch(tmp_path, "batch_100_synthetic_3.png", "batch_100_synthetic_4.png",
           "batch_200_synthetic_3.png", "batch_200_synthetic_10.png")
    frames = preview_frames(str(tmp_path), sample_index=3)
    assert [p.split("/")[-1].split("\\")[-1] for _, p in frames] == [
        "batch_100_synthetic_3.png", "batch_200_synthetic_3.png"]


def test_old_and_new_preview_names_interleave_by_step(tmp_path):
    """Runs straddling the batch_ -> step_ rename (UPGRADES #67) stay ordered."""
    _touch(tmp_path, "step_3000_synthetic_1.png", "batch_2000_synthetic_1.png")
    assert [s for s, _ in preview_frames(str(tmp_path), 1)] == [2000, 3000]


def test_non_preview_files_are_ignored(tmp_path):
    _touch(tmp_path, "history.png", "batch_100_synthetic_1.png")
    assert len(preview_frames(str(tmp_path), None)) == 1
