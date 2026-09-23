"""Focused tests: the append-only run log (plan 0.4, UPGRADES #16).

The `.txt` pair this replaces is rewritten wholesale from an in-memory list on
every save, so an unclean exit silently diverges it from the run. And because
`--max_rss_mb` restarts are routine, the log must survive a relaunch and must
let a consumer tell replayed rows from new ones: `global_step` is persisted
only every 50 steps (rewinds up to 49 on restart) and the `batch` counter is
recovered by globbing snapshot dirs (rewinds up to `cleanup_milestone`).
"""

import json

from snowgan.metrics import MetricsLog, read_last_launch


def test_writer_appends_rather_than_truncating(tmp_path):
    """A writer copied from log.save_history would open "w" and wipe the log
    on every restart. Two MetricsLog instances model two launches."""
    first = MetricsLog(str(tmp_path))
    first.write({"global_step": 1})
    first.write({"global_step": 2})

    second = MetricsLog(str(tmp_path))
    second.write({"global_step": 3})

    lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3, "second launch truncated the first launch's rows"


def test_records_carry_launch_identity_and_sequence(tmp_path):
    log = MetricsLog(str(tmp_path), launch_id="abc123")
    log.write({"global_step": 7})
    log.write({"global_step": 8})

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().strip().split("\n")]
    assert [r["launch_id"] for r in records] == ["abc123", "abc123"]
    assert [r["seq"] for r in records] == [1, 2]
    assert all("wall_time" in r for r in records)


def test_read_last_launch_ignores_replayed_rows(tmp_path):
    """The restart rewind means global_step repeats across launches. A gate
    that read the whole file would evaluate on duplicated data."""
    first = MetricsLog(str(tmp_path), launch_id="launch1")
    for step in (1, 2, 3):
        first.write({"global_step": step, "wasserstein": -10.0})

    # Restart: global_step rewinds, the same steps are replayed with new values.
    second = MetricsLog(str(tmp_path), launch_id="launch2")
    for step in (2, 3, 4):
        second.write({"global_step": step, "wasserstein": -500.0})

    rows = read_last_launch(str(tmp_path / "metrics.jsonl"))
    assert [r["global_step"] for r in rows] == [2, 3, 4]
    assert all(r["wasserstein"] == -500.0 for r in rows)


def test_read_last_launch_tolerates_a_torn_final_line(tmp_path):
    """Killed mid-write is the expected end state for an OOM-killed run, not
    an error condition."""
    log = MetricsLog(str(tmp_path), launch_id="L")
    log.write({"global_step": 1})
    with open(tmp_path / "metrics.jsonl", "a") as handle:
        handle.write('{"global_step": 2, "part')

    rows = read_last_launch(str(tmp_path / "metrics.jsonl"))
    assert [r["global_step"] for r in rows] == [1]


def test_read_last_launch_on_missing_file(tmp_path):
    assert read_last_launch(str(tmp_path / "nope.jsonl")) == []


def test_write_event_tags_the_record(tmp_path):
    log = MetricsLog(str(tmp_path))
    log.write_event("run_start", disc_steps=2, lambda_gp=10.0)

    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record["event"] == "run_start"
    assert record["disc_steps"] == 2
    assert record["lambda_gp"] == 10.0


def test_log_failure_does_not_raise(tmp_path, capsys):
    """Telemetry must never take down a healthy training run."""
    log = MetricsLog(str(tmp_path))
    log.path = str(tmp_path / "no_such_dir" / "metrics.jsonl")

    log.write({"global_step": 1})
    log.write({"global_step": 2})

    assert "metrics log unavailable" in capsys.readouterr().out
