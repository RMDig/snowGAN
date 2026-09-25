"""Append-only JSON Lines run log (UPGRADES #16, plan item 0.4).

Replaces the pair of ``*_loss.txt`` files as the *analysis* record. Those
files are full-file rewrites from an in-memory list on every save
(``log.save_history``), so an unclean exit silently diverges them from the
run, and they carry one conflated scalar per side — which is why no verdict in
``docs/experiments.md`` was ever reached with a number that meant what it was
read to mean.

Two properties this writer has that the ``.txt`` pair does not:

**Append mode.** ``log.save_history`` opens with ``"w"``. A writer copied from
it would wipe the log on every ``--max_rss_mb`` restart, and those restarts are
routine.

**Replay is detectable.** ``global_step`` is persisted only every 50 steps and
rewinds by up to 49 on restart; the ``batch`` counter is recovered by globbing
snapshot directories and rewinds by up to ``cleanup_milestone``. Both therefore
repeat across a restart. Every record carries a ``launch_id`` (unique per
process) and a per-launch monotonic ``seq``, so a consumer can take the last
launch's rows, or the last occurrence of each ``global_step``, instead of
reading replayed rows as new ones.
"""

import json
import os
import time
import uuid


class MetricsLog:
    """Append-only JSONL sink for per-step training metrics.

    Failures to write are swallowed with a one-time warning: the run log is an
    observation channel, and losing it must never take down a training run that
    is otherwise healthy. (This is a deliberate, documented exception to
    CLAUDE.md §2 rather than a bandaid — the recovery path is "training
    continues, log is incomplete", which is the correct behavior for telemetry.)
    """

    FILENAME = "metrics.jsonl"

    def __init__(self, save_dir, launch_id=None):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, self.FILENAME)
        self.launch_id = launch_id or uuid.uuid4().hex[:12]
        self.seq = 0
        self._warned = False

    def write(self, record):
        """Append one record, stamped with launch identity and sequence.

        Args:
            record (dict): JSON-serializable metrics for one train step.
        """
        self.seq += 1
        payload = {
            "launch_id": self.launch_id,
            "seq": self.seq,
            "wall_time": time.time(),
        }
        payload.update(record)
        try:
            with open(self.path, "a") as handle:
                handle.write(json.dumps(payload, default=float) + "\n")
        except OSError as error:
            if not self._warned:
                print(f"Warning: metrics log unavailable ({error}); training continues.")
                self._warned = True

    def write_event(self, event, **fields):
        """Append a non-per-step record (run start, gate evaluation, kill-check)."""
        self.write({"event": event, **fields})


def read_last_launch(path):
    """Read a ``metrics.jsonl`` and return only the most recent launch's rows.

    The helper the gate checks in ``docs/plans/training_dynamics_recovery_plan.md``
    §4 are defined against: reading the whole file would include steps replayed
    after a restart and evaluate a gate on duplicated data.

    Returns:
        list[dict]: records whose ``launch_id`` equals that of the final record,
        in file order. Empty list when the file is missing or empty.
    """
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line (killed mid-write) is expected, not an error.
                continue
    if not records:
        return []
    last_launch = records[-1].get("launch_id")
    return [r for r in records if r.get("launch_id") == last_launch]
