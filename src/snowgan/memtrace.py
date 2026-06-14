"""Production CPU-RAM leak attributor — native-heap focused.

DIAGNOSTIC ONLY (diag/memtrace-native branch). No-op unless SNOWGAN_MEMTRACE
is set. Three CPU-RAM leaks were already found and fixed (#21/#22/#23); a
residual ~+1 MiB/batch still ratchets RSS over long runs and OOM-kills training.
That residual is NATIVE (RSS climbs while Python tracemalloc stays flat), so the
old tracemalloc-only probe is blind to it by construction. This module adds the
fine-grained *native* instruments that can actually localize it.

Enable on the real run (the leak does not reproduce in a fresh bench process):

    SNOWGAN_MEMTRACE=100        # snapshot every 100 batches (interval, batches)
    SNOWGAN_MEMTRACE_FILE=path  # JSONL output (default <save_dir>/memtrace-<pid>.jsonl)
    SNOWGAN_MEMTRACE_TRIM=1      # also run malloc_trim(0) each tick + log reclaimed RSS
    SNOWGAN_MEMTRACE_PYTHON=1    # ALSO enable 30-frame tracemalloc snapshot-diff (heavy)

Each tick writes one JSONL record. Read it like this:

  * RSS climbs, mallinfo2.uordblks (in-use arena bytes) climbs in lockstep
        -> a genuine native leak living in glibc arenas. malloc_trim won't help
           (reclaimed≈0) because the bytes are *live*. Substep deltas say where.
  * RSS climbs but mallinfo2.hblkhd (mmap'd bytes) / maps_count climbs
        -> unbounded mmap growth (e.g. cuDNN/cuda JIT module cache per step).
           This is the signature of the deferred Blackwell/cuDNN-fallback theory.
  * RSS climbs, uordblks flat, fordblks (free-in-arena) climbs, reclaimed>0
        -> arena fragmentation only; not a true leak, malloc_trim recovers it.
  * RSS and tracemalloc climb together (PYTHON mode) -> Python leak; top_growers
        names the file:line. (Default native-only mode leaves these fields null.)

The native probes (mallinfo2 / smaps_rollup / maps count / leak-rate fit) are
near-zero overhead and safe to leave on for a multi-day run. tracemalloc and the
malloc_trim test are opt-in because they are heavier / perturb the arenas.
"""
import ctypes
import ctypes.util
import gc
import json
import os
import time
import tracemalloc
from collections import deque

# ---------------------------------------------------------------------------
# Native (libc) bindings. All guarded — degrade to None off Linux/glibc.
# ---------------------------------------------------------------------------

class _Mallinfo2(ctypes.Structure):
    # glibc 2.33+ struct mallinfo2: every field is size_t (64-bit), unlike the
    # legacy `mallinfo` whose `int` fields overflow past 2 GiB — and a leaking
    # TF process is well past 2 GiB, so mallinfo2 is required, not optional.
    _fields_ = [
        ("arena", ctypes.c_size_t),     # total bytes in non-mmapped arenas
        ("ordblks", ctypes.c_size_t),   # number of free chunks
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),     # number of mmapped regions
        ("hblkhd", ctypes.c_size_t),    # total bytes in mmapped regions
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),  # total in-use bytes (the leak shows here)
        ("fordblks", ctypes.c_size_t),  # total free bytes inside arenas (frag)
        ("keepcost", ctypes.c_size_t),  # releasable bytes at the top of the heap
    ]


def _load_libc():
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        if hasattr(libc, "mallinfo2"):
            libc.mallinfo2.restype = _Mallinfo2
        return libc
    except Exception:
        return None


_LIBC = _load_libc()
_MIB = 1024.0 * 1024.0


def _mallinfo2():
    """glibc arena accounting in MiB, or None when unavailable."""
    if _LIBC is None or not hasattr(_LIBC, "mallinfo2"):
        return None
    try:
        mi = _LIBC.mallinfo2()
        return {
            "arena_mib": mi.arena / _MIB,
            "hblkhd_mib": mi.hblkhd / _MIB,
            "uordblks_mib": mi.uordblks / _MIB,
            "fordblks_mib": mi.fordblks / _MIB,
            "keepcost_mib": mi.keepcost / _MIB,
        }
    except Exception:
        return None


def _malloc_trim():
    """Return free arena memory to the OS. True if the call ran."""
    if _LIBC is None or not hasattr(_LIBC, "malloc_trim"):
        return False
    try:
        _LIBC.malloc_trim(0)
        return True
    except Exception:
        return False


def _rss_mib():
    """Process RSS in MiB from /proc, or -1.0 off Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def _maps_count():
    """Number of VMAs in /proc/self/maps. Unbounded growth = mmap leak
    (cuDNN/cuda JIT module cache mmaps a fresh region per fallback compile)."""
    try:
        with open("/proc/self/maps") as f:
            return sum(1 for _ in f)
    except OSError:
        return -1


def _smaps_rollup():
    """Kernel-aggregated RSS split (anon vs file-backed) in MiB, or None.
    Anonymous growth localizes the leak to the heap (malloc), not file maps."""
    out = {}
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                for key, field in (("Rss:", "rss_mib"),
                                   ("Anonymous:", "anon_mib"),
                                   ("Private_Dirty:", "priv_dirty_mib")):
                    if line.startswith(key):
                        out[field] = int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return out or None


# ---------------------------------------------------------------------------
# Leak-rate fit: least-squares slope over a rolling window of samples.
# ---------------------------------------------------------------------------

def _linfit(xs, ys):
    """Slope of ys vs xs by ordinary least squares, plus R². Returns
    (slope, r2) or (None, None) when undetermined (<2 pts or no x-spread)."""
    n = len(xs)
    if n < 2:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy * sxy / (sxx * syy)) if syy > 0 else None
    return slope, r2


class MemTrace:
    """Per-tick native-heap snapshot writer. Created via maybe_create_memtrace."""

    def __init__(self, interval, save_dir=None, top=12):
        self.interval = max(1, int(interval))
        self.top = top
        self.trim = bool(os.environ.get("SNOWGAN_MEMTRACE_TRIM"))
        self.python = bool(os.environ.get("SNOWGAN_MEMTRACE_PYTHON"))
        self._prev_snap = None
        # Rolling window for the leak-rate fit (batch, elapsed_s, rss).
        self._window = deque(maxlen=64)
        self._t0 = time.monotonic()
        # Per-step substep RSS marks, filled by note() during train_step.
        self._marks = []

        path = os.environ.get("SNOWGAN_MEMTRACE_FILE")
        if not path:
            base = save_dir or os.getcwd()
            try:
                os.makedirs(base, exist_ok=True)
            except OSError:
                base = os.getcwd()
            path = os.path.join(base, f"memtrace-{os.getpid()}.jsonl")
        self.path = path
        self._fh = None
        try:
            self._fh = open(self.path, "a", buffering=1)  # line-buffered
        except OSError as e:
            print(f"MEMTRACE could not open {self.path}: {e}", flush=True)

        if self.python and not tracemalloc.is_tracing():
            tracemalloc.start(30)  # deep enough to attribute through wrappers

        print(
            f"MEMTRACE enabled: interval={self.interval} native=on "
            f"trim={'on' if self.trim else 'off'} "
            f"python_tracemalloc={'on' if self.python else 'off'} -> {self.path}",
            flush=True,
        )

    # -- per-step substep marks (called from the eager train_step) -----------

    def due(self, batch):
        """True when this batch should emit a snapshot. Drives both the
        within-train_step substep marks and the post-step tick from one
        cadence, so substep detail always lines up with the native snapshot."""
        return batch % self.interval == 0

    def note(self, label):
        """Record an RSS + uordblks waypoint. Cheap; only called on due batches
        (the train loop passes trace=True). uordblks (glibc live arena) is the
        RELIABLE per-phase counter — RSS lags the actual malloc, so RSS substep
        deltas misattribute. Consecutive deltas are emitted by the next tick()."""
        mi = _mallinfo2() or {}
        self._marks.append((label, _rss_mib(), mi.get("uordblks_mib")))

    def _drain_substeps(self):
        marks, self._marks = self._marks, []
        if len(marks) < 2:
            return None
        out = {}
        for (la, ra, ua), (_, rb, ub) in zip(marks, marks[1:]):
            out[f"{la}_mib"] = rb - ra
            if ua is not None and ub is not None:
                out[f"{la}_uord_mib"] = ub - ua  # reliable: live-arena delta for this phase
        out["total_mib"] = marks[-1][1] - marks[0][1]
        if marks[0][2] is not None and marks[-1][2] is not None:
            out["total_uord_mib"] = marks[-1][2] - marks[0][2]
        return out

    # -- periodic snapshot ---------------------------------------------------

    def tick(self, batch):
        try:
            rss = _rss_mib()
            self._window.append((float(batch), time.monotonic() - self._t0, rss))

            rec = {
                "ts": time.time(),
                "batch": int(batch),
                "rss_mib": round(rss, 2),
                "mallinfo2": _mallinfo2(),
                "maps_count": _maps_count(),
                "smaps": _smaps_rollup(),
                "gc_objects": len(gc.get_objects()),
                "substeps": self._drain_substeps(),
            }

            batches = [w[0] for w in self._window]
            secs = [w[1] for w in self._window]
            rsss = [w[2] for w in self._window]
            slope_b, r2 = _linfit(batches, rsss)
            slope_s, _ = _linfit(secs, rsss)
            rec["leak_rate"] = {
                "mib_per_batch": round(slope_b, 4) if slope_b is not None else None,
                "mib_per_hour": round(slope_s * 3600.0, 2) if slope_s is not None else None,
                "window": len(self._window),
                "r2": round(r2, 4) if r2 is not None else None,
            }

            if self.trim:
                before = _rss_mib()
                ran = _malloc_trim()
                after = _rss_mib()
                rec["trim"] = {
                    "ran": ran,
                    "rss_before_mib": round(before, 2),
                    "rss_after_mib": round(after, 2),
                    "reclaimed_mib": round(before - after, 2),
                }

            if self.python:
                cur, peak = tracemalloc.get_traced_memory()
                rec["tracemalloc_cur_mib"] = round(cur / 1e6, 2)
                rec["tracemalloc_peak_mib"] = round(peak / 1e6, 2)
                snap = tracemalloc.take_snapshot()
                if self._prev_snap is not None:
                    growers = []
                    for s in snap.compare_to(self._prev_snap, "lineno")[: self.top]:
                        frame = s.traceback[0]
                        growers.append({
                            "loc": f"{frame.filename}:{frame.lineno}",
                            "size_diff_mib": round(s.size_diff / 1e6, 3),
                            "size_mib": round(s.size / 1e6, 3),
                            "count_diff": s.count_diff,
                        })
                    rec["top_growers"] = growers
                self._prev_snap = snap

            if self._fh is not None:
                self._fh.write(json.dumps(rec) + "\n")

            self._print_line(rec)
        except Exception as e:
            print(f"MEMTRACE error (continuing): {e}", flush=True)

    def _print_line(self, rec):
        mi = rec.get("mallinfo2") or {}
        lr = rec.get("leak_rate") or {}
        ss = rec.get("substeps") or {}
        parts = [
            f"MEMTRACE batch={rec['batch']}",
            f"rss={rec['rss_mib']:.0f}MiB",
            f"uord={mi.get('uordblks_mib', float('nan')):.0f}",
            f"ford={mi.get('fordblks_mib', float('nan')):.0f}",
            f"mmap={mi.get('hblkhd_mib', float('nan')):.0f}",
            f"maps={rec['maps_count']}",
            f"rate={lr.get('mib_per_batch')}MiB/b",
        ]
        if ss:
            parts.append(
                f"[disc={ss.get('disc_loop_mib', 0):+.1f} "
                f"gen={ss.get('gen_loop_mib', 0):+.1f} "
                f"post={ss.get('poststep_mib', 0):+.1f}]"
            )
        if "trim" in rec:
            parts.append(f"reclaimed={rec['trim']['reclaimed_mib']:+.1f}MiB")
        print(" ".join(parts), flush=True)

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def maybe_create_memtrace(save_dir=None):
    """Build a MemTrace if SNOWGAN_MEMTRACE is set, else None (zero overhead)."""
    raw = os.environ.get("SNOWGAN_MEMTRACE")
    if not raw:
        return None
    try:
        interval = int(raw)
    except ValueError:
        interval = 100
    return MemTrace(interval, save_dir=save_dir)
