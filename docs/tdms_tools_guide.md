# tdms\_tools\_guide.md

This guide is written for newcomers first. We start with a minimal working example, then explain typical usage and design choices, add practical multi‑channel patterns, and finish with the technical dataclass details.

---

## Hello, chunks (minimal example)

```python
from tdms_tools import TDMSDataset

# Point to a folder containing one or more .tdms files
# (Windows users: r"C:\\data\\run01")
ds = TDMSDataset("./data/run01")

# Build the index (collects per-file timing metadata)
ds.build_index(channels=["wfg"])  # optional filter speeds it up

# Stream fixed-size windows ("chunks") across files
for chunk in ds.iter_chunks(
    channels=["wfg"],          # one or more channel names
    chunk_samples=1_000_000,    # samples per window
    absolute_time=False,        # -> chunk.t is float seconds since dataset start
):
    t = chunk.t                 # time axis (len N)
    y = chunk.data["wfg"]      # samples (len N)
    # process(t, y)
```

**What you get per iteration:**

* `chunk.t`: a NumPy array of length `chunk_samples`
* `chunk.data`: a dict mapping channel → NumPy array of samples
* `chunk.gap_before`: seconds between this chunk and the previous emitted chunk
* `chunk.source`: `(file_path, start_index_in_file, end_index_in_file)` for traceability

> Tip: Set `absolute_time=True` to get `datetime64[ns]` timestamps in `chunk.t`. That’s handy when correlating with external logs.

---

## Typical usage & design choices (the “why”)

* **Streaming‑first**: `iter_chunks` lets you process arbitrarily large recordings in constant memory by yielding fixed windows.
* **Relative vs absolute time**: You pick:

  * `absolute_time=False` → `chunk.t` is **seconds since dataset start** (easy math)
  * `absolute_time=True`  → `chunk.t` is **`datetime64[ns]`** (easy correlation)
* **Indexing before work**: `build_index(...)` computes per‑file timing (`dt`, `t0_abs`, lengths). It also checks intra‑file consistency when `strict=True` (default).
* **Primary channel for timing**: when chunking multiple channels, timing is anchored to the first listed channel.
* **Bounded windows**: Use `chunk_samples` (or `chunk_sec`) and `hop` (step size) for overlap/no‑overlap pipelines.
* **Edge handling**:

  * `drop_incomplete=True` discards a short tail at the end
  * `drop_incomplete=False` with `tail_fill` = `"nan"` or `"zero"` pads the last chunk
* **Gaps**: `gap_before` is reported so you can detect discontinuities; the `insert_gaps` flag is accepted for API parity but is a **no‑op** in this implementation.
* **t0/t1** limits: `t0` and `t1` are **relative seconds** from dataset start; chunking respects this span across files.

---

## Practical recipes

### A) Two channels, no overlap

```python
channels = ["wfg", "ch2_bQPDx--"]
for c in ds.iter_chunks(channels=channels, chunk_samples=16384, absolute_time=False):
    wfg = c.data["wfg"]
    bqx = c.data["ch2_bQPDx--"]
    # joint analysis here
```

### B) Overlap for spectral analysis (75% overlap)

```python
N = 16384
H = 4096   # hop < N → overlap
for c in ds.iter_chunks(channels=["wfg"], chunk_samples=N, hop=H, absolute_time=False):
    x = c.data["wfg"]
    # window → FFT → average, etc.
```

### C) Detect gaps/drift on the fly

```python
prev_end = None
for c in ds.iter_chunks(channels=["wfg"], chunk_samples=8192, absolute_time=False):
    if prev_end is not None:
        dt = float(np.median(np.diff(c.t))) if c.t.size > 1 else np.nan
        if (c.t[0] - prev_end) > 0.25 * dt:
            print(f"gap near t={c.t[0]:.6f}s: Δ={(c.t[0] - prev_end):+.3e}s")
    prev_end = c.t[-1]
```

### D) PSD via Welch (streamed)

```python
import numpy as np
from scipy.signal import welch

fs = None
acc = None
n = 0

for c in ds.iter_chunks(channels=["ch2_bQPDx--"], chunk_samples=2**15, absolute_time=False):
    x = c.data["ch2_bQPDx--"]
    if fs is None and c.t.size > 1:
        fs = 1.0 / np.median(np.diff(c.t))
    f, Pxx = welch(x, fs=fs, window="hann", nperseg=4096, noverlap=2048)
    acc = Pxx if acc is None else (acc + Pxx)
    n += 1

Pxx_avg = acc / max(n, 1)
```

### E) Materialize a window by absolute time

```python
import numpy as np

start = np.datetime64("2025-08-14T12:00:00", "ns")
stop  = np.datetime64("2025-08-14T12:00:05", "ns")
t_abs, y = ds.read_window_abs("wfg", start, stop)
```

### F) Materialize a window by relative seconds

```python
t_rel_s, y, t_start_abs = ds.read_window_rel("wfg", 0.0, 10.0)
```

### G) Streaming integral

```python
area, n_used = ds.integrate("wfg", absolute_time=False)
```

### H) Export to Zarr (disk‑backed)

```python
out = ds.to_zarr("out.zarr", channels=["wfg"], absolute_time=False, overwrite=True)
```

### I) Inspect the build index (needs pandas)

```python
df = ds.index_df()   # columns: file, group, channel, n, dt, sr, t0_abs, t_end_abs
```

### J) Estimate sample-interval drift from metadata

```python
summary, residuals = ds.estimate_dt_drift("wfg", dt_nominal=1/200_000)
print(summary)
# keys: dt_hat, sr_hat, bias_ppm, drift_ms_per_hour, ...
```

---

## Technical backend — dataclasses

### `ChannelMeta`

Represents per‑file, per‑channel facts discovered during indexing.

```python
@dataclass(frozen=True)
class ChannelMeta:
    file: Path                 # file path
    group: str                 # TDMS group
    channel: str               # channel name
    n: int                     # number of samples in this file
    dt: float                  # seconds per sample (wf_increment)
    sr: float                  # sample rate (Hz)
    t0_abs: np.datetime64      # absolute start timestamp

    @property
    def t_end_abs(self) -> np.datetime64: ...
```

**Notes:**

* `dt` and `t0_abs` are read via `_read_dt_t0(...)` from channel/group/file properties (no full `time_track()` materialization).
* Files are sorted by `(t0_abs, path, channel)` to form a consistent timeline.

### `Chunk`

A streamed window emitted by `iter_chunks`.

```python
@dataclass
class Chunk:
    t: np.ndarray                          # time axis (datetime64[ns] if absolute; else float seconds)
    data: Dict[str, np.ndarray]            # channel → samples (len == len(t))
    absolute: bool                         # whether t is absolute
    gap_before: float                      # seconds since previous emitted chunk
    source: Tuple[Path, int, int]          # (file_path, start_idx, end_idx) for primary channel
```

**Key behaviors tied to the implementation:**

* When `absolute_time=True`, `t` is built from the dataset’s earliest `t0_abs` and `dt` using an integer‑nanosecond clock (no float drift accumulation).
* When `absolute_time=False`, `t` is float seconds; the start is `start_sample * dt`.
* `gap_before` is computed from `dt` and chunk starts; useful for gap detection.

### Helper: diagnose an unknown `Chunk`

```python
def diagnose_chunk(c):
    """Print a concise report: time dtype/length, dt/fs estimate, per-channel stats, and source indices."""
    import numpy as np
    def secs(t):
        if np.issubdtype(t.dtype, np.datetime64):
            return (t.astype("datetime64[ns]") - t[0]).astype("timedelta64[ns]").astype(np.int64) / 1e9
        return t.astype(float)
    print(f"type={type(c).__name__}, absolute={getattr(c,'absolute',None)}, gap_before={getattr(c,'gap_before',None)}")
    t = np.asarray(c.t)
    print(f"t: dtype={t.dtype}, len={t.size}")
    if t.size > 1:
        dts = np.diff(secs(t))
        dt = float(np.median(dts))
        print(f"dt≈{dt:.9e}s, fs≈{(1.0/dt) if dt>0 else float('nan'):.3f}Hz")
    for name, arr in c.data.items():
        x = np.asarray(arr)
        nan = int(np.isnan(x).sum()) if np.issubdtype(x.dtype, np.floating) else 0
        print(f"{name:>16}: n={x.size}, dtype={x.dtype}, nan={nan}")
    print("source:", getattr(c, "source", None))
```

---

## Reference — main APIs (thin)

* `TDMSDataset(folder, group=None, strict=True)`

  * `build_index(channels=None) -> list[ChannelMeta]`
  * `channels() -> list[str]`
  * `index_df()` *(needs pandas)*
  * `iter_chunks(channels, *, chunk_samples=None, chunk_sec=None, hop=None, t0=None, t1=None, absolute_time=True, drop_incomplete=True, tail_fill="skip", skip_nan_chunks=False, insert_gaps=False, dtype=None)`
  * `read_all(channel, absolute_time=False, dtype=None)`
  * `integrate(channel, t_start=None, t_stop=None, absolute_time=True)`
  * `read_window_abs(channel, t0_abs, t1_abs, *, chunk_samples=1_000_000, dtype=None, return_relative=False, rel_origin=None)`
  * `read_window_rel(channel, t0_rel_s, t1_rel_s, *, chunk_samples=1_000_000, dtype=None)`
  * `to_zarr(out_path, channels=None, chunk_samples=1_000_000, absolute_time=False, insert_gaps=False, dtype=None, overwrite=False)`
  * `estimate_dt_drift(channel, dt_nominal=None)`
