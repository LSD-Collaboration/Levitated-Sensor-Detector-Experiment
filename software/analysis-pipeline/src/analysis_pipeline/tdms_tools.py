"""
tdms_tools.py
--------------
A readable, battery-included TDMS toolkit for large datasets.

Readability & maintainability practices used here
-------------------------------------------------
1) Small public surface: a single class `TDMSDataset` + clear dataclasses.
2) Cohesive methods: index -> iterate -> materialize -> analyze -> export.
3) Docstrings in NumPy style; type hints everywhere.
4) Helpers for repeated logic (time axis building, property reads).
5) Logging instead of prints; explicit errors and early returns.
6) Optional deps (pandas, zarr) are isolated behind feature checks.
7) Comments explain *why*, not obvious line-by-line behavior.
8) Consistent naming and return types; no implicit global state.

Typical usage
-------------
>>> from tdms_tools import TDMSDataset
>>> ds = TDMSDataset(r"C:\data\run01")
>>> ds.build_index(channels=["wfg"])
>>> for chunk in ds.iter_chunks(channels=["wfg"], chunk_samples=1_000_000):
...     process(chunk.t, chunk.data["wfg"])

Optional exports / analytics
----------------------------
- ds.to_zarr("out.zarr", channels=["wfg"])   # disk-backed store
- ds.estimate_dt_drift("wfg", dt_nominal=1/200_000)  # metadata-only drift
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union
import argparse
import logging
import numpy as np
from nptdms import TdmsFile
import zarr
import pandas as pd


logger = logging.getLogger(__name__)
if not logger.handlers:
    # Library defaults to WARNING; applications can reconfigure.
    logging.basicConfig(level=logging.WARNING)


# ----------------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------------

# ---- Shared dataclasses & helpers (module scope) ----------------------------


@dataclass(frozen=True)
class ChannelMeta:
    """Per-file, per-channel metadata used for indexing and checks."""

    file: Path
    group: str
    channel: str
    n: int
    dt: float  # seconds per sample (wf_increment)
    sr: float  # Hz
    t0_abs: np.datetime64  # absolute start time

    @property
    def t_end_abs(self) -> np.datetime64:
        ns = int(round(self.n * self.dt * 1e9))
        return self.t0_abs + np.timedelta64(ns, "ns")


@dataclass
class Chunk:
    """A streamed chunk of samples for one or more channels."""

    t: np.ndarray
    data: Dict[str, np.ndarray]
    absolute: bool
    gap_before: float
    source: Tuple[Path, int, int]


def _read_dt_t0(ch, grp, td) -> Tuple[Optional[float], Optional[np.datetime64]]:
    """Retrieve (dt, t0_abs) without materializing time_track()."""
    dt = None
    t0 = None
    for props in (ch.properties, grp.properties, td.properties):
        if dt is None and "wf_increment" in props:
            dt = float(props["wf_increment"])
        if t0 is None and "wf_start_time" in props:
            t0 = np.datetime64(props["wf_start_time"], "ns")
        if dt is not None and t0 is not None:
            break
    return dt, t0


def _time_axis(
    start: int,
    count: int,
    dt: float,
    absolute: bool,
    t0_abs: Optional[np.datetime64],
    t0_rel: float,
) -> np.ndarray:
    """Create a time axis of length `count` starting at sample `start`."""
    if not absolute:
        # relative seconds; caller passes t0_rel = start*dt
        return t0_rel + np.arange(0, count, dtype=np.float64) * dt
    if t0_abs is None:
        raise ValueError("absolute time requested but t0_abs is missing")
    step_ns = int(round(dt * 1e9))
    offs_ns0 = int(round(start * dt * 1e9))
    td_ns = (np.arange(0, count, dtype=np.int64) * step_ns + offs_ns0).astype("timedelta64[ns]")
    return t0_abs + td_ns


def _require_zarr() -> None:
    """Guarded import error for optional zarr dependency."""
    if zarr is None:
        raise RuntimeError("Zarr support requires 'zarr'. Install via: pip install zarr numcodecs")


__all__ = ["TDMSDataset", "ChannelMeta", "Chunk"]
# ----------------------------------------------------------------------------


@dataclass
class TDMSDataset:
    """Index and stream TDMS files from a folder (readable edition).

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing .tdms files.
    group : str, optional
        Group name to target. If None, the first group in each file is used.
    strict : bool, default True
        If True, assert identical dt/t0 for all requested channels *within* a file.
    """

    def __init__(
        self, folder: Union[str, Path], group: Optional[str] = None, strict: bool = True
    ) -> None:
        self.folder = Path(folder)
        self.group = group
        self.strict = strict

        self.files: List[Path] = sorted(self.folder.glob("*.tdms"))
        if not self.files:
            raise FileNotFoundError(f"No .tdms files in {self.folder}")

        self._index: List[ChannelMeta] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def build_index(self, channels: Optional[Sequence[str]] = None) -> List[ChannelMeta]:
        """Scan files and collect ChannelMeta rows (sorted by t0_abs)."""
        rows: List[ChannelMeta] = []
        for p in self.files:
            with TdmsFile.open(p) as td:
                grp = td[self.group] if self.group else td.groups()[0]
                chs = grp.channels()
                if channels is not None:
                    filt = set(channels)
                    chs = [c for c in chs if c.name in filt]
                # optional intra-file consistency check (lengths)
                if self.strict and chs:
                    lens = {len(c) for c in chs}
                    if len(lens) != 1:
                        raise RuntimeError(f"Inconsistent channel lengths in {p}")
                for ch in chs:
                    n = len(ch)
                    dt, t0 = _read_dt_t0(ch, grp, td)
                    if dt is None or t0 is None:
                        raise RuntimeError(f"Missing timing metadata in {p}:{grp.name}/{ch.name}")
                    rows.append(
                        ChannelMeta(
                            file=p,
                            group=grp.name,
                            channel=ch.name,
                            n=n,
                            dt=dt,
                            sr=1.0 / dt,
                            t0_abs=t0,
                        )
                    )
        rows.sort(key=lambda m: (m.t0_abs, m.file.as_posix(), m.channel))
        self._index = rows
        logger.debug("Indexed %d entries across %d files", len(rows), len(self.files))
        return rows

    def channels(self) -> List[str]:
        if not self._index:
            self.build_index()
        return sorted({m.channel for m in self._index})

    def index_df(self):
        """Return a pandas.DataFrame view of the index (if pandas available)."""
        if pd is None:
            raise RuntimeError("pandas is not installed.")
        if not self._index:
            self.build_index()
        return pd.DataFrame(
            [
                {
                    "file": m.file,
                    "group": m.group,
                    "channel": m.channel,
                    "n": m.n,
                    "dt": m.dt,
                    "sr": m.sr,
                    "t0_abs": m.t0_abs,
                    "t_end_abs": m.t_end_abs,
                }
                for m in self._index
            ]
        )

    # ------------------------------------------------------------------
    # Iteration / streaming
    # ------------------------------------------------------------------
    def iter_chunks(
        self,
        channels: Union[str, Sequence[str]],
        *,
        chunk_samples: Optional[int] = None,  # samples per chunk
        chunk_sec: Optional[float] = None,  # or seconds per chunk
        hop: Optional[int] = None,  # step in samples (overlap if < chunk)
        t0: Optional[float] = None,  # seconds from dataset start
        t1: Optional[float] = None,  # seconds from dataset start
        absolute_time: bool = True,  # return datetime64[ns] time axis if True
        drop_incomplete: bool = True,  # drop final short chunk
        tail_fill: str = "skip",  # "skip" | "nan" | "zero"
        skip_nan_chunks: bool = False,
        insert_gaps: bool = False,  # accepted for API parity (no-op here)
        dtype: Optional[np.dtype] = None,
    ) -> Generator[Chunk, None, None]:
        """
        Yields Chunk(t, data, absolute, gap_before, source) across file boundaries.
        """
        if isinstance(channels, str):
            channels = [channels]
        ch_list = list(channels)
        if not ch_list:
            return

        # Index ensures we know dt and start time
        if not self._index:
            self.build_index(channels=ch_list)

        # Use primary channel for timing
        primary = ch_list[0]
        metas = [m for m in self._index if m.channel == primary]
        if not metas:
            return
        metas.sort(key=lambda m: (m.t0_abs, m.file.as_posix()))

        # Sanity: constant dt across files (within 1e-9 s tolerance)
        dts = np.array([m.dt for m in metas], dtype=float)
        if np.ptp(dts) > 1e-9:
            raise RuntimeError(f"Sample interval (dt) varies across files for '{primary}'.")

        dt = float(np.median(dts))
        fs = 1.0 / dt
        dataset_start_abs = min(m.t0_abs for m in metas)

        # Resolve chunking
        if chunk_samples is None and chunk_sec is None:
            chunk_samples = 16384
        if chunk_samples is None:
            chunk_samples = max(1, int(round(chunk_sec * fs)))
        if hop is None:
            hop = chunk_samples
        if hop <= 0:
            raise ValueError("hop must be >= 1")
        N = int(chunk_samples)

        # Global sample window [s0, s1)
        s0 = 0 if t0 is None else max(0, int(np.floor(t0 * fs)))
        s1 = None if t1 is None else max(s0, int(np.ceil(t1 * fs)))

        # Rolling buffers across files
        buf: Dict[str, np.ndarray] = {ch: np.empty(0, dtype=dtype or float) for ch in ch_list}
        t0_buf_samples: Optional[int] = None
        global_cursor = 0  # samples from dataset start up to current file
        prev_chunk_start: Optional[int] = None
        prev_len: Optional[int] = None

        last_path: Optional[Path] = None

        for p in self.files:
            with TdmsFile.read(str(p), memmap_dir=".") as td:
                grp = td[self.group] if self.group else td.groups()[0]
                # resolve channels in this file
                chan_objs = {}
                for ch in ch_list:
                    try:
                        chan_objs[ch] = next(c for c in grp.channels() if c.name == ch)
                    except StopIteration:
                        raise KeyError(f"Channel {ch!r} not found in {p}")
                if self.strict:
                    lens = {len(c) for c in chan_objs.values()}
                    if len(lens) != 1:
                        raise RuntimeError(f"Inconsistent channel lengths in {p}")
                n_file = len(next(iter(chan_objs.values())))
                if n_file == 0:
                    continue

                # Skip files entirely before s0
                if global_cursor + n_file <= s0:
                    global_cursor += n_file
                    continue

                # Restrict to [s0, s1)
                start_in_file = max(0, s0 - global_cursor)
                end_in_file = n_file if s1 is None else max(0, min(n_file, s1 - global_cursor))
                if end_in_file <= start_in_file:
                    break

                # Read segments
                segs = {}
                for ch, cob in chan_objs.items():
                    seg = np.asarray(cob[start_in_file:end_in_file])
                    if dtype is not None and seg.dtype != dtype:
                        seg = seg.astype(dtype, copy=False)
                    else:
                        seg = seg.astype(float, copy=False)
                    segs[ch] = seg

                if t0_buf_samples is None:
                    t0_buf_samples = global_cursor + start_in_file

                # Concatenate tail buffer + new data
                cats = {ch: np.concatenate([buf[ch], segs[ch]], axis=0) for ch in ch_list}

                # Emit fixed-size windows
                s = 0
                limit = cats[primary].shape[0] - N + 1
                while s < limit:
                    start_sample = t0_buf_samples + s

                    # Build time axis
                    if absolute_time:
                        t_axis = _time_axis(
                            start_sample, N, dt, absolute=True, t0_abs=dataset_start_abs, t0_rel=0.0
                        )
                    else:
                        t0_rel = start_sample * dt
                        t_axis = _time_axis(0, N, dt, absolute=False, t0_abs=None, t0_rel=t0_rel)

                    # Assemble data window
                    data = {ch: cats[ch][s : s + N] for ch in ch_list}

                    # NaN-skip policy
                    if skip_nan_chunks and any(np.isnan(arr).any() for arr in data.values()):
                        s += hop
                        continue

                    # Gap relative to previous emitted chunk (seconds)
                    if prev_chunk_start is None:
                        gap_before = 0.0
                    else:
                        gap_before = (start_sample - (prev_chunk_start + prev_len)) * dt

                    yield Chunk(
                        t=t_axis,
                        data=data,
                        absolute=absolute_time,
                        gap_before=float(gap_before),
                        source=(p, start_in_file + s, start_in_file + s + N),
                    )
                    last_path = p
                    prev_chunk_start = start_sample
                    prev_len = N
                    s += hop

                # Carry leftover
                for ch in ch_list:
                    buf[ch] = cats[ch][s:]
                t0_buf_samples = t0_buf_samples + s
                global_cursor += n_file

                if s1 is not None and global_cursor >= s1:
                    break

        # Tail (optional)
        have_tail = next(iter(buf.values())).size > 0
        if have_tail and not drop_incomplete:
            # pad or skip tail
            tail_sizes = {ch: buf[ch].size for ch in ch_list}
            if any(sz == 0 for sz in tail_sizes.values()):
                # if some channel ended earlier, harmonize by pad
                pass
            if tail_fill == "skip":
                return
            need = N - next(iter(buf.values())).size
            if need <= 0:
                need = 0
            if tail_fill == "nan":
                pad = {ch: np.full(need, np.nan, dtype=dtype or float) for ch in ch_list}
            elif tail_fill == "zero":
                pad = {ch: np.zeros(need, dtype=dtype or float) for ch in ch_list}
            else:
                raise ValueError("tail_fill must be 'skip' | 'nan' | 'zero'")

            data = {ch: np.concatenate([buf[ch], pad[ch]], axis=0) for ch in ch_list}
            start_sample = t0_buf_samples
            if absolute_time:
                t_axis = _time_axis(start_sample, N, dt, True, dataset_start_abs, 0.0)
            else:
                t_axis = _time_axis(0, N, dt, False, None, start_sample * dt)

            gap_before = (
                0.0
                if prev_chunk_start is None
                else (start_sample - (prev_chunk_start + (prev_len or 0))) * dt
            )
            yield Chunk(
                t=t_axis,
                data=data,
                absolute=absolute_time,
                gap_before=float(gap_before),
                source=(last_path or self.files[-1], 0, N),
            )

    # ------------------------------------------------------------------
    # Convenience materializers
    # ------------------------------------------------------------------
    def read_all(
        self,
        channel: str,
        absolute_time: bool = False,
        dtype: Optional[np.dtype] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Materialize a single channel across all files (uses iter_chunks)."""
        t_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for ch in self.iter_chunks(channels=[channel], absolute_time=absolute_time, dtype=dtype):
            if ch.t.size and channel in ch.data:
                t_parts.append(ch.t)
                y_parts.append(ch.data[channel])
        if not t_parts:
            return np.array([]), np.array([])
        return np.concatenate(t_parts), np.concatenate(y_parts)

    def integrate(
        self,
        channel: str,
        t_start: Optional[Union[np.datetime64, float]] = None,
        t_stop: Optional[Union[np.datetime64, float]] = None,
        absolute_time: bool = True,
    ) -> Tuple[float, int]:
        """Streaming trapezoidal integral over a time window (no full load)."""
        total = 0.0
        samples = 0
        prev_t = None
        prev_y = None

        for ch in self.iter_chunks(channels=[channel], absolute_time=absolute_time):
            t = ch.t
            y = ch.data[channel]
            if t.size == 0:
                continue

            # Cheap per-chunk windowing
            if t_start is not None:
                t_mask = (t >= t_start) if absolute_time else (t >= float(t_start))
                if not np.any(t_mask):
                    continue
                t = t[t_mask]
                y = y[-t.size :]  # align y to sliced t
            if t_stop is not None:
                t_mask = (t <= t_stop) if absolute_time else (t <= float(t_stop))
                t = t[t_mask]
                y = y[: t.size]
            if t.size == 0:
                continue

            if absolute_time:
                t_sec = (t.astype("datetime64[ns]") - t[0]) / np.timedelta64(1, "s")
                if prev_t is not None and prev_y is not None:
                    dt_bridge = (t[0] - prev_t) / np.timedelta64(1, "s")
                    if dt_bridge > 0:
                        total += 0.5 * (prev_y + y[0]) * float(dt_bridge)
                        samples += 1
                if t_sec.size >= 2:
                    total += np.trapz(y, x=t_sec)
                    samples += t.size
                prev_t = t[-1].astype("datetime64[ns]")
            else:
                if prev_t is not None and prev_y is not None:
                    dt_bridge = float(t[0] - prev_t)
                    if dt_bridge > 0:
                        total += 0.5 * (prev_y + y[0]) * dt_bridge
                        samples += 1
                if t.size >= 2:
                    total += np.trapz(y, x=t)
                    samples += t.size
                prev_t = float(t[-1])
            prev_y = y[-1]

            if t_stop is not None and (
                (absolute_time and t[-1] > t_stop)
                or (not absolute_time and float(t[-1]) > float(t_stop))
            ):
                break

        return float(total), int(samples)

    # ------------------------------------------------------------------
    # Exports / analytics
    # ------------------------------------------------------------------
    def to_zarr(
        self,
        out_path: Union[str, Path],
        channels: Optional[Sequence[str]] = None,
        chunk_samples: int = 1_000_000,
        absolute_time: bool = False,
        insert_gaps: bool = False,
        dtype: Optional[np.dtype] = None,
        overwrite: bool = False,
    ) -> Path:
        """Write selected channels to a Zarr store (streamed, chunked)."""
        _require_zarr()
        if not self._index:
            self.build_index(channels=channels)
        ch_names = list(sorted(channels)) if channels is not None else self.channels()
        if not ch_names:
            raise ValueError("No channels to export.")

        # Total samples from the primary channel
        primary = ch_names[0]
        total = sum(m.n for m in self._index if m.channel == primary)
        if total == 0:
            raise RuntimeError(f"Primary channel '{primary}' has zero samples.")

        out_path = Path(out_path)
        mode = "w" if overwrite else "w-"
        root = zarr.open_group(str(out_path), mode=mode)

        time_arr = None
        ch_arrays: Dict[str, "zarr.Array"] = {}  # type: ignore[name-defined]

        epoch = np.datetime64("1970-01-01T00:00:00", "ns")
        offset = 0
        sr_first: Optional[float] = None

        it = self.iter_chunks(
            channels=ch_names,
            chunk_samples=chunk_samples,
            absolute_time=absolute_time,
            insert_gaps=insert_gaps,
            dtype=dtype,
        )
        for chunk in it:
            if chunk.t.size == 0:
                continue  # gap marker (not used by default)

            if time_arr is None:
                # Create datasets after first chunk reveals dtype
                chunk_len = min(chunk_samples, total)
                if absolute_time:
                    time_arr = root.require_dataset(
                        "time", shape=(total,), chunks=(chunk_len,), dtype="int64"
                    )
                    root.attrs["time_units"] = "ns since 1970-01-01T00:00:00"
                else:
                    time_arr = root.require_dataset(
                        "time", shape=(total,), chunks=(chunk_len,), dtype="float64"
                    )
                    root.attrs["time_units"] = "s since first_sample"
                ch_group = root.require_group("ch")
                for name, arr in chunk.data.items():
                    dtyp = arr.dtype if dtype is None else np.dtype(dtype)
                    ch_arrays[name] = ch_group.require_dataset(
                        name, shape=(total,), chunks=(chunk_len,), dtype=dtyp
                    )

            # Write time
            if absolute_time:
                t_ns = (
                    (chunk.t.astype("datetime64[ns]") - epoch)
                    .astype("timedelta64[ns]")
                    .astype("int64")
                )
                time_arr[offset : offset + t_ns.size] = t_ns
            else:
                time_arr[offset : offset + chunk.t.size] = chunk.t.astype("float64")

            # Write channels
            for name, arr in chunk.data.items():
                out = (
                    arr if (dtype is None or arr.dtype == dtype) else arr.astype(dtype, copy=False)
                )
                ch_arrays[name][offset : offset + out.size] = out

            if sr_first is None:
                m = next(
                    (
                        m
                        for m in self._index
                        if (m.file == chunk.source[0] and m.channel == primary)
                    ),
                    None,
                )
                sr_first = None if m is None else m.sr

            offset += chunk.t.size

        # Write a light-weight index group for downstream timing tools
        metas = [m for m in self._index if m.channel == primary]
        metas.sort(key=lambda m: (m.t0_abs, m.file.as_posix()))
        idx_grp = root.require_group("index")
        offsets = np.zeros(len(metas), dtype="int64")
        k = 0
        for i, m in enumerate(metas):
            offsets[i] = k
            k += m.n
        idx_grp.require_dataset(
            "sample_offset", shape=offsets.shape, chunks=offsets.shape, dtype="int64"
        )[:] = offsets
        idx_grp.require_dataset("n", shape=(len(metas),), chunks=(len(metas),), dtype="int64")[
            :
        ] = np.array([m.n for m in metas])
        idx_grp.require_dataset("dt", shape=(len(metas),), chunks=(len(metas),), dtype="float64")[
            :
        ] = np.array([m.dt for m in metas])
        t0_ns = (
            (np.array([m.t0_abs for m in metas], dtype="datetime64[ns]") - epoch)
            .astype("timedelta64[ns]")
            .astype("int64")
        )
        idx_grp.require_dataset("t0_abs_ns", shape=t0_ns.shape, chunks=t0_ns.shape, dtype="int64")[
            :
        ] = t0_ns
        idx_grp.attrs.update({"files": [m.file.as_posix() for m in metas]})

        # Store-level metadata
        root.attrs.update(
            {
                "channels": ch_names,
                "primary_channel": primary,
                "total_samples": int(total),
                "sr_hz": None if sr_first is None else float(sr_first),
                "absolute_time": bool(absolute_time),
                "chunk_samples": int(chunk_samples),
                "strict": bool(self.strict),
                "group": self.group,
            }
        )
        return out_path

    def estimate_dt_drift(
        self,
        channel: str,
        dt_nominal: Optional[float] = None,
    ) -> Tuple[dict, np.ndarray]:
        """Estimate effective dt (and bias) from metadata only.

        Returns
        -------
        summary : dict
            Keys include dt_hat, sr_hat, bias_ppm, drift_ms_per_hour.
        residuals : np.ndarray
            Columns [k, t_sec, t_fit_sec, resid_sec] at file starts/ends.
        """
        if not self._index:
            self.build_index(channels=[channel])
        metas = [m for m in self._index if m.channel == channel]
        if not metas:
            raise ValueError(f"Channel '{channel}' not found in index.")
        metas.sort(key=lambda m: (m.t0_abs, m.file.as_posix()))

        # Sample index ↔ absolute time pairs (file start and end)
        k_list: List[int] = []
        t_list: List[np.datetime64] = []
        k = 0
        for m in metas:
            k_list.extend([k, k + m.n])
            t_list.extend([m.t0_abs, m.t_end_abs])
            k += m.n

        k_arr = np.array(k_list, dtype=np.int64)
        t0 = t_list[0]
        t_sec = (np.array(t_list, dtype="datetime64[ns]") - t0).astype("timedelta64[ns]").astype(
            "int64"
        ) / 1e9

        # Linear model: t ≈ a + dt_hat * k
        A = np.vstack([k_arr.astype(float), np.ones_like(k_arr, dtype=float)]).T
        dt_hat, a = np.linalg.lstsq(A, t_sec, rcond=None)[0]

        if dt_nominal is None:
            dt_nominal = float(np.median([m.dt for m in metas]))

        bias = dt_hat - dt_nominal
        bias_ppm = (bias / dt_nominal) * 1e6
        sr_hat = 1.0 / dt_hat if dt_hat != 0 else np.nan
        drift_ms_per_hour = (bias * 3600.0) * 1e3

        t_fit = a + dt_hat * k_arr
        resid = t_sec - t_fit
        residuals = np.column_stack([k_arr, t_sec, t_fit, resid])

        summary = {
            "dt_hat": float(dt_hat),
            "sr_hat": float(sr_hat),
            "bias_ppm": float(bias_ppm),
            "drift_ms_per_hour": float(drift_ms_per_hour),
            "dt_nominal": float(dt_nominal),
            "intercept_sec": float(a),
            "samples_total": int(k_arr.max(initial=0)),
            "files": len(metas),
        }
        return summary, residuals

    # -----------------------------------------------------
    # Time Series Read-out
    # -----------------------------------------------------
    def read_window_abs(
        self,
        channel: str,
        t0_abs: np.datetime64,
        t1_abs: np.datetime64,
        *,
        chunk_samples: int = 1_000_000,
        dtype: Optional[np.dtype] = None,
        return_relative: bool = False,
        rel_origin: Optional[np.datetime64] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Materialize (t, y) for `channel` in [t0_abs, t1_abs] across all files.

        If return_relative=True, returns t as float seconds since `rel_origin`
        (default: t0_abs if rel_origin is None). Otherwise returns datetime64[ns].
        """
        if t1_abs <= t0_abs:
            return np.array([]), np.array([])

        # Ensure index is ready
        if not self._index:
            self.build_index(channels=[channel])

        metas = [m for m in self._index if m.channel == channel]
        if not metas:
            return np.array([]), np.array([])

        # Preselect overlapping files
        def overlaps(m: ChannelMeta) -> bool:
            return not (m.t_end_abs < t0_abs or m.t0_abs > t1_abs)

        subset_files = [m.file for m in metas if overlaps(m)]
        if not subset_files:
            return np.array([]), np.array([])

        # Preserve original order
        subset_files = [p for p in self.files if p in set(subset_files)]

        # Temporarily narrow file list
        orig_files = self.files
        self.files = subset_files
        t_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        try:
            for ch in self.iter_chunks(
                channels=[channel],
                chunk_samples=chunk_samples,
                absolute_time=True,
                dtype=dtype,
            ):
                if ch.t.size == 0:
                    continue
                if ch.t[-1] < t0_abs:
                    continue
                if ch.t[0] > t1_abs:
                    break

                mask = (ch.t >= t0_abs) & (ch.t <= t1_abs)
                if np.any(mask):
                    t_parts.append(ch.t[mask])
                    y_parts.append(ch.data[channel][mask])
        finally:
            self.files = orig_files

        if not t_parts:
            return np.array([]), np.array([])

        t_abs = np.concatenate(t_parts).astype("datetime64[ns]")
        y = np.concatenate(y_parts)

        if return_relative:
            origin = t0_abs if rel_origin is None else rel_origin
            t_rel = (t_abs - origin) / np.timedelta64(1, "s")
            return t_rel.astype(float), y
        else:
            return t_abs, y

    def read_window_rel(
        self,
        channel: str,
        t0_rel_s: float,
        t1_rel_s: float,
        *,
        chunk_samples: int = 1_000_000,
        dtype: Optional[np.dtype] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.datetime64]:
        """
        Materialize (t_rel_s, y, dataset_start_abs) where t_rel_s are seconds since
        the dataset's absolute start for `channel`.
        """
        if not self._index:
            self.build_index(channels=[channel])
        metas = [m for m in self._index if m.channel == channel]
        if not metas:
            return np.array([]), np.array([]), np.datetime64("NaT", "ns")

        dataset_start_abs = min(m.t0_abs for m in metas)

        def s_to_ns(s: float) -> np.timedelta64:
            return np.timedelta64(int(round(float(s) * 1e9)), "ns")

        t0_abs = dataset_start_abs + s_to_ns(t0_rel_s)
        t1_abs = dataset_start_abs + s_to_ns(t1_rel_s)

        t_rel_s_arr, y = self.read_window_abs(
            channel,
            t0_abs,
            t1_abs,
            chunk_samples=chunk_samples,
            dtype=dtype,
            return_relative=True,
            rel_origin=dataset_start_abs,
        )
        return t_rel_s_arr, y, dataset_start_abs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TDMS toolkit (streaming, export, drift).")
    ap.add_argument("folder", type=str, help="Folder with .tdms files")
    ap.add_argument("--group", type=str, default=None, help="TDMS group name")
    ap.add_argument("--channel", type=str, default=None, help="Channel for examples")
    ap.add_argument(
        "--to-zarr", dest="to_zarr", type=str, default=None, help="Write Zarr store at path"
    )
    ap.add_argument("--drift", action="store_true", help="Estimate dt drift for --channel")
    args = ap.parse_args()

    ds = TDMSDataset(args.folder, group=args.group)
    if args.to_zarr:
        ds.build_index(channels=[args.channel] if args.channel else None)
        out = ds.to_zarr(
            args.to_zarr,
            channels=[args.channel] if args.channel else None,
            absolute_time=False,
            overwrite=True,
        )
        print("Wrote Zarr:", out)

    if args.drift and args.channel:
        summary, _ = ds.estimate_dt_drift(args.channel)
        print(summary)
