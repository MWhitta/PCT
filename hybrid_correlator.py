"""
hybrid_correlator.py -- Build a single wide-dynamic-range g2(tau)-1 curve by
gracefully merging the hardware multi-tau correlogram (fast lags) with a
software correlation of the raw photon-count-rate traces (slow lags).

Motivation
----------
A hardware correlator (e.g. LS Instruments, pseudo-cross-correlation) reports
g2 down to ~ns-us lags but, at long lags, suffers from triangular-averaging
distortion, a small number of independent decorrelation windows, and slow
"artifacts" from large aggregates that are poorly characterized.  Meanwhile the
instrument streams a photon count-rate trace read out every few milliseconds
for the whole acquisition (here ~52 ms bins over ~240 s).  Software-correlating
that trace -- and averaging across replicates -- yields a *better* estimate of
the slow dynamics than the hardware's long-lag channels.  The default hybrid
builder uses only software-correlation lags at or above 100 ms, so the first
coarse-bin pooled/stitched lag point cannot steer the merge or inversion.

Rather than hard-truncating the hardware curve at some lag, this module merges
the two estimators with **inverse-variance (graceful) blending** on a common
lag grid:

    g2_merged(tau) = [ g2_hw/sigma_hw^2 + g2_sw/sigma_sw^2 ]
                     / [ 1/sigma_hw^2 + 1/sigma_sw^2 ]
    sigma_merged(tau) = 1 / sqrt( 1/sigma_hw^2 + 1/sigma_sw^2 )

so that each lag is dominated by whichever measurement is locally more
reliable, with a smooth transition through the overlap region.  The merged
sigma is exactly what CORENN (see corenn.py) needs to weight the inversion.

The software correlation uses the *symmetric normalization*
``g2(k) = <a_i b_{i+k}> / (<a_i> <b_{i+k}>)`` with the direct/delayed monitor
averages taken over the same overlapping segment (reduces drift bias), and
defaults to A x B *cross*-correlation to suppress detector afterpulsing and
dead-time -- mirroring the hardware's pseudo-cross-correlation.

Author: Michael L. Whittaker (and Claude), 2026.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import corenn as C


# ---------------------------------------------------------------------------
# Software multi-tau / log-lag correlation
# ---------------------------------------------------------------------------
def _log_integer_lags(n: int, per_octave: int = 8, max_frac: float = 0.2
                      ) -> np.ndarray:
    """Unique integer lags log-spaced from 1 to ``max_frac * n``.

    Capping at a fraction of the record length keeps each lag estimate built
    from many independent products (good long-lag statistics).
    """
    kmax = max(2, int(n * max_frac))
    n_oct = max(1.0, np.log2(kmax))
    count = int(np.ceil(n_oct * per_octave)) + 1
    lags = np.unique(np.round(np.geomspace(1, kmax, count)).astype(int))
    return lags[lags >= 1]


def _filter_lag_min(
    tau: np.ndarray,
    arrays: Sequence[np.ndarray],
    lag_min: Optional[float],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Drop software-correlation lags below the trusted handoff lag."""
    tau = np.asarray(tau, float)
    out = [np.asarray(a) for a in arrays]
    if lag_min is None:
        return tau, out
    keep = tau >= float(lag_min)
    if not np.any(keep):
        raise ValueError(f"No software-correlation lags >= {lag_min:g} s.")
    return tau[keep], [a[keep] for a in out]


def software_correlate(signal_a: np.ndarray, dt: float,
                       signal_b: Optional[np.ndarray] = None,
                       per_octave: int = 8, max_frac: float = 0.2
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Log-lag estimate of ``g2(tau) - 1`` from a uniformly sampled trace.

    Uses symmetric normalization.  When ``signal_b`` is given, the A x B cross
    correlation is computed (afterpulse/dead-time immune); otherwise the
    autocorrelation is returned.

    Returns ``(tau, g2_minus_1)``.
    """
    a = np.asarray(signal_a, float)
    b = a if signal_b is None else np.asarray(signal_b, float)
    n = a.size
    lags = _log_integer_lags(n, per_octave, max_frac)
    tau = lags * dt
    g2m = np.empty(lags.size)
    for i, k in enumerate(lags):
        ad = a[: n - k]
        bd = b[k:]
        mean_d = ad.mean()
        mean_l = bd.mean()
        denom = mean_d * mean_l
        g2m[i] = (np.mean(ad * bd) / denom - 1.0) if denom > 0 else 0.0
    return tau, g2m


def software_correlate_gapped(
    signal_a: np.ndarray,
    dt: float,
    signal_b: Optional[np.ndarray] = None,
    per_octave: int = 8,
    max_frac: float = 0.2,
    min_pairs: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-lag ``g2(tau)-1`` from a stitched trace with NaN gaps.

    NaN samples mark intervals where no counts were acquired between sequential
    repetitions.  Pairs whose endpoints are both observed are retained, including
    cross-repetition pairs at their true elapsed separation; pairs touching a
    gap are excluded.  The normalization is recomputed from the valid direct and
    delayed endpoints at each lag.

    Returns ``(tau, g2_minus_1, n_pairs)``.
    """
    a = np.asarray(signal_a, float)
    b = a if signal_b is None else np.asarray(signal_b, float)
    if a.shape != b.shape:
        raise ValueError("Gapped correlation channels must have the same shape.")
    n = a.size
    lags = _log_integer_lags(n, per_octave, max_frac)
    tau, g2m, n_pairs = [], [], []
    for k in lags:
        ad = a[: n - k]
        bd = b[k:]
        good = np.isfinite(ad) & np.isfinite(bd)
        count = int(np.count_nonzero(good))
        if count < min_pairs:
            continue
        adv = ad[good]
        bdv = bd[good]
        mean_d = float(adv.mean())
        mean_l = float(bdv.mean())
        denom = mean_d * mean_l
        if denom <= 0:
            continue
        tau.append(k * dt)
        g2m.append(float(np.mean(adv * bdv) / denom - 1.0))
        n_pairs.append(count)
    if not tau:
        raise ValueError("No lags have enough valid pairs for gapped correlation.")
    return np.asarray(tau), np.asarray(g2m), np.asarray(n_pairs, int)


def _parse_start_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%y %I:%M:%S %p",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def _duration_seconds(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        secs = 0.0
        for part in text.split(":"):
            secs = secs * 60.0 + float(part)
        return secs
    try:
        return float(text)
    except ValueError:
        return float("nan")


def stitch_count_traces(
    traces: Sequence[Dict[str, object]],
    channel_a: str = "cr_a",
    channel_b: str = "cr_b",
    clip_negative_gaps: bool = True,
) -> Dict[str, object]:
    """Place sequential repetition count traces onto one elapsed-time grid.

    Repetition ``start_datetime`` metadata is used when available.  Missing
    acquisition intervals are represented as NaNs, so
    :func:`software_correlate_gapped` can include only physically observed
    endpoint pairs.  If timestamps imply a small negative gap, likely from
    one-second timestamp precision or rounded duration metadata, the later
    segment is shifted to start at the previous segment end when
    ``clip_negative_gaps`` is true.
    """
    usable = []
    for i, tr in enumerate(traces):
        t = np.asarray(tr.get("time"), float)
        a = tr.get(channel_a)
        if t.size < 2 or a is None:
            continue
        start = tr.get("start_datetime")
        duration = _duration_seconds(tr.get("duration_s"))
        dt = float(np.median(np.diff(t)))
        span = float(t[-1] - t[0] + dt)
        usable.append({
            "index": i,
            "trace": tr,
            "time": t,
            "start": start,
            "duration": duration if np.isfinite(duration) else span,
            "span": span,
            "dt": dt,
        })
    if not usable:
        raise ValueError("No usable traces to stitch.")
    dt = float(np.median([u["dt"] for u in usable]))
    have_starts = all(u["start"] is not None for u in usable)
    if have_starts:
        usable.sort(key=lambda u: u["start"])
        t0 = usable[0]["start"]
        for u in usable:
            u["raw_start_s"] = (u["start"] - t0).total_seconds()
    else:
        offset = 0.0
        for u in usable:
            u["raw_start_s"] = offset
            offset += u["duration"]

    prev_end = None
    segments = []
    for u in usable:
        raw_start = float(u["raw_start_s"])
        raw_gap = float("nan") if prev_end is None else raw_start - prev_end
        start = raw_start
        if prev_end is not None and raw_gap < 0 and clip_negative_gaps:
            start = prev_end
        adjusted_gap = float("nan") if prev_end is None else start - prev_end
        end = start + u["span"]
        u["start_s"] = start
        u["end_s"] = end
        segments.append({
            "index": u["index"],
            "start_s": start,
            "end_s": end,
            "raw_gap_s": raw_gap,
            "gap_s": adjusted_gap,
            "duration_s": u["duration"],
            "span_s": u["span"],
            "start_datetime": u["start"],
        })
        prev_end = end

    n = int(np.floor(max(u["end_s"] for u in usable) / dt)) + 1
    a_grid = np.full(n, np.nan)
    b_grid = np.full(n, np.nan)
    for u in usable:
        rel_t = np.asarray(u["time"], float) - float(u["time"][0])
        idx = np.rint((u["start_s"] + rel_t) / dt).astype(int)
        good = (idx >= 0) & (idx < n)
        a_grid[idx[good]] = np.asarray(u["trace"][channel_a], float)[good]
        b = u["trace"].get(channel_b)
        if b is not None:
            b_grid[idx[good]] = np.asarray(b, float)[good]
    out = {
        "time": np.arange(n) * dt,
        channel_a: a_grid,
        "dt": dt,
        "segments": segments,
        "has_absolute_starts": have_starts,
    }
    if np.isfinite(b_grid).any():
        out[channel_b] = b_grid
    return out


def correlate_stitched_traces(
    traces: Sequence[Dict[str, object]],
    cross: bool = True,
    channel_a: str = "cr_a",
    channel_b: str = "cr_b",
    per_octave: int = 8,
    max_frac: float = 0.2,
    min_pairs: int = 32,
    lag_min: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Gap-aware software correlation of sequential repetition traces.

    This is the physically appropriate concatenation mode when repetitions were
    started in rapid succession but are not literally continuous acquisitions.
    """
    stitched = stitch_count_traces(traces, channel_a=channel_a, channel_b=channel_b)
    b = stitched.get(channel_b) if cross else None
    tau, g2m, n_pairs = software_correlate_gapped(
        np.asarray(stitched[channel_a], float),
        float(stitched["dt"]),
        np.asarray(b, float) if b is not None else None,
        per_octave=per_octave,
        max_frac=max_frac,
        min_pairs=min_pairs,
    )
    tau, (g2m, n_pairs) = _filter_lag_min(tau, [g2m, n_pairs], lag_min)
    return tau, g2m, n_pairs, stitched


def correlate_replicate_traces(
    traces: Sequence[Dict[str, np.ndarray]],
    cross: bool = True,
    channel_a: str = "cr_a",
    channel_b: str = "cr_b",
    per_octave: int = 8,
    max_frac: float = 0.2,
    lag_min: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Software-correlate several count traces and ensemble-average them.

    Parameters
    ----------
    traces : list of dicts
        Each has ``'time'`` and at least one count-rate channel (Hz).
    cross : bool
        If True (default) and two channels exist, compute A x B cross
        correlation per replicate; else autocorrelate the first channel.

    Returns
    -------
    tau, g2m_mean, sigma, n_rep
        ``sigma`` is the standard error across replicates (std / sqrt(n_rep)),
        floored to avoid zero weights.
    """
    curves = []
    tau_ref = None
    for tr in traces:
        t = np.asarray(tr["time"], float)
        if t.size < 8:
            continue
        dt = float(np.median(np.diff(t)))
        a = np.asarray(tr.get(channel_a), float)
        b = tr.get(channel_b)
        if cross and b is not None:
            tau, g2m = software_correlate(a, dt, np.asarray(b, float),
                                          per_octave, max_frac)
        else:
            tau, g2m = software_correlate(a, dt, None, per_octave, max_frac)
        if tau_ref is None:
            tau_ref = tau
            curves.append(g2m)
        elif tau.shape == tau_ref.shape and np.allclose(tau, tau_ref, rtol=1e-6):
            curves.append(g2m)
        else:  # resample onto the reference grid in log-tau
            curves.append(np.interp(np.log(tau_ref), np.log(tau), g2m))
    if not curves:
        raise ValueError("No usable count traces.")
    M = np.vstack(curves)
    n_rep = M.shape[0]
    g2m_mean = M.mean(axis=0)
    if n_rep >= 2:
        sigma = M.std(axis=0, ddof=1) / np.sqrt(n_rep)
    else:
        sigma = np.full_like(g2m_mean, np.nan)
    floor = max(np.nanmax(np.abs(g2m_mean)) * 1e-3, 1e-9)
    sigma = np.where(np.isfinite(sigma) & (sigma > floor), sigma, floor)
    tau_ref, (g2m_mean, sigma) = _filter_lag_min(
        tau_ref, [g2m_mean, sigma], lag_min)
    return tau_ref, g2m_mean, sigma, n_rep


# ---------------------------------------------------------------------------
# Hardware-side uncertainty
# ---------------------------------------------------------------------------
def hardware_sigma(tau: np.ndarray, g2m: np.ndarray, beta: float,
                   duration: float, tau_distort: float = 1.0,
                   distort_power: float = 1.5,
                   replicate_sigma: Optional[np.ndarray] = None) -> np.ndarray:
    """Expected uncertainty of the hardware correlogram.

    The hardware multi-tau correlator is the gold standard through the main
    decay, so the base uncertainty there is just the measured replicate scatter
    (or a small Schaetzel floor).  Beyond ``tau_distort`` (seconds) the long-lag
    channels suffer triangular-averaging distortion and pick up irreproducible
    aggregate "artifacts"; we inflate the uncertainty by
    ``(1 + (tau/tau_distort)^distort_power)`` so the blend hands those lags over
    to the software correlation.
    """
    g1 = np.sqrt(np.clip(g2m, 0.0, None) / max(beta, 1e-12))
    floor = 0.0
    if replicate_sigma is not None:
        k = max(3, int(0.15 * tau.size))
        floor = float(np.median(replicate_sigma[:k]))
    base = C.schatzel_sigma(tau, g1, beta, duration, sigma_floor=floor)
    if replicate_sigma is not None:
        base = np.maximum(base, np.asarray(replicate_sigma, float))
    inflate = 1.0 + (np.asarray(tau, float) / tau_distort) ** distort_power
    return base * inflate


def software_sigma(tau: np.ndarray, g2m: np.ndarray, beta: float,
                   duration: float, bin_time: float, n_rep: int = 1,
                   n_reliable_bins: float = 4.0, coarse_power: float = 1.5,
                   replicate_sigma: Optional[np.ndarray] = None) -> np.ndarray:
    """Expected uncertainty of the software (count-trace) correlation.

    Beyond the Schaetzel statistical noise, the software estimate is unreliable
    at its shortest lags (within a few bin times the coarse binning and finite
    fast statistics bias the estimate).  We inflate the uncertainty by
    ``(1 + (n_reliable_bins*bin_time / tau)^coarse_power)`` so the hardware wins
    the main decay.  Ensemble averaging across ``n_rep`` traces lowers the
    statistical floor.
    """
    g1 = np.sqrt(np.clip(g2m, 0.0, None) / max(beta, 1e-12))
    floor = 0.0
    if replicate_sigma is not None:
        floor = float(np.median(replicate_sigma))
    base = C.schatzel_sigma(tau, g1, beta, duration, sigma_floor=floor)
    base = base / max(np.sqrt(n_rep), 1.0)
    if replicate_sigma is not None:
        base = np.maximum(base, np.asarray(replicate_sigma, float) / max(np.sqrt(n_rep), 1.0))
    coarse = 1.0 + (n_reliable_bins * bin_time / np.asarray(tau, float)) ** coarse_power
    return base * coarse


# ---------------------------------------------------------------------------
# Graceful inverse-variance merge
# ---------------------------------------------------------------------------
@dataclass
class MergedCorrelation:
    tau: np.ndarray
    g2m: np.ndarray
    sigma: np.ndarray
    components: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    scale_sw: float = 1.0


def merge_correlations(
    tau_hw: np.ndarray, g2_hw: np.ndarray, sigma_hw: np.ndarray,
    tau_sw: np.ndarray, g2_sw: np.ndarray, sigma_sw: np.ndarray,
    match_overlap: bool = True,
    overlap: Optional[Tuple[float, float]] = None,
    n_grid: int = 220,
) -> MergedCorrelation:
    """Inverse-variance blend of a hardware and a software g2-1 curve.

    The software curve is first rescaled (single positive factor) so that its
    amplitude matches the hardware curve in the overlap region -- both measure
    ``beta * g1(tau)^2`` for the same field correlation, but with different
    coherence factors.  The two are then interpolated onto a common log-lag
    grid and combined with inverse-variance weights; lags covered by only one
    source take that source directly.
    """
    tau_hw = np.asarray(tau_hw, float); g2_hw = np.asarray(g2_hw, float)
    tau_sw = np.asarray(tau_sw, float); g2_sw = np.asarray(g2_sw, float)
    sigma_hw = np.asarray(sigma_hw, float); sigma_sw = np.asarray(sigma_sw, float)

    lo = min(tau_hw.min(), tau_sw.min())
    hi = max(tau_hw.max(), tau_sw.max())
    if overlap is None:
        overlap = (max(tau_hw.min(), tau_sw.min()),
                   min(tau_hw.max(), tau_sw.max()))

    # --- match software amplitude to hardware over the overlap -----------
    scale = 1.0
    if match_overlap and overlap[1] > overlap[0]:
        mask_hw = (tau_hw >= overlap[0]) & (tau_hw <= overlap[1])
        if np.count_nonzero(mask_hw) >= 3:
            g_sw_at_hw = np.interp(np.log(tau_hw[mask_hw]),
                                   np.log(tau_sw), g2_sw)
            w = 1.0 / np.clip(sigma_hw[mask_hw] ** 2, 1e-18, None)
            num = float(np.sum(w * g2_hw[mask_hw] * g_sw_at_hw))
            den = float(np.sum(w * g_sw_at_hw ** 2))
            if den > 0 and np.isfinite(num / den) and num / den > 0:
                scale = num / den
    g2_sw_s = scale * g2_sw
    sigma_sw_s = scale * sigma_sw

    # --- common log-lag grid ---------------------------------------------
    tau = np.geomspace(lo, hi, n_grid)
    log_tau = np.log(tau)

    def _interp(tsrc, gsrc, ssrc):
        g = np.interp(log_tau, np.log(tsrc), gsrc, left=np.nan, right=np.nan)
        s = np.interp(log_tau, np.log(tsrc), ssrc, left=np.nan, right=np.nan)
        return g, s

    g_hw, s_hw = _interp(tau_hw, g2_hw, sigma_hw)
    g_sw, s_sw = _interp(tau_sw, g2_sw_s, sigma_sw_s)

    w_hw = np.where(np.isfinite(g_hw), 1.0 / np.clip(s_hw**2, 1e-18, None), 0.0)
    w_sw = np.where(np.isfinite(g_sw), 1.0 / np.clip(s_sw**2, 1e-18, None), 0.0)
    g_hw0 = np.where(np.isfinite(g_hw), g_hw, 0.0)
    g_sw0 = np.where(np.isfinite(g_sw), g_sw, 0.0)

    wsum = w_hw + w_sw
    good = wsum > 0
    g2m = np.zeros_like(tau)
    sigma = np.full_like(tau, np.nan)
    g2m[good] = (w_hw[good] * g_hw0[good] + w_sw[good] * g_sw0[good]) / wsum[good]
    sigma[good] = 1.0 / np.sqrt(wsum[good])

    return MergedCorrelation(
        tau=tau[good], g2m=g2m[good], sigma=sigma[good], scale_sw=scale,
        components={
            "hardware": {"tau": tau_hw, "g2m": g2_hw, "sigma": sigma_hw},
            "software": {"tau": tau_sw, "g2m": g2_sw_s, "sigma": sigma_sw_s},
            "weights": {"tau": tau[good], "w_hw": w_hw[good], "w_sw": w_sw[good]},
        },
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline from an LS Instruments measurement directory
# ---------------------------------------------------------------------------
def load_measurement_traces(measurement_dir: str,
                            count_trace_file: str = "Count Trace.csv"
                            ) -> List[Dict[str, object]]:
    """Load every repetition's count trace and timing metadata."""
    rep_dirs = sorted(glob.glob(os.path.join(measurement_dir, "Repetition *")))
    traces = []
    for rd in rep_dirs:
        path = os.path.join(rd, count_trace_file)
        if os.path.exists(path):
            try:
                tr = C.load_count_trace(path)
                spath = os.path.join(rd, "Summary.csv")
                if os.path.exists(spath):
                    meta = C.load_summary(spath)
                    tr["start_date"] = meta.get("Start date")
                    tr["start_datetime"] = _parse_start_datetime(meta.get("Start date"))
                    tr["duration_s"] = _duration_seconds(meta.get("Duration [s]"))
                tr["path"] = path
                tr["rep_dir"] = rd
                traces.append(tr)
            except Exception:
                continue
    return traces


def build_hybrid_dataset(
    measurement_dir: str,
    cross: bool = True,
    per_octave: int = 8,
    max_frac: float = 0.2,
    software_lag_min: float = 0.1,
    tau_distort: float = 1.0,
    overlap: Optional[Tuple[float, float]] = None,
    correlation_file: str = "Correlation Function.csv",
) -> Tuple[C.DLSDataset, MergedCorrelation]:
    """Build a wide-range :class:`corenn.DLSDataset` for one measurement.

    Loads the hardware correlogram (replicate-averaged) and the count traces,
    software-correlates the latter across replicates, and gracefully merges the
    two by inverse-variance blending.  Returns the merged dataset (ready for
    :func:`corenn.corenn_invert`) and the :class:`MergedCorrelation` diagnostics.
    By default the software curve is truncated to ``tau >= 0.1 s`` before
    scaling and merging.
    """
    # hardware-only base (avoids recursion: load_measurement_replicates now
    # delegates here when hybrid is requested).
    base, reps = C._load_hardware_replicates(
        measurement_dir, correlation_file=correlation_file)
    sigma_hw_rep = C.empirical_sigma(reps) if len(reps) >= 2 else None
    duration = base.duration if base.duration else float(base.tau[-1])
    sigma_hw = hardware_sigma(base.tau, base.g2m, base.beta, duration,
                              tau_distort=tau_distort,
                              replicate_sigma=sigma_hw_rep)

    traces = load_measurement_traces(measurement_dir)
    if not traces:
        raise ValueError(f"No count traces under {measurement_dir}.")
    tau_sw, g2_sw, sigma_sw_rep, n_rep = correlate_replicate_traces(
        traces, cross=cross, per_octave=per_octave, max_frac=max_frac,
        lag_min=software_lag_min)
    bin_time = float(np.median(np.diff(traces[0]["time"])))
    sigma_sw = software_sigma(tau_sw, g2_sw, base.beta, duration, bin_time,
                              n_rep=n_rep, replicate_sigma=sigma_sw_rep)

    # match software amplitude to hardware in a window straddling the handoff
    if overlap is None:
        overlap = (max(5 * bin_time, base.tau.min()), min(5 * tau_distort,
                   base.tau.max(), tau_sw.max()))
    merged = merge_correlations(
        base.tau, base.g2m, sigma_hw, tau_sw, g2_sw, sigma_sw,
        match_overlap=True, overlap=overlap)

    ds = C.DLSDataset(
        tau=merged.tau, g2m=merged.g2m, q=base.q, beta=base.beta,
        sigma=merged.sigma, temperature=base.temperature,
        viscosity=base.viscosity, n_solvent=base.n_solvent,
        duration=duration, count_rate=base.count_rate,
        metadata={**base.metadata, "hybrid": True, "n_rep_trace": n_rep,
                  "scale_sw": merged.scale_sw,
                  "software_lag_min_s": software_lag_min},
    )
    return ds, merged


def build_stitched_dataset(
    measurement_dir: str,
    cross: bool = True,
    per_octave: int = 8,
    max_frac: float = 0.2,
    software_lag_min: float = 0.1,
    tau_distort: float = 1.0,
    min_pairs: int = 100,
    overlap: Optional[Tuple[float, float]] = None,
    correlation_file: str = "Correlation Function.csv",
) -> Tuple["C.DLSDataset", MergedCorrelation]:
    """Wide-range :class:`corenn.DLSDataset` from the GAP-AWARE STITCHED software
    correlation of the sequential count-rate traces.

    Unlike :func:`build_hybrid_dataset` (which pools the per-repetition software
    curves), this concatenates every repetition into one gap-aware time series
    (inter-repetition gaps are NaN-filled and skipped by the estimator) and
    correlates it as a single record.  The software correlation therefore reaches
    lags approaching the *full elapsed time* of the run (minutes, not the few
    minutes of one repetition), giving clean long-lag statistics.  Merged with
    the hardware short-lag curve, this extends the resolvable range to large
    (slow / aggregate) radii -- see ``demo_stitched_correlation.py``.

    The amplitude match in :func:`merge_correlations` (``match_overlap=True``)
    rescales each measurement's software curve to *its own* hardware overlap, so
    the stitched merge is self-consistent per angle and is suitable for the
    cross-angle coupled inversion :func:`corenn.corenn_multi_angle`.
    """
    base, reps = C._load_hardware_replicates(
        measurement_dir, correlation_file=correlation_file)
    sigma_hw_rep = C.empirical_sigma(reps) if len(reps) >= 2 else None
    duration_hw = base.duration if base.duration else float(base.tau[-1])
    sigma_hw = hardware_sigma(base.tau, base.g2m, base.beta, duration_hw,
                              tau_distort=tau_distort, replicate_sigma=sigma_hw_rep)

    traces = load_measurement_traces(measurement_dir)
    if not traces:
        raise ValueError(f"No count traces under {measurement_dir}.")
    tau_st, g_st, n_pairs, stitched = correlate_stitched_traces(
        traces, cross=cross, per_octave=per_octave, max_frac=max_frac,
        min_pairs=min_pairs, lag_min=software_lag_min)
    duration_st = float(stitched["time"][-1] + stitched["dt"])
    sigma_st = software_sigma(tau_st, g_st, base.beta, duration_st,
                              float(stitched["dt"]), n_rep=1)

    if overlap is None:
        overlap = (max(5 * float(stitched["dt"]), base.tau.min()),
                   min(5 * tau_distort, base.tau.max(), tau_st.max()))
    merged = merge_correlations(
        base.tau, base.g2m, sigma_hw, tau_st, g_st, sigma_st,
        match_overlap=True, overlap=overlap)

    ds = C.DLSDataset(
        tau=merged.tau, g2m=merged.g2m, q=base.q, beta=base.beta,
        sigma=merged.sigma, temperature=base.temperature,
        viscosity=base.viscosity, n_solvent=base.n_solvent,
        duration=duration_st, count_rate=base.count_rate,
        metadata={**base.metadata, "hybrid": True, "stitched": True,
                  "n_pairs_max": int(np.nanmax(n_pairs)) if len(n_pairs) else 0,
                  "scale_sw": merged.scale_sw,
                  "software_lag_min_s": software_lag_min},
    )
    return ds, merged
