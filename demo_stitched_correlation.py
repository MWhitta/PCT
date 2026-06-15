"""
demo_stitched_correlation.py -- Diagnose per-repeat pooled vs gap-aware stitched
software correlation, and compare the resulting CORENN inversions.

Run:
    python demo_stitched_correlation.py [measurement_dir]

The default measurement is LiTFSI 20m / 22C / Measurement 1, which has five
sequential four-minute repetitions with count traces.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corenn as C
import hybrid_correlator as H


SOFTWARE_LAG_MIN = 0.1

DEFAULT_MEAS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "LiTFSI",
    "20m",
    "22C",
    "Measurement 1",
)


def _scaled_individual_software_curves(traces, scale, per_octave=8, max_frac=0.2,
                                       lag_min=SOFTWARE_LAG_MIN):
    curves = []
    for tr in traces:
        t = np.asarray(tr["time"], float)
        if t.size < 8 or "cr_a" not in tr:
            continue
        dt = float(np.median(np.diff(t)))
        b = np.asarray(tr["cr_b"], float) if "cr_b" in tr else None
        tau, g2m = H.software_correlate(
            np.asarray(tr["cr_a"], float), dt, b, per_octave=per_octave,
            max_frac=max_frac,
        )
        keep = tau >= lag_min
        if not np.any(keep):
            continue
        tau, g2m = tau[keep], g2m[keep]
        curves.append((tau, scale * g2m))
    return curves


def _make_stitched_hybrid(meas_dir, base, reps, traces, tau_distort=1.0,
                          per_octave=8, max_frac=0.2, min_pairs=100,
                          software_lag_min=SOFTWARE_LAG_MIN):
    sigma_hw_rep = C.empirical_sigma(reps) if len(reps) >= 2 else None
    duration_hw = base.duration if base.duration else float(base.tau[-1])
    sigma_hw = H.hardware_sigma(
        base.tau, base.g2m, base.beta, duration_hw,
        tau_distort=tau_distort, replicate_sigma=sigma_hw_rep,
    )
    tau_st, g_st, n_pairs, stitched = H.correlate_stitched_traces(
        traces, per_octave=per_octave, max_frac=max_frac, min_pairs=min_pairs,
        lag_min=software_lag_min,
    )
    duration_st = float(stitched["time"][-1] + stitched["dt"])
    sigma_st = H.software_sigma(
        tau_st, g_st, base.beta, duration_st, float(stitched["dt"]), n_rep=1,
    )
    overlap = (
        max(5 * float(stitched["dt"]), base.tau.min()),
        min(5 * tau_distort, base.tau.max(), tau_st.max()),
    )
    merged = H.merge_correlations(
        base.tau, base.g2m, sigma_hw, tau_st, g_st, sigma_st,
        match_overlap=True, overlap=overlap,
    )
    ds = C.DLSDataset(
        tau=merged.tau,
        g2m=merged.g2m,
        q=base.q,
        beta=base.beta,
        sigma=merged.sigma,
        temperature=base.temperature,
        viscosity=base.viscosity,
        n_solvent=base.n_solvent,
        duration=duration_st,
        count_rate=base.count_rate,
        metadata={**base.metadata, "hybrid": True, "stitched": True,
                  "scale_sw": merged.scale_sw,
                  "software_lag_min_s": software_lag_min},
    )
    return ds, merged, stitched, n_pairs


def _peak_text(result):
    if not result.peaks:
        return "no peak"
    return ", ".join(
        f"{p['radius_nm']:.0f} nm ({p['rel_amount'] * 100:.0f}%)"
        for p in result.peaks[:3]
    )


def _tail_fraction(result, threshold_nm):
    r_nm = result.R * 1e9
    total = float(np.trapz(result.intensity, result.R))
    if total <= 0:
        return float("nan")
    mask = r_nm >= threshold_nm
    return float(np.trapz(result.intensity[mask], result.R[mask]) / total)


def _main_mode_nm(result, tail_threshold_nm=1.0e4):
    r_nm = result.R * 1e9
    y = result.R * result.intensity
    mask = r_nm < tail_threshold_nm
    if not np.any(mask) or not np.any(np.isfinite(y[mask])):
        return float("nan")
    idx_local = int(np.nanargmax(y[mask]))
    return float(r_nm[mask][idx_local])


def _psd_label(name, result):
    mode_nm = _main_mode_nm(result)
    tail_10um = _tail_fraction(result, 1.0e4)
    return f"{name}: main mode {mode_nm:.0f} nm, >10um {tail_10um * 100:.1f}%"


def main(meas_dir):
    if not os.path.isdir(meas_dir):
        raise SystemExit(f"Measurement directory not found: {meas_dir}")

    base, reps = C.load_measurement_replicates(meas_dir)
    traces = H.load_measurement_traces(meas_dir)
    if not traces:
        raise SystemExit(f"No count traces under {meas_dir}")

    raw_hybrid, raw_merged = H.build_hybrid_dataset(
        meas_dir, software_lag_min=SOFTWARE_LAG_MIN)
    stitched_hybrid, stitched_merged, stitched, n_pairs = _make_stitched_hybrid(
        meas_dir, base, reps, traces,
    )

    raw_sw = raw_merged.components["software"]
    stitched_sw = stitched_merged.components["software"]
    hw = raw_merged.components["hardware"]
    raw_individual = _scaled_individual_software_curves(
        traces, raw_merged.scale_sw,
    )

    radius_range = (C.DEFAULT_CORENN_RADIUS_RANGE[0], 5e-5)
    invert_kwargs = dict(radius_range=radius_range, n_basis=30, n_R=200,
                         fit_baseline=True)
    res_hw = C.corenn_invert(base, replicates=reps, **invert_kwargs)
    res_raw = C.corenn_invert(raw_hybrid, **invert_kwargs)
    res_st = C.corenn_invert(stitched_hybrid, **invert_kwargs)

    # Compare software estimators over their common lag range after both are
    # scaled to the hardware overlap.
    lo = max(raw_sw["tau"].min(), stitched_sw["tau"].min())
    hi = min(raw_sw["tau"].max(), stitched_sw["tau"].max())
    overlap = (raw_sw["tau"] >= lo) & (raw_sw["tau"] <= hi)
    st_at_raw = np.interp(
        np.log(raw_sw["tau"][overlap]), np.log(stitched_sw["tau"]),
        stitched_sw["g2m"],
    )
    sw_delta = st_at_raw - raw_sw["g2m"][overlap]
    sw_rms = float(np.sqrt(np.mean(sw_delta ** 2))) if sw_delta.size else np.nan

    gaps = [s["gap_s"] for s in stitched["segments"][1:]]
    raw_gaps = [s["raw_gap_s"] for s in stitched["segments"][1:]]
    observed = np.isfinite(stitched["cr_a"])
    elapsed = float(stitched["time"][-1] + stitched["dt"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axA, axB, axC, axD = axes.ravel()

    # (A) software correlogram estimators
    axA.plot(hw["tau"], hw["g2m"], "k.", ms=2.5, alpha=0.35,
             label="hardware")
    for i, (tau_i, g_i) in enumerate(raw_individual):
        axA.plot(tau_i, g_i, color="0.7", lw=0.8, alpha=0.45,
                 label="raw repeat software" if i == 0 else None)
    axA.plot(raw_sw["tau"], raw_sw["g2m"], "C0o", ms=4,
             label="raw repeats pooled")
    axA.plot(stitched_sw["tau"], stitched_sw["g2m"], "C3s", ms=3.5,
             label="gap-aware stitched")
    axA.set_xscale("log")
    axA.set_xlabel(r"lag $\tau$ [s]")
    axA.set_ylabel(r"$g_2 - 1$")
    axA.set_title("(A) Software correlograms, scaled to hardware overlap")
    axA.axvline(SOFTWARE_LAG_MIN, color="0.35", ls=":", lw=1.0,
                label="software cutoff")
    axA.legend(fontsize=8)

    # (B) merged hybrid curves and CORENN overlays
    axB.plot(raw_hybrid.tau, raw_hybrid.g2m, "C0.", ms=3,
             label="hybrid: raw pooled")
    axB.plot(stitched_hybrid.tau, stitched_hybrid.g2m, "C3.", ms=3,
             label="hybrid: stitched")
    axB.plot(res_raw.tau, res_raw.g2_model, "C0-", lw=1.3,
             label="CORENN model raw")
    axB.plot(res_st.tau, res_st.g2_model, "C3-", lw=1.3,
             label="CORENN model stitched")
    axB.set_xscale("log")
    axB.set_xlabel(r"lag $\tau$ [s]")
    axB.set_ylabel(r"$g_2 - 1$")
    axB.set_title("(B) Merged curves used for inversion")
    axB.legend(fontsize=8)

    # (C) stitched timeline + estimator difference
    axC.axhline(0.0, color="0.3", lw=0.8)
    axC.plot(raw_sw["tau"][overlap], sw_delta, "C4o-", ms=3,
             label=f"stitched - raw pooled; RMS={sw_rms:.2g}")
    axC.set_xscale("log")
    axC.set_xlabel(r"lag $\tau$ [s]")
    axC.set_ylabel(r"$\Delta(g_2 - 1)$")
    axC2 = axC.twinx()
    axC2.plot(stitched_sw["tau"], n_pairs, "-", color="0.55", lw=1.2,
              label="stitched valid pairs")
    axC2.set_ylabel("valid endpoint pairs")
    lines1, labels1 = axC.get_legend_handles_labels()
    lines2, labels2 = axC2.get_legend_handles_labels()
    axC.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    axC.set_title("(C) Estimator difference and stitched pair count")

    # (D) CORENN inversions
    per_decade = lambda r: r.R * r.intensity
    axD.plot(res_hw.R * 1e9, per_decade(res_hw), "-", color="0.25", lw=1.6,
             label=_psd_label("hardware only", res_hw))
    axD.plot(res_raw.R * 1e9, per_decade(res_raw), "C0-", lw=2,
             label=_psd_label("raw-pooled hybrid", res_raw))
    axD.plot(res_st.R * 1e9, per_decade(res_st), "C3-", lw=2,
             label=_psd_label("stitched hybrid", res_st))
    axD.set_xscale("log")
    axD.set_xlabel("radius [nm]")
    axD.set_ylabel(r"$R\,I(R)$ (intensity / decade)")
    axD.set_title("(D) CORENN PSD comparison")
    axD.legend(fontsize=8)

    fig.suptitle(
        f"Raw pooled vs gap-aware stitched software correlation\n{meas_dir}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("stitched_correlation_demo.png", dpi=130)

    print("Saved stitched_correlation_demo.png")
    print(f"measurement: {meas_dir}")
    print(f"repeats: {len(traces)}")
    print(f"gaps_s: {gaps}")
    print(f"raw_gaps_s: {raw_gaps}")
    print(f"stitched elapsed: {elapsed:.1f}s; observed bins: {observed.sum()}; "
          f"total bins: {observed.size}; missing bins: {np.count_nonzero(~observed)}")
    print(f"raw pooled software tau max: {raw_sw['tau'].max():.1f}s")
    print(f"stitched software tau max: {stitched_sw['tau'].max():.1f}s")
    print(f"software lag min used: {SOFTWARE_LAG_MIN:.3f}s")
    print(f"software overlap RMS difference: {sw_rms:.4g}")
    print("\n--- Hardware only ---")
    print(res_hw.summary())
    print(f"main mode <10 um: {_main_mode_nm(res_hw):.1f} nm")
    print(f"tail mass >10 um: {_tail_fraction(res_hw, 1.0e4) * 100:.2f}%")
    print("\n--- Hybrid: raw repeats pooled ---")
    print(res_raw.summary())
    print(f"main mode <10 um: {_main_mode_nm(res_raw):.1f} nm")
    print(f"tail mass >10 um: {_tail_fraction(res_raw, 1.0e4) * 100:.2f}%")
    print("\n--- Hybrid: gap-aware stitched repeats ---")
    print(res_st.summary())
    print(f"main mode <10 um: {_main_mode_nm(res_st):.1f} nm")
    print(f"tail mass >10 um: {_tail_fraction(res_st, 1.0e4) * 100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MEAS)
