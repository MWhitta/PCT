"""
demo_microsecond_wiggles.py -- Quantify reproducible microsecond-lag structure
in MAPCS correlograms across angle and polarization.

The diagnostic fits the slower population with nonlinear CORENN outside the
microsecond band, predicts that slow model back into the band, and summarizes
the residual relative to replicate scatter. A true small/fast population should
produce q- and polarization-dependent residual metrics; a fixed electronic
artifact should be nearly invariant in lag shape and amplitude across channels.

Run:  python demo_microsecond_wiggles.py [path_to_temperature_dir]
"""

import csv
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corenn as C

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(ROOT, "data", "SISV_1%", "11.5C")
OUT_FIG = os.path.join(ROOT, "microsecond_wiggles_demo.png")
OUT_CSV = os.path.join(ROOT, "reports", "microsecond_wiggles_metrics.csv")

LAG_BAND = (8e-7, 1e-4)
BASELINE_LAG_MIN = 3e-4
INVERT_KWARGS = {
    "n_basis": 22,
    "n_R": 160,
    "n_lambda": 16,
    "lambda_range": (1e-7, 1e3),
}
POL_ORDER = ("VH", "VV", "3D")
COLORS = {"VH": "C3", "VV": "C0", "3D": "C2"}
MARKERS = {"VH": "s", "VV": "o", "3D": "^"}


def _safe_replicates(temp_dir, measurement):
    if measurement is None:
        return []
    meas_dir = os.path.join(temp_dir, f"Measurement {int(measurement)}")
    try:
        _, reps = C.load_measurement_replicates(meas_dir)
    except Exception:
        return []
    return reps


def load_diagnostics(temp_dir):
    exp = C.load_mapcs_experiment(temp_dir)
    diagnostics = {pol: [] for pol in POL_ORDER}
    for pol in POL_ORDER:
        for ds in exp.get(pol, []):
            reps = _safe_replicates(temp_dir, ds.metadata.get("measurement"))
            diag = C.short_lag_residual_diagnostic(
                ds,
                replicates=reps,
                lag_band=LAG_BAND,
                baseline_lag_min=BASELINE_LAG_MIN,
                invert_kwargs=INVERT_KWARGS,
            )
            diag["dataset"] = ds
            diag["replicates"] = reps
            diagnostics[pol].append(diag)
    return diagnostics


def _interp_band(diag, grid, values, normalize_beta=False):
    mask = diag["band_mask"]
    tau = np.asarray(diag["tau"])[mask]
    y = np.asarray(values, float)[mask]
    good = np.isfinite(tau) & np.isfinite(y) & (tau > 0)
    if np.count_nonzero(good) < 2:
        return None
    if normalize_beta:
        beta = max(abs(float(diag["dataset"].beta)),
                   abs(float(diag["baseline_result"].beta)), 1e-12)
        y = y / beta
    return np.interp(np.log(grid), np.log(tau[good]), y[good])


def build_template(diagnostics, grid):
    curves = []
    for pol in POL_ORDER:
        for diag in diagnostics[pol]:
            y = _interp_band(diag, grid, diag["wiggle_residual"], normalize_beta=True)
            if y is None:
                continue
            y = y - np.nanmean(y)
            rms = float(np.sqrt(np.nanmean(y * y)))
            if np.isfinite(rms) and rms > 0:
                curves.append(y / rms)
    if not curves:
        return np.zeros_like(grid)
    template = np.nanmedian(np.vstack(curves), axis=0)
    template = template - np.nanmean(template)
    rms = float(np.sqrt(np.nanmean(template * template)))
    return template / rms if rms > 0 else template


def add_template_metrics(diagnostics, grid, template):
    for pol in POL_ORDER:
        for diag in diagnostics[pol]:
            y = _interp_band(diag, grid, diag["wiggle_residual"], normalize_beta=True)
            s = _interp_band(diag, grid, diag["sigma"], normalize_beta=True)
            if y is None or s is None:
                diag["template_amp_over_beta"] = float("nan")
                diag["template_z"] = float("nan")
                continue
            y = y - np.nanmean(y)
            good = np.isfinite(y) & np.isfinite(s) & (s > 0) & np.isfinite(template)
            if np.count_nonzero(good) < 3:
                diag["template_amp_over_beta"] = float("nan")
                diag["template_z"] = float("nan")
                continue
            w = 1.0 / np.maximum(s[good], 1e-12) ** 2
            t = template[good]
            den = float(np.sum(w * t * t))
            if den <= 0:
                amp = se = z = float("nan")
            else:
                amp = float(np.sum(w * y[good] * t) / den)
                se = float(1.0 / np.sqrt(den))
                z = amp / se if se > 0 else float("nan")
            diag["template_amp_over_beta"] = amp
            diag["template_z"] = z


def closest_angle(diagnostics, target=85.0):
    out = {}
    for pol in POL_ORDER:
        items = diagnostics[pol]
        if items:
            out[pol] = min(items, key=lambda d: abs(float(d["angle_deg"]) - target))
    return out


def write_metrics(temp_dir, diagnostics):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = [
        "temperature_dir", "polarization", "measurement", "angle_deg", "q_1_per_m",
        "n_band", "sigma_source", "slow_radius_floor_nm", "baseline_lag_min_s",
        "rms_residual_percent_beta", "rms_z_log", "mean_residual_percent_beta",
        "peak_to_peak_percent_beta", "max_abs_z", "lag_at_max_abs_us",
        "wiggle_rms_percent_beta", "wiggle_rms_z_log",
        "wiggle_peak_to_peak_percent_beta", "wiggle_max_abs_z",
        "template_amp_percent_beta", "template_z",
    ]
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for pol in POL_ORDER:
            for d in diagnostics[pol]:
                writer.writerow({
                    "temperature_dir": os.path.relpath(temp_dir, ROOT),
                    "polarization": pol,
                    "measurement": d["measurement"],
                    "angle_deg": d["angle_deg"],
                    "q_1_per_m": d["q"],
                    "n_band": d["n_band"],
                    "sigma_source": d["sigma_source"],
                    "slow_radius_floor_nm": 1e9 * d["slow_radius_floor_m"],
                    "baseline_lag_min_s": d["baseline_lag_min"],
                    "rms_residual_percent_beta": 100.0 * d["rms_residual_over_beta"],
                    "rms_z_log": d["rms_z_log"],
                    "mean_residual_percent_beta": 100.0 * (
                        d["mean_residual_log"]
                        / max(abs(float(d["dataset"].beta)),
                              abs(float(d["baseline_result"].beta)), 1e-12)
                    ),
                    "peak_to_peak_percent_beta": 100.0 * d["peak_to_peak_over_beta"],
                    "max_abs_z": d["max_abs_z"],
                    "lag_at_max_abs_us": 1e6 * d["lag_at_max_abs_s"],
                    "wiggle_rms_percent_beta": 100.0 * d["wiggle_rms_over_beta"],
                    "wiggle_rms_z_log": d["wiggle_rms_z_log"],
                    "wiggle_peak_to_peak_percent_beta": 100.0 * d["wiggle_peak_to_peak_over_beta"],
                    "wiggle_max_abs_z": d["wiggle_max_abs_z"],
                    "template_amp_percent_beta": 100.0 * d["template_amp_over_beta"],
                    "template_z": d["template_z"],
                })


def plot_diagnostics(temp_dir, diagnostics, grid, template):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    axA, axB, axC, axD, axE, axF = axes.ravel()
    reps = closest_angle(diagnostics)
    label = os.path.relpath(temp_dir, ROOT)
    lag_lo, lag_hi = LAG_BAND

    for pol, diag in reps.items():
        color = COLORS[pol]
        tau = diag["tau"]
        window = (tau >= lag_lo / 2.0) & (tau <= 2e-3)
        axA.semilogx(tau[window], diag["g2m"][window], color=color, lw=1.5,
                     label=f"{pol} data, {diag['angle_deg']:.0f} deg")
        axA.semilogx(tau[window], diag["g2_model"][window], color=color, lw=1.1,
                     ls="--")

        mask = diag["band_mask"]
        beta = max(abs(float(diag["dataset"].beta)),
                   abs(float(diag["baseline_result"].beta)), 1e-12)
        y = 100.0 * diag["residual"][mask] / beta
        trend = 100.0 * diag["residual_trend"][mask] / beta
        s = 100.0 * diag["sigma"][mask] / beta
        axB.semilogx(tau[mask], y, color=color, lw=1.6, label=pol)
        axB.semilogx(tau[mask], trend, color=color, lw=1.0, ls=":")
        axB.fill_between(tau[mask], y - s, y + s, color=color, alpha=0.12, lw=0)

    axA.axvspan(lag_lo, lag_hi, color="0.86", zorder=-2)
    axA.set_xlabel("lag time [s]")
    axA.set_ylabel(r"$g_2(\tau)-1$")
    axA.set_title("(A) Raw correlograms and slow CORENN baseline")
    axA.legend(fontsize=8)

    axB.axhline(0.0, color="0.25", lw=0.8)
    axB.set_xlabel("lag time [s]")
    axB.set_ylabel("residual [% beta]")
    axB.set_title("(B) Microsecond residuals at representative angle")
    axB.legend(fontsize=8)

    axC.semilogx(grid, template, color="0.15", lw=1.8)
    axC.axhline(0.0, color="0.55", lw=0.8)
    axC.set_xlabel("lag time [s]")
    axC.set_ylabel("unit-RMS detrended shape")
    axC.set_title("(C) Matched wiggle template")

    for pol in POL_ORDER:
        ds = diagnostics[pol]
        if not ds:
            continue
        angles = np.array([float(d["angle_deg"]) for d in ds])
        order = np.argsort(angles)
        color = COLORS[pol]
        marker = MARKERS[pol]
        axD.plot(angles[order], [ds[i]["rms_z_log"] for i in order],
                 marker + "-", color=color, ms=4, lw=1.2, label=pol)
        axE.plot(angles[order], [100.0 * ds[i]["wiggle_rms_over_beta"] for i in order],
                 marker + "-", color=color, ms=4, lw=1.2, label=pol)
        axF.plot(angles[order], [100.0 * ds[i]["template_amp_over_beta"] for i in order],
                 marker + "-", color=color, ms=4, lw=1.2, label=pol)

    axD.axhline(3.0, color="0.35", ls="--", lw=0.9, label="3 sigma")
    axD.set_xlabel("scattering angle [deg]")
    axD.set_ylabel("band RMS residual / sigma")
    axD.set_title("(D) Significance of short-lag mismatch")
    axD.legend(fontsize=8)

    axE.set_xlabel("scattering angle [deg]")
    axE.set_ylabel("detrended wiggle RMS [% beta]")
    axE.set_title("(E) Oscillatory wiggle amplitude")
    axE.legend(fontsize=8)

    axF.axhline(0.0, color="0.35", lw=0.8)
    axF.set_xlabel("scattering angle [deg]")
    axF.set_ylabel("wiggle template amplitude [% beta]")
    axF.set_title("(F) Consistent wiggle-template amplitude")
    axF.legend(fontsize=8)

    fig.suptitle(
        f"Microsecond-lag residual diagnostic: {label}  "
        f"band={lag_lo:.1e}-{lag_hi:.1e} s; slow fit starts at {BASELINE_LAG_MIN:.1e} s",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=120)


def print_summary(diagnostics):
    print("=== Microsecond residual summary ===")
    print(f"lag band = {LAG_BAND[0]:.2e} to {LAG_BAND[1]:.2e} s")
    print(f"slow CORENN fit starts at tau >= {BASELINE_LAG_MIN:.2e} s")
    for pol in POL_ORDER:
        ds = diagnostics[pol]
        if not ds:
            continue
        rms_pct = np.array([100.0 * d["rms_residual_over_beta"] for d in ds])
        wiggle_pct = np.array([100.0 * d["wiggle_rms_over_beta"] for d in ds])
        z = np.array([d["rms_z_log"] for d in ds])
        amp = np.array([100.0 * d["template_amp_over_beta"] for d in ds])
        tz = np.array([d["template_z"] for d in ds])
        print(
            f"{pol:>2}: n={len(ds):2d}  "
            f"median RMS={np.nanmedian(rms_pct):5.2f}% beta  "
            f"median wiggle={np.nanmedian(wiggle_pct):5.2f}% beta  "
            f"median RMS_z={np.nanmedian(z):5.2f}  "
            f"median template={np.nanmedian(amp):+5.2f}% beta  "
            f"max |template_z|={np.nanmax(np.abs(tz)):5.1f}"
        )


def main(temp_dir):
    if not os.path.isdir(temp_dir):
        raise SystemExit(f"Data directory not found: {temp_dir}")
    diagnostics = load_diagnostics(temp_dir)
    grid = np.geomspace(LAG_BAND[0], LAG_BAND[1], 80)
    template = build_template(diagnostics, grid)
    add_template_metrics(diagnostics, grid, template)
    write_metrics(temp_dir, diagnostics)
    plot_diagnostics(temp_dir, diagnostics, grid, template)
    print(f"Saved {os.path.relpath(OUT_FIG, ROOT)}")
    print(f"Saved {os.path.relpath(OUT_CSV, ROOT)}")
    print_summary(diagnostics)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
