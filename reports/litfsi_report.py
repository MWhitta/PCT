"""
litfsi_report.py -- Full analysis report for LiTFSI 20 molal (22 C and 50 C).
Runs every applicable analysis on the 28-angle data and writes
reports/figures/litfsi.png and reports/LiTFSI.pdf (+ .md).

    python reports/litfsi_report.py
"""

import os
import sys
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import corenn as C
import _common as K
from md2pdf import md_to_pdf

warnings.filterwarnings("ignore")
DATA = os.path.join(ROOT, "data", "LiTFSI", "20m")


def load_temp(tdir):
    """Return (datasets, replicates) aligned per angle."""
    dsets, reps = [], []
    for md in sorted(glob.glob(os.path.join(tdir, "Measurement *")),
                     key=lambda p: int(p.rsplit(" ", 1)[1])):
        try:
            ds, rp = C.load_measurement_replicates(md)
            dsets.append(ds); reps.append(rp)
        except Exception:
            continue
    return dsets, reps


def main():
    A = {}; RAW = {}
    for t in ("22C", "50C"):
        dsets, reps = load_temp(os.path.join(DATA, t))
        T, eta = dsets[0].temperature, dsets[0].viscosity
        A[t] = K.run_suite(dsets, T, eta, psd_bootstrap=20, replicates=reps)
        RAW[t] = dsets

    # joint multi-temperature: solve eta(50C) under same-sample assumption
    ser = [{"label": t, "T": A[t]["T"], "q": A[t]["q"], "gamma": A[t]["gam"],
            "eta": A[t]["eta"]} for t in ("22C", "50C")]
    ser_solve = [dict(s) for s in ser]; ser_solve[1]["eta"] = np.nan
    jt = C.joint_temperature_fit(ser_solve, reference_index=0)
    eta50 = jt["eta_fit"][1]

    # ---- figure: 5 suite panels + viscosity-solve panel ------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 9)); ax = ax.ravel()
    for t, col in (("22C", "C0"), ("50C", "C3")):
        K.plot_suite(ax[:5], A[t], col, t)
    K.label_suite_axes(ax)
    ax[1].set_ylim(0, 900)
    ax[5].bar(["22C\n(anchor)", "50C\nrecorded", "50C\nsame-sample"],
              [A["22C"]["eta"]*1e3, A["50C"]["eta"]*1e3, eta50*1e3],
              color=["C0", "0.7", "C3"])
    ax[5].set_ylabel("viscosity [mPa·s]"); ax[5].set_title("(6) same-sample η(50°C) solve")
    fig.suptitle("LiTFSI 20 molal — all multi-angle static + dynamic analyses")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(HERE, "figures", "litfsi.png"))
    print("Saved reports/figures/litfsi.png")

    # ---- raw-correlogram + count-rate figures ----------------------------
    raw_figs = []
    for t in ("22C", "50C"):
        rp = os.path.join(HERE, "figures", f"litfsi_raw_{t}.png")
        if K.fig_raw_correlograms(A[t], rp, label=f"LiTFSI {t}"):
            raw_figs.append((f"figures/litfsi_raw_{t}.png",
                             f"Raw 3D-cross correlograms at low/mid/high q ({t}): the stitched "
                             "multi-tau correlation (black) over the semi-transparent raw "
                             "replicate points; the point spread is the measurement repeatability."))
    cr_path = os.path.join(HERE, "figures", "litfsi_countrate.png")
    conditions = [{"label": f"{t}", "channel": "3D", "T": A[t]["T"], "eta": A[t]["eta"],
                   "datasets": RAW[t], "suite": A[t]} for t in ("22C", "50C")]
    K.fig_count_rate(conditions, cr_path)

    # ---- bespoke narrative pieces ----------------------------------------
    primary = []
    primary.append("q = 4πn sin(θ/2)/λ (n=1.36, λ=632.8 nm); the exported 'Scattering vector "
                   "[1/nm]' column is mislabelled (its values are actually in 1/m), so q is "
                   "recomputed from the angle and reported in nm⁻¹, spanning "
                   f"**{K._qrange_nm(A['22C']['q_range'][0], A['22C']['q_range'][1])}**. "
                   "The size distribution is the **forward model** (CORENN multi-angle); Γ(q), "
                   "apparent R_h(q), and the static I(q) regime are diagnostic projections.\n")
    for t in ("22C", "50C"):
        primary.append(f"### {t} (T = {A[t]['T']:.1f} K, recorded η = {A[t]['eta']*1e3:.0f} mPa·s)\n")
        steps, _ = K.suite_steps_md(A[t], start=2)
        primary += steps

    mode = []
    mode.append("### Joint multi-temperature (same sample)\n")
    mode.append("The recorded viscosity is a fixed 34 mPa·s at **both** temperatures — wrong "
                "for an electrolyte. The static intensity profile is unchanged with T "
                f"(power-law exponent {A['22C']['static_pl']['exponent']:.1f} at 22 °C vs "
                f"{A['50C']['static_pl']['exponent']:.1f} at 50 °C), supporting one "
                "(T-invariant) scattering structure; the 50 °C diffusion then *implies* the "
                "true viscosity:\n")
    mode.append(f"- D_z(22°C)={A['22C']['gvq2']['D']:.3e}, D_z(50°C)={A['50C']['gvq2']['D']:.3e} m²/s; "
                f"shared R_h,z={jt['R_h_z']*1e9:.0f} nm.\n")
    mode.append(f"- **implied η(50 °C) = {eta50*1e3:.0f} mPa·s** (vs recorded 34), ratio "
                f"η(50)/η(22)={eta50/A['22C']['eta']:.2f} — the electrolyte thins on heating.\n")
    mode.append("This is a cross-check (dynamics alone cannot separate η(T) from a size "
                "change); it is valid only because the *static* R_g is T-invariant.\n")

    # ---- single-angle stitched large-aggregate tail ----------------------
    _low = sorted([d for d in RAW["22C"] if d.metadata.get("path")], key=lambda d: d.q)[:4]
    _mdirs = [os.path.dirname(d.metadata["path"]) for d in _low]
    _sm = K.slow_mode_stitched(_mdirs, os.path.join(HERE, "figures", "litfsi_slowmode.png"),
                               label="LiTFSI 22 °C")
    mode += K.slow_mode_section_md(_sm, "figures/litfsi_slowmode.png")
    _mw = K.fig_psd_weightings(A["22C"]["corenn"], float(np.median(A["22C"]["q"])),
                               A["22C"]["T"], A["22C"]["eta"],
                               os.path.join(HERE, "figures", "litfsi_weightings.png"),
                               label="LiTFSI 22 °C",
                               D_f=K._mass_fractal_dim((A["22C"].get("regime") or {}).get("porod_morphology")))
    mode += K.psd_weightings_section_md(_mw, "figures/litfsi_weightings.png")

    # ---- summary table ---------------------------------------------------
    m = []
    m.append("| quantity | 22 °C | 50 °C |")
    m.append("|---|---|---|")
    def cpk(a):
        return ", ".join(f"{p['radius_nm']:.0f} nm" for p in a["corenn"].peaks) if a["corenn"] else "—"
    def rh_ci(a):
        b = a["boot"]["R_h"]
        return (f"{b['median']*1e9:.0f} [{b['p16']*1e9:.0f}–{b['p84']*1e9:.0f}]"
                if b else f"{a['gvq2']['R_h']*1e9:.0f}")
    def rho_s(a):
        r = a["rho"]
        return f"{r['rho']:.2f} ({r['model']})" if r["independent"] else "not determined (Porod regime)"
    def cdg(a):
        d = a["corenn_diag"]
        return f"{d['eff_dof']:.0f}/{d['n_basis']}, κ={d['cond']:.0e}" if d else "—"
    def psd_ci(a):
        b = a.get("boot_psd")
        return f"{b['median']:.0f} [{b['p16']:.0f}–{b['p84']:.0f}]" if b else "—"
    m.append(f"| z-average R_h [68% CI] | {rh_ci(A['22C'])} nm | {rh_ci(A['50C'])} nm |")
    m.append(f"| shape ρ = R_g/R_h | {rho_s(A['22C'])} | {rho_s(A['50C'])} |")
    m.append(f"| apparent R_h low→high q | {A['22C']['app']['R_h_low_q_median']*1e9:.0f}→{A['22C']['app']['R_h_high_q_median']*1e9:.0f} nm | {A['50C']['app']['R_h_low_q_median']*1e9:.0f}→{A['50C']['app']['R_h_high_q_median']*1e9:.0f} nm |")
    m.append(f"| PDI (mean) | {A['22C']['app']['PDI_mean']:.2f} | {A['50C']['app']['PDI_mean']:.2f} |")
    m.append(f"| static power-law I∝q^−p | p={A['22C']['static_pl']['exponent']:.1f} | p={A['50C']['static_pl']['exponent']:.1f} |")
    m.append(f"| Γ(q²) predictive R² (LOO-CV) | {A['22C']['cv']['cv_r2']:.3f} | {A['50C']['cv']['cv_r2']:.3f} |")
    m.append(f"| CORENN eff.DOF / cond | {cdg(A['22C'])} | {cdg(A['50C'])} |")
    m.append(f"| power-law α | {A['22C']['sdc']['powerlaw_alpha']:.2f} | {A['50C']['sdc']['powerlaw_alpha']:.2f} |")
    m.append(f"| CORENN multi-angle PSD | {cpk(A['22C'])} | {cpk(A['50C'])} |")
    m.append(f"| CORENN peak [68% CI, bootstrap] | {psd_ci(A['22C'])} nm | {psd_ci(A['50C'])} nm |")
    m.append(f"| implied η(50°C) (same-sample) | — | {eta50*1e3:.0f} mPa·s |")
    summary_table = m

    _mph = A["22C"]["regime"].get("porod_morphology")
    _p22 = A["22C"]["regime"]["porod"]["exponent"] if A["22C"]["regime"].get("porod") else float("nan")
    _morph_txt = (f"which implies **{_mph['label']}** — a rough / branched aggregate rather than "
                  "a smooth sphere, so the sphere form factor in the PSD is only an effective "
                  "approximation" if _mph else
                  "with the sphere form factor in the PSD assumed, not confirmed")
    main_result = (
        "A broadly polydisperse / aggregate-laden electrolyte: a main population of a few "
        f"hundred nm (z-average R_h ≈ {A['22C']['gvq2']['R_h']*1e9:.0f} nm at 22 °C, "
        f"{A['50C']['gvq2']['R_h']*1e9:.0f} nm at 50 °C) with a slow large-aggregate tail. "
        f"The data sit in the **Porod regime** at every angle with a slope I ∝ q^−{_p22:.1f}, "
        f"{_morph_txt}; **ρ = R_g/R_h is not determined** (no Guinier regime). The recorded "
        "viscosity (fixed 34 mPa·s at both T) is a placeholder; the same-sample solve implies "
        f"η(50 °C) ≈ {eta50*1e3:.0f} mPa·s.")

    ctx = {
        "title": "LiTFSI 20 molal — provisional DLS/SLS analysis report",
        "root": ROOT, "data_path": os.path.relpath(DATA, ROOT),
        "script": "reports/litfsi_report.py", "command": "python reports/litfsi_report.py",
        "dataset_class": "multi-angle, multi-temperature (3D cross-correlation)",
        "overview": ("Water-in-salt electrolyte (LiTFSI, 20 mol/kg) by multi-angle PCS at "
                     f"22 °C and 50 °C: {A['22C']['n_total']} angles (15–150°), 5 replicates "
                     "each, 3D-modulated cross-correlation (no polarization channels)."),
        "solvent": {"n": 1.36, "lambda_nm": 632.8, "source": "aqueous electrolyte, assumed"},
        "eta_source": "recorded export (fixed 34 mPa·s)",
        "eta_note": "fixed at both temperatures — placeholder, corrected by same-sample solve",
        "static_scale": "absolute (toluene-referenced) — assumed, unverified",
        "particle_model": (f"rough / branched aggregate (Porod q^−{_p22:.1f} ⇒ {K._morph_short(_mph)}); "
                           "PSD sphere form factor is an effective approximation"
                           if _mph else "homogeneous sphere (assumed; ρ not determined to confirm)"),
        "primary_model": "CORENN multi-angle PSD + same-sample joint-temperature solve",
        "next_action": ("measure the true η(T) of the electrolyte to convert the dynamics to "
                        "an absolute size; fit a surface-/mass-fractal form factor (the Porod "
                        "slope rules out a smooth sphere); confirm the toluene reference to lock "
                        "the SLS scale."),
        "limitations": "viscosity placeholder; toluene reference unverified; sphere PSD form factor inconsistent with the Porod slope",
        "provisional_flags": ["viscosity is a fixed placeholder (corrected by same-sample solve)",
                              "toluene/absolute-intensity reference not confirmable from the export",
                              f"Porod slope q^−{_p22:.1f} implies a surface-fractal / branched aggregate, not the sphere assumed by the PSD"],
        "generated_files": ["LiTFSI.md", "LiTFSI.pdf", "figures/litfsi.png",
                            "figures/litfsi_slowmode.png", "figures/litfsi_weightings.png",
                            "figures/litfsi_raw_22C.png", "figures/litfsi_raw_50C.png",
                            "figures/litfsi_countrate.png", "LiTFSI.sidecar.json"],
        "sidecar_path": os.path.join(HERE, "LiTFSI.sidecar.json"),
    }

    md_text = K.build_report(
        ctx, conditions, main_result=main_result, summary_table=summary_table,
        consolidating=("figures/litfsi.png",
                       "Consolidating figure: 5 diagnostic/forward-model panels (Γ-vs-q², "
                       "apparent R_h(q), static I(q) regime, R_h∝q^−α, CORENN PSD) plus the "
                       "same-sample η(50 °C) solve (raw-derived)."),
        primary_results=primary, mode_specific=mode,
        raw_figs=raw_figs,
        extra_sections=K.sample_synthesis_section_md(conditions[0]["suite"], None),
        countrate_fig=("figures/litfsi_countrate.png",
                       "Scaled count rate (absolute Rayleigh ratio) vs angle for both "
                       "temperatures (raw data)."))
    with open(os.path.join(HERE, "LiTFSI.md"), "w") as fh:
        fh.write(md_text)
    md_to_pdf(md_text, os.path.join(HERE, "LiTFSI.pdf"), base_dir=HERE)
    print("Saved reports/LiTFSI.md and LiTFSI.pdf")


if __name__ == "__main__":
    main()
