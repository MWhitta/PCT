"""
sisv_report.py -- Full analysis report for SISV_1% (polymer in THF/DOX 7:3)
across 7 temperatures, each a MAPCS experiment (VH/VV/3D, 27 angles).  Runs the
complete static+dynamic suite on the VV channel at every temperature plus the
depolarized (VV/VH) and joint-temperature analyses.  Writes
reports/figures/sisv.png and reports/SISV.pdf (+ .md).

    python reports/sisv_report.py
"""

import os
import sys
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
DATA = os.path.join(ROOT, "data", "SISV_1%")


def main():
    series = C.load_temperature_series(DATA)
    rows = []
    for Tc, exp in series.items():
        vv, vh = exp["VV"], exp["VH"]
        if not vv:
            continue
        T = vv[0].temperature or (Tc + 273.15)
        eta = vv[0].viscosity
        suite = K.run_suite(vv, T, eta,
                            replicates=[d.metadata.get("replicates") for d in vv])
        dep = C.depolarized_diffusion(vv, vh, temperature=T, viscosity=eta)
        I_vv = float(np.nanmedian([d.metadata.get("scaled_count_rate", np.nan) for d in vv]))
        rows.append({"Tc": Tc, "T": T, "eta": eta, "suite": suite, "dep": dep,
                     "I_vv": I_vv, "vv": vv, "vh": vh, "d3": exp.get("3D", [])})

    rep = rows[0]   # representative (lowest) temperature for the detailed suite
    rep["suite"]["boot_psd"] = K.bootstrap_psd_peak(rep["suite"], n_boot=20)
    Tc = np.array([r["Tc"] for r in rows])
    Rh = np.array([r["suite"]["gvq2"]["R_h"] for r in rows])*1e9
    PDI = np.array([r["suite"]["app"]["PDI_mean"] for r in rows])
    Ivv = np.array([r["I_vv"] for r in rows])
    depol = np.array([r["dep"]["depol_mean"] for r in rows])*100

    # same-sample implied eta(T)
    ser = [{"label": f"{r['Tc']}C", "T": r["T"], "q": r["suite"]["q"],
            "gamma": r["suite"]["gam"], "eta": None} for r in rows]
    ser[0]["eta"] = rows[0]["eta"]
    jt = C.joint_temperature_fit(ser, reference_index=0)
    eta_rel = jt["eta_fit"] / jt["eta_fit"][0]

    # ---- figure: representative-T suite (top) + kinetic trends (bottom) ---
    fig = plt.figure(figsize=(15, 9))
    axs = [fig.add_subplot(2, 3, i + 1) for i in range(5)]
    K.plot_suite(axs, rep["suite"], "C0", f"{rep['Tc']}°C")
    K.label_suite_axes(axs)
    axk = fig.add_subplot(2, 3, 6)
    axk.plot(Tc, Rh, "o-", color="C0", label=r"$R_h$ [nm]")
    a2 = axk.twinx(); a2.plot(Tc, PDI, "s--", color="C1"); a2.set_ylabel("PDI", color="C1")
    a3 = axk.twinx(); a3.spines["right"].set_position(("axes", 1.18))
    a3.plot(Tc, depol, "^:", color="C3"); a3.set_ylabel("depol [%]", color="C3")
    axk.set_xlabel("T [°C]"); axk.set_ylabel(r"$R_h$ [nm]", color="C0")
    axk.set_title("(6) temperature series"); axk.legend(fontsize=K.FONT_LEGEND_SMALL, loc="upper right")
    fig.suptitle(f"SISV_1% polymer (THF/DOX) — full suite ({rep['Tc']:.1f} °C) + temperature series")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(HERE, "figures", "sisv.png"))
    print("Saved reports/figures/sisv.png")

    # ---- full analysis-panel suite at EVERY temperature ------------------
    per_temp_figs = []
    for r in rows:
        s = r["suite"]; dep = r["dep"]
        figT, axT = plt.subplots(2, 3, figsize=(15, 9)); axT = axT.ravel()
        K.plot_suite(axT[:5], s, "C0", f"{r['Tc']:.1f}°C")
        K.label_suite_axes(axT)
        # panel 6: VV/VH depolarized decay rates (polarization data)
        axT[5].plot(dep["q_vv"]**2*1e-14, dep["gamma_vv"], "o", ms=4, color="C0", label="VV")
        axT[5].plot(dep["q_vh"]**2*1e-14, dep["gamma_vh"], "s", ms=4, color="C3", label="VH")
        axT[5].set_xlabel(r"$q^2$ [$10^{-4}$ nm$^{-2}$]"); axT[5].set_ylabel(r"$\Gamma$ [s$^{-1}$]")
        axT[5].set_title(f"(6) VV/VH depolarized (depol {dep['depol_mean']*100:.2f}%)")
        axT[5].legend(fontsize=K.FONT_LEGEND_SMALL)
        figT.suptitle(f"SISV_1% — all analysis panels at {r['Tc']:.1f} °C "
                      f"(R_h,z={s['gvq2']['R_h']*1e9:.0f} nm, PDI={s['app']['PDI_mean']:.2f})")
        figT.tight_layout(rect=[0, 0, 1, 0.96])
        fn = f"sisv_T{str(r['Tc']).replace('.', 'p')}.png"
        figT.savefig(os.path.join(HERE, "figures", fn))
        plt.close(figT)
        per_temp_figs.append((r["Tc"], fn))
    print(f"Saved {len(per_temp_figs)} per-temperature analysis figures")

    # ---- raw + count-rate figures ----------------------------------------
    raw_figs = []
    rrp = os.path.join(HERE, "figures", "sisv_raw_rep.png")
    if K.fig_raw_correlograms(rep["suite"], rrp, label=f"SISV VV {rep['Tc']:.1f}°C"):
        raw_figs.append(("figures/sisv_raw_rep.png",
                         f"Raw VV correlograms at low/mid/high q ({rep['Tc']:.1f} °C): the "
                         "stitched multi-tau correlation (black) over the semi-transparent raw "
                         "replicate points."))
    conditions = [{"label": f"{r['Tc']:.1f} °C", "channel": "VV", "T": r["T"],
                   "eta": r["eta"], "datasets": r["vv"], "suite": r["suite"]} for r in rows]
    crp = os.path.join(HERE, "figures", "sisv_countrate.png")
    K.fig_count_rate(conditions, crp)
    hap = os.path.join(HERE, "figures", "sisv_highangle.png")
    has_ha = K.fig_high_angle_diagnostics(rep["vv"], rep["vh"], rep["d3"], rep["suite"],
                                          rep["dep"], hap, label=f"SISV {rep['Tc']:.1f}°C")

    # ---- bespoke narrative -----------------------------------------------
    primary = [f"*Detailed suite shown at the representative temperature "
               f"({rep['Tc']:.1f} °C); all temperatures are tabulated below and in the "
               "per-temperature panels.*\n"]
    steps, _ = K.suite_steps_md(rep["suite"], channel="VV", start=1)
    primary += steps

    iv = int(np.argmin(Ivv))
    mode = []
    mode.append("### Temperature series (kinetics)\n")
    mode.append(f"- **z-average R_h** falls smoothly {Rh[0]:.0f}→{Rh[-1]:.0f} nm from "
                f"{Tc[0]:.1f}→{Tc[-1]:.1f} °C (but see the viscosity caveat below).\n")
    mode.append(f"- **PDI** stays narrow and stable (mean {PDI.mean():.2f}, "
                f"{PDI.min():.2f}–{PDI.max():.2f}) — a well-defined single population.\n")
    mode.append(f"- **static intensity** is non-monotonic: a minimum near {Tc[iv]:.1f} °C "
                f"({Ivv[iv]:.0f} kHz/mW) then a rise above ~22 °C — a candidate "
                "conformational/aggregation transition.\n")
    mode.append("### Polarization (VV/VH depolarized)\n")
    mode.append(f"The depolarization ratio I_VH/I_VV is very small (mean {depol.mean():.3f} %), "
                "i.e. the polymer is nearly optically isotropic. Splitting translational from "
                "rotational diffusion (Γ_VV=q²D_t, Γ_VH=q²D_t+6D_r) gives a small, noisy VH "
                "intercept: for this ~120 nm polymer rotational diffusion (D_r∝1/R³) sits near "
                "the resolution limit, so the depolarization ratio is the robust observable.\n")
    if has_ha:
        mode += K.high_angle_section_md(rep["vv"], rep["suite"], rep["dep"], "figures/sisv_highangle.png")
    _low = sorted([d for d in rep["vv"] if d.metadata.get("path")], key=lambda d: d.q)[:4]
    _sm = K.slow_mode_stitched([os.path.dirname(d.metadata["path"]) for d in _low],
                               os.path.join(HERE, "figures", "sisv_slowmode.png"),
                               label=f"SISV {rep['Tc']:.1f}°C")
    mode += K.slow_mode_section_md(_sm, "figures/sisv_slowmode.png")
    mode += K.psd_weightings_section_md(
        K.fig_psd_weightings(rep["suite"]["corenn"], float(np.median(rep["suite"]["q"])),
                             rep["suite"]["T"], rep["suite"]["eta"],
                             os.path.join(HERE, "figures", "sisv_weightings.png"),
                             label=f"SISV {rep['Tc']:.1f}°C",
                             D_f=K._mass_fractal_dim((rep["suite"].get("regime") or {}).get("porod_morphology"))),
        "figures/sisv_weightings.png")
    mode.append("### Joint multi-temperature (same sample)\n")
    mode.append("The recorded viscosity is a fixed 0.47 mPa·s at every temperature (THF "
                "actually thins on heating). Assuming a T-invariant polymer size, the implied "
                f"viscosity drops **{(1-eta_rel[-1])*100:.0f} %** over {Tc[0]:.0f}→{Tc[-1]:.0f} °C "
                "— consistent with THF — so the apparent R_h decline is mostly the "
                "fixed-viscosity artifact, not real shrinkage.\n")
    mode.append("### All analysis panels, at every temperature\n")
    mode.append("The complete static+dynamic suite (Γ-vs-q², apparent R_h(q), regime-aware "
                "static I(q), R_h∝q^−α, CORENN PSD, VV/VH depolarized) at each temperature:\n")
    for Tc_i, fn in per_temp_figs:
        mode.append(f"**T = {Tc_i:.1f} °C**\n")
        mode.append(f"![SISV {Tc_i}](figures/{fn})\n")

    # ---- summary table (per temperature) ---------------------------------
    m = []
    m.append("| T [°C] | R_h,z [68% CI nm] | CV R² | PDI | α | CORENN peak | I_VV | depol [%] | η/η_lo |")
    m.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        s = r["suite"]
        cp = (f"{s['corenn'].peaks[0]['radius_nm']:.0f} nm" if s["corenn"] and s["corenn"].peaks else "—")
        b = s["boot"]["R_h"]
        rh = (f"{b['median']*1e9:.0f} [{b['p16']*1e9:.0f}–{b['p84']*1e9:.0f}]"
              if b else f"{s['gvq2']['R_h']*1e9:.0f}")
        m.append(f"| {r['Tc']:.1f} | {rh} | {s['cv']['cv_r2']:.3f} | {s['app']['PDI_mean']:.2f} | "
                 f"{s['sdc']['powerlaw_alpha']:.2f} | {cp} | {r['I_vv']:.0f} | "
                 f"{r['dep']['depol_mean']*100:.3f} | {eta_rel[i]:.2f} |")
    summary_table = m

    main_result = (
        "A narrow, ~120 nm, nearly optically isotropic polymer. The z-average R_h appears to "
        f"fall {Rh[0]:.0f}→{Rh[-1]:.0f} nm over {Tc[0]:.1f}→{Tc[-1]:.1f} °C, but this is mostly "
        "the **fixed-viscosity artifact** (η recorded as a constant 0.47 mPa·s); the same-"
        "sample solve implies the true η drops ~"
        f"{(1-eta_rel[-1])*100:.0f}% so the size is essentially T-independent. PDI stays "
        f"~{PDI.mean():.2f} and the predictive CV R² is ~1.0 (clean monodisperse). **ρ = R_g/R_h "
        "is not determined** (low-q structure factor, not a Guinier R_g); the static intensity "
        "shows a weak transition near 15–22 °C.")

    ctx = {
        "title": "SISV_1% polymer (THF/DOX 7:3) — provisional DLS analysis report",
        "root": ROOT, "data_path": os.path.relpath(DATA, ROOT),
        "script": "reports/sisv_report.py", "command": "python reports/sisv_report.py",
        "dataset_class": "multi-angle, multi-temperature, polarized (VH/VV/3D)",
        "overview": (f"Polymer in THF/dioxane by multi-angle PCS across {len(rows)} "
                     f"temperatures ({Tc.min():.1f}–{Tc.max():.1f} °C); each is a MAPCS "
                     "experiment (27 angles × {VH,VV,3D}, 5 replicates). The full suite runs "
                     "on VV at every temperature."),
        "solvent": {"n": 1.40, "lambda_nm": 632.8, "source": "THF/dioxane 7:3, assumed"},
        "eta_source": "recorded export (fixed 0.47 mPa·s)",
        "eta_note": "fixed at all temperatures — placeholder; THF thins on heating",
        "static_scale": "absolute (toluene-referenced) — assumed, unverified",
        "particle_model": "homogeneous sphere (assumed; ρ not determined — low-q S(q))",
        "primary_model": "CORENN multi-angle PSD (VV) + same-sample joint-temperature solve",
        "next_action": ("supply the true η(T) of THF/dioxane to remove the apparent-shrinkage "
                        "artifact; confirm the toluene reference to lock the SLS scale."),
        "limitations": "viscosity placeholder; MAPCS replicates not exposed (angle bootstrap); sphere model assumed",
        "provisional_flags": ["viscosity is a fixed placeholder at all T (apparent R_h(T) is mostly artifact)",
                              "toluene/absolute-intensity reference not confirmable from the export",
                              "sphere form factor assumed (ρ not determined — low-q structure factor)"],
        "generated_files": (["SISV.md", "SISV.pdf", "figures/sisv.png", "figures/sisv_raw_rep.png",
                            "figures/sisv_slowmode.png", "figures/sisv_weightings.png",
                            "figures/sisv_countrate.png", "figures/sisv_highangle.png",
                            "SISV.sidecar.json"]
                            + [f"figures/{fn}" for _, fn in per_temp_figs]),
        "sidecar_path": os.path.join(HERE, "SISV.sidecar.json"),
    }

    md_text = K.build_report(
        ctx, conditions, main_result=main_result, summary_table=summary_table,
        consolidating=("figures/sisv.png",
                       f"Consolidating figure: full suite at {rep['Tc']:.1f} °C (5 panels) plus "
                       "the temperature series of R_h/PDI/depolarization (raw-derived)."),
        primary_results=primary, mode_specific=mode, raw_figs=raw_figs,
        extra_sections=K.sample_synthesis_section_md(rep["suite"], rep["dep"]),
        countrate_fig=("figures/sisv_countrate.png",
                       "Scaled count rate (absolute Rayleigh ratio) vs angle, all "
                       "temperatures (raw data)."),
        per_curve_conditions=[conditions[0]])
    with open(os.path.join(HERE, "SISV.md"), "w") as fh:
        fh.write(md_text)
    md_to_pdf(md_text, os.path.join(HERE, "SISV.pdf"), base_dir=HERE)
    print("Saved reports/SISV.md and SISV.pdf")


if __name__ == "__main__":
    main()
