"""
jm_f0_report.py -- Full analysis report for JM_F0 (25 C): a MAPCS experiment
with VH/VV/3D channels at 28 angles.  Runs the complete static+dynamic suite on
the VV channel plus the depolarized (VV/VH) and 2D-vs-3D analyses.  Writes
reports/figures/jm_f0.png and reports/JM_F0.pdf (+ .md).

    python reports/jm_f0_report.py
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
DATA = os.path.join(ROOT, "data", "JM_F0", "25C")


def main():
    exp = C.load_mapcs_experiment(DATA)
    vv, vh, d3 = exp["VV"], exp["VH"], exp["3D"]
    T = vv[0].temperature or 298.15
    eta = vv[0].viscosity or 0.89e-3

    suite = K.run_suite(vv, T, eta, psd_bootstrap=20,
                        replicates=[d.metadata.get("replicates") for d in vv])
    dep = C.depolarized_diffusion(vv, vh, temperature=T, viscosity=eta, tau_max=0.05)

    # depolarization ratio + 2D/3D coherence vs angle
    a_vv = np.array([d.metadata.get("angle_deg") for d in vv])
    scr_vv = np.array([d.metadata.get("scaled_count_rate") for d in vv])
    a_vh = np.array([d.metadata.get("angle_deg") for d in vh])
    scr_vh = np.array([d.metadata.get("scaled_count_rate") for d in vh])
    ratios = np.array([scr_vh[int(np.argmin(np.abs(a_vh - av)))] / s if s else np.nan
                       for av, s in zip(a_vv, scr_vv)])
    o = np.argsort(a_vv)
    beta_vv = np.array([d.beta for d in vv]); a_3d = np.array([d.metadata.get("angle_deg") for d in d3])
    beta_3d = np.array([d.beta for d in d3]); o3 = np.argsort(a_3d)

    # ---- figure: 5 suite panels + polarization panel ---------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 9)); ax = ax.ravel()
    K.plot_suite(ax[:5], suite, "C0", "VV")
    K.label_suite_axes(ax)
    ax[5].plot(a_vv[o], ratios[o]*100, "o-", ms=4, color="C3", label=r"depol $I_{VH}/I_{VV}$ [%]")
    ax[5].axhline(dep["depol_mean"]*100, color="C3", ls="--")
    a2 = ax[5].twinx()
    a2.plot(a_vv[o], beta_vv[o], "s:", ms=3, color="C0", label="2D β")
    a2.plot(a_3d[o3], beta_3d[o3], "^:", ms=3, color="C2", label="3D β")
    a2.set_ylabel("intercept β")
    ax[5].set_xlabel("angle [deg]"); ax[5].set_ylabel("depol [%]", color="C3")
    ax[5].set_title("(6) depolarization + 2D/3D β"); ax[5].legend(fontsize=K.FONT_LEGEND_SMALL, loc="upper left")
    fig.suptitle("JM_F0 (25 °C) — full suite (VV) + polarized (VH/VV/3D) analyses")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(HERE, "figures", "jm_f0.png"))
    print("Saved reports/figures/jm_f0.png")

    # ---- raw + count-rate figures ----------------------------------------
    raw_figs = []
    rp = os.path.join(HERE, "figures", "jm_f0_raw.png")
    if K.fig_raw_correlograms(suite, rp, label="JM_F0 VV"):
        raw_figs.append(("figures/jm_f0_raw.png",
                         "Raw VV correlograms at low/mid/high q: the stitched multi-tau "
                         "correlation (black) over the semi-transparent raw points."))
    conditions = [{"label": "25 °C", "channel": "VV", "T": T, "eta": eta,
                   "datasets": vv, "suite": suite}]
    cr_path = os.path.join(HERE, "figures", "jm_f0_countrate.png")
    K.fig_count_rate([{"label": "25 °C", "channel": "VV", "datasets": vv},
                      {"label": "25 °C", "channel": "VH", "datasets": vh}], cr_path)
    ha_path = os.path.join(HERE, "figures", "jm_f0_highangle.png")
    has_ha = K.fig_high_angle_diagnostics(vv, vh, d3, suite, dep, ha_path, label="JM_F0")

    # ---- bespoke narrative -----------------------------------------------
    primary = []
    primary.append("**Channel identification.** Per angle, the two pseudo-cross measurements "
                   "are VH (low count rate) and VV (high count rate); the third is 3D "
                   "modulated cross. q is recomputed from the angle. The full suite runs on "
                   "the polarized VV channel.\n")
    steps, _ = K.suite_steps_md(suite, channel="VV", start=1)
    primary += steps

    mode = []
    mode += K.rotational_section_md(dep, sample_note=(
        "These are µm-scale aggregates, for which whole-particle rotation (D_r ∝ 1/R³) is far too "
        "slow to measure."))
    mode.append("### Optical anisotropy (depolarization ratio)\n")
    mode.append(f"The intensity ratio I_VH/I_VV averages **{dep['depol_mean']*100:.2f} %** "
                f"({np.nanmin(ratios)*100:.2f}–{np.nanmax(ratios)*100:.2f} % across angles) — a "
                "small but measurable optical anisotropy.\n")
    mode.append("### 2D vs 3D cross-correlation (multiple scattering)\n")
    mode.append("The 3D modulated cross-correlation suppresses multiple scattering. The "
                f"coherence factor (intercept β) averages {beta_vv.mean():.2f} for 2D VV vs "
                f"{beta_3d.mean():.2f} for 3D; their angular dependence (consolidating panel 6) "
                "flags where multiple-scattering / geometry corrections matter.\n")
    if has_ha:
        mode += K.high_angle_section_md(vv, suite, dep, "figures/jm_f0_highangle.png")
    _low = sorted([d for d in vv if d.metadata.get("path")], key=lambda d: d.q)[:4]
    _sm = K.slow_mode_stitched([os.path.dirname(d.metadata["path"]) for d in _low],
                               os.path.join(HERE, "figures", "jm_f0_slowmode.png"), label="JM_F0")
    mode += K.slow_mode_section_md(_sm, "figures/jm_f0_slowmode.png")
    mode += K.psd_weightings_section_md(
        K.fig_psd_weightings(suite["corenn"], float(np.median(suite["q"])), suite["T"],
                             suite["eta"], os.path.join(HERE, "figures", "jm_f0_weightings.png"),
                             label="JM_F0",
                             D_f=K._mass_fractal_dim((suite.get("regime") or {}).get("porod_morphology"))),
        "figures/jm_f0_weightings.png")

    # ---- summary table ---------------------------------------------------
    cp = ", ".join(f"{p['radius_nm']:.0f} nm ({p['rel_amount']*100:.0f}%)" for p in suite["corenn"].peaks) if suite["corenn"] else "—"
    m = []
    m.append("| quantity | value |")
    m.append("|---|---|")
    m.append(f"| angles × channels | {len(vv)} × (VH, VV, 3D) |")
    _b = suite["boot"]["R_h"]; _rho = suite["rho"]; _cd = suite["corenn_diag"]
    _rh = (f"{_b['median']*1e9:.0f} [68% {_b['p16']*1e9:.0f}–{_b['p84']*1e9:.0f}; "
           f"95% {_b['p2_5']*1e9:.0f}–{_b['p97_5']*1e9:.0f}]" if _b
           else f"{suite['gvq2']['R_h']*1e9:.0f}")
    _rhostr = (f"{_rho['rho']:.2f} ({_rho['model']})" if _rho["independent"]
               else "not determined (Porod regime)")
    m.append(f"| z-average R_h [68%/95% CI] | {_rh} nm |")
    m.append(f"| shape ρ = R_g/R_h | {_rhostr} |")
    m.append(f"| apparent R_h low→high q | {suite['app']['R_h_low_q_median']*1e9:.0f}→{suite['app']['R_h_high_q_median']*1e9:.0f} nm |")
    m.append(f"| static power-law I∝q^−p | p={suite['static_pl']['exponent']:.1f} (r²={suite['static_pl']['r2']:.2f}) |")
    m.append(f"| Γ(q²) predictive R² (LOO-CV) | {suite['cv']['cv_r2']:.3f} (in-sample {suite['gvq2']['r2']:.3f}) |")
    m.append(f"| CORENN eff.DOF / cond | {_cd['eff_dof']:.0f}/{_cd['n_basis']}, κ={_cd['cond']:.0e} |" if _cd else "| CORENN eff.DOF / cond | — |")
    m.append(f"| CORENN multi-angle PSD | {cp} |")
    _bp = suite.get("boot_psd")
    m.append((f"| CORENN peak [68%/95% CI, bootstrap] | {_bp['median']:.0f} "
              f"[68% {_bp['p16']:.0f}–{_bp['p84']:.0f}; 95% {_bp['p2_5']:.0f}–{_bp['p97_5']:.0f}] nm |")
             if _bp else "| CORENN peak [CI, bootstrap] | — |")
    m.append(f"| 6 D_r (VH−VV intercept) | {dep['six_Dr']:.1f} /s ({K.rotational_verdict(dep)['label']}) |")
    m.append(f"| depolarization ratio | {dep['depol_mean']*100:.2f} % |")
    m.append(f"| mean intercept β (2D / 3D) | {beta_vv.mean():.2f} / {beta_3d.mean():.2f} |")
    summary_table = m

    main_result = (
        "A large, polydisperse, weakly optically anisotropic aggregate sample. The z-average "
        f"R_h is **{suite['gvq2']['R_h']*1e9:.0f} nm** but at/beyond the resolution limit — the "
        f"angle bootstrap spans {(_b['p2_5']*1e9 if _b else float('nan')):.0f}–"
        f"{(_b['p97_5']*1e9 if _b else float('nan')):.0f} nm and the CORENN peak is rail-pinned "
        "at the largest resolvable size, so the size is a **lower bound**, not a measurement. "
        "The data are deep in the **Porod regime**, so **ρ = R_g/R_h is not determined**. VV "
        "and VH report the same translational dynamics; rotational diffusion is below "
        f"resolution; the {dep['depol_mean']*100:.1f}% depolarization quantifies a small "
        "optical anisotropy.")

    ctx = {
        "title": "JM_F0 (25 °C) — provisional polarized DLS analysis report",
        "root": ROOT, "data_path": os.path.relpath(DATA, ROOT),
        "script": "reports/jm_f0_report.py", "command": "python reports/jm_f0_report.py",
        "dataset_class": "multi-angle, single-temperature, polarized (VH/VV/3D)",
        "overview": (f"Multi-angle PCS at 25 °C: {len(vv)} angles (15–150°), three "
                     "measurements per angle — depolarized VH, polarized VV (both 2D "
                     "pseudo-cross), and 3D modulated cross. The full suite runs on VV."),
        "solvent": {"n": 1.33, "lambda_nm": 632.8, "source": "assumed aqueous"},
        "eta_source": "recorded export / fallback 0.89 mPa·s",
        "eta_note": "verify solvent and temperature",
        "static_scale": "absolute (toluene-referenced) — assumed, unverified",
        "particle_model": "homogeneous sphere (assumed; ρ not determined; depolarization ⇒ weak anisotropy)",
        "primary_model": "CORENN multi-angle PSD (VV) + depolarized VV/VH model",
        "next_action": ("measure at lower angle / use a longer wavelength to bring the µm "
                        "aggregates inside the resolvable window; confirm solvent η and the "
                        "toluene reference."),
        "limitations": "size at/beyond resolution limit; sphere model assumed",
        "provisional_flags": ["z-average size is at/beyond the q-resolution limit (lower bound)",
                              "single repetition per angle — no replicate-repeatability CI (only the angle/model bootstrap)",
                              "toluene/absolute-intensity reference not confirmable from the export",
                              "sphere form factor assumed (ρ not determined); ~2% depolarization implies weak anisotropy"],
        "generated_files": ["JM_F0.md", "JM_F0.pdf", "figures/jm_f0.png", "figures/jm_f0_raw.png",
                            "figures/jm_f0_slowmode.png", "figures/jm_f0_weightings.png",
                            "figures/jm_f0_countrate.png", "figures/jm_f0_highangle.png",
                            "JM_F0.sidecar.json"],
        "sidecar_path": os.path.join(HERE, "JM_F0.sidecar.json"),
    }

    md_text = K.build_report(
        ctx, conditions, main_result=main_result, summary_table=summary_table,
        consolidating=("figures/jm_f0.png",
                       "Consolidating figure: 5 suite panels (VV) plus depolarization ratio "
                       "and 2D/3D coherence β vs angle (raw-derived)."),
        primary_results=primary, mode_specific=mode, raw_figs=raw_figs,
        extra_sections=K.sample_synthesis_section_md(suite, dep),
        countrate_fig=("figures/jm_f0_countrate.png",
                       "Scaled count rate (absolute Rayleigh ratio) vs angle, VV and VH "
                       "channels (raw data)."))
    with open(os.path.join(HERE, "JM_F0.md"), "w") as fh:
        fh.write(md_text)
    md_to_pdf(md_text, os.path.join(HERE, "JM_F0.pdf"), base_dir=HERE)
    print("Saved reports/JM_F0.md and JM_F0.pdf")


if __name__ == "__main__":
    main()
