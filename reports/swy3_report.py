"""
swy3_report.py -- Full analysis report for SWy3_1: SWy-3 Wyoming montmorillonite
clay in water (22 C), a single-temperature MAPCS experiment with VH/VV/3D
channels at 14 angles (20-150 deg), 10 repetitions each.  Runs the complete
static+dynamic suite on the VV channel plus the depolarized (VV/VH) and
2D-vs-3D analyses, and writes reports/figures/swy3.png and reports/SWy3.pdf
(+ .md, + .sidecar.json).

    python reports/swy3_report.py
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
DATA = os.path.join(ROOT, "data", "SWy3_1")


def main():
    exp = C.load_mapcs_experiment(DATA)
    vv, vh, d3 = exp["VV"], exp["VH"], exp["3D"]
    T = vv[0].temperature or 295.15
    eta = vv[0].viscosity or 0.955e-3

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
    fig.suptitle("SWy-3 montmorillonite (22 °C) — full suite (VV) + polarized (VH/VV/3D)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(HERE, "figures", "swy3.png"))
    print("Saved reports/figures/swy3.png")

    # ---- raw + count-rate figures ----------------------------------------
    raw_figs = []
    rp = os.path.join(HERE, "figures", "swy3_raw.png")
    if K.fig_raw_correlograms(suite, rp, label="SWy-3 VV"):
        raw_figs.append(("figures/swy3_raw.png",
                         "Raw VV correlograms at low/mid/high q: the stitched multi-tau "
                         "correlation (black) over the semi-transparent raw points."))
    conditions = [{"label": "22 °C", "channel": "VV", "T": T, "eta": eta,
                   "datasets": vv, "suite": suite}]
    cr_path = os.path.join(HERE, "figures", "swy3_countrate.png")
    K.fig_count_rate([{"label": "22 °C", "channel": "VV", "datasets": vv},
                      {"label": "22 °C", "channel": "VH", "datasets": vh}], cr_path)
    ha_path = os.path.join(HERE, "figures", "swy3_highangle.png")
    has_ha = K.fig_high_angle_diagnostics(vv, vh, d3, suite, dep, ha_path, label="SWy-3")

    # ---- bespoke narrative -----------------------------------------------
    primary = []
    primary.append("**Channel identification.** Per angle, the two pseudo-cross measurements "
                   "are VH (low count rate) and VV (high count rate); the third is a 3D "
                   "modulated cross. q is recomputed from the angle. The full static+dynamic "
                   "suite runs on the polarized VV channel.\n")
    steps, _ = K.suite_steps_md(suite, channel="VV", start=1)
    primary += steps

    mode = []
    mode += K.rotational_section_md(dep, sample_note=(
        "These are sub-µm clay flocs, for which whole-floc rotation (D_r ∝ 1/R³) is intrinsically "
        "too slow to resolve; the individual platelets are strongly anisotropic but their "
        "rotation is carried only by the weak depolarized channel."))
    mode.append("### Optical anisotropy (depolarization ratio)\n")
    mode.append(f"The depolarized ratio I_VH/I_VV averages **{dep['depol_mean']*100:.2f} %** "
                f"({np.nanmin(ratios)*100:.2f}–{np.nanmax(ratios)*100:.2f} % across angles). "
                "Montmorillonite platelets are optically anisotropic (form birefringence of a "
                "thin disk), so a non-zero VH signal is expected; the ratio quantifies the "
                "anisotropy of the scattering objects (single platelets vs. more isotropic "
                "flocs) and is the robust polarization observable here.\n")
    # ---- platelet (oblate-disk) Stokes–Einstein --------------------------
    _cr = suite["corenn"]
    _nmd, _ = K.mode_descriptors(_cr.R, np.where(_cr.R > 0, _cr.intensity / _cr.R**6, 0.0))
    _R0nm = _nmd[0]["r_peak"] * 1e9 if _nmd else float("nan")
    _gl = suite.get("gvq2_linear") or suite["gvq2"]
    _Rh_nm = (_gl.get("R_h") or suite["gvq2"]["R_h"]) * 1e9
    _a_floc = K.disk_radius_translational(_gl["D"], T, eta) * 1e9      # = (π/2)·R_h
    _a_prim = K.DISK_SPHERE_FACTOR * _R0nm
    _Dr = (dep.get("six_Dr") or 0.0) / 6.0
    _a_rot = K.disk_radius_rotational(_Dr, T, eta) * 1e9
    _rv = K.rotational_verdict(dep)
    mode.append("### Platelet hydrodynamics (oblate-disk Stokes–Einstein)\n")
    mode.append(
        "The Guinier–Porod slope is dimensionally ~2D and the depolarized channel shows optical "
        "anisotropy — both pointing to **thin plates** (consistent with montmorillonite's platelet "
        "habit), so the sphere relation R_h = k_BT/(6πηD) is not the natural size. For a thin "
        "**oblate disk** of radius a the orientation-averaged translational "
        "diffusion is **D_t = k_BT/(12ηa)** — a disk diffuses like a sphere of radius R_h = (2/π)a, "
        "so the **platelet radius a = (π/2)·R_h ≈ 1.57 R_h** (tumbling rotation "
        "D_r = 3k_BT/(32ηa³)).\n")
    mode.append(
        f"- **Individual platelets** (number-weighted primary, R_h ≈ {_R0nm:.0f} nm) → disk radius "
        f"**a ≈ {_a_prim:.0f} nm** (~{2*_a_prim:.0f} nm lateral) — in the montmorillonite "
        "platelet-size range, and the morphology for which the thin-disk law is rigorous.\n")
    mode.append(
        f"- **Dominant scatterers** (z-average R_h ≈ {_Rh_nm:.0f} nm) → effective disk radius "
        f"**≈ {_a_floc:.0f} nm**, but these are **fractal flocs (D_f ≈ 2.4), not single platelets**, "
        "so neither sphere nor thin-disk is exact; the sphere R_h stays the conventional Stokes "
        "radius and the disk value is only an effective size.\n")
    if _Dr > 0 and np.isfinite(_a_rot):
        mode.append(
            f"- **Rotation (depolarized).** Taken at face value the measured 6D_r = "
            f"{dep['six_Dr']:.1f}/s (D_r ≈ {_Dr:.1f}/s) gives a disk radius **≈ {_a_rot:.0f} nm** "
            f"via D_r = 3k_BT/(32ηa³); but rotation is **{_rv['label']}** (weak VH channel), so this "
            "is illustrative, not a measurement.\n")
    mode.append(
        f"The thin-disk Stokes–Einstein is the right morphology for the **primary platelets** "
        f"(~{_a_prim:.0f} nm radius); the micron-scale flocs that dominate the scattering are "
        "fractal aggregates best described by the sphere R_h.\n")

    mode.append("### 2D vs 3D cross-correlation (multiple scattering)\n")
    mode.append("The 3D modulated cross-correlation suppresses multiple scattering, important "
                "for a turbid clay suspension. The coherence factor (intercept β) averages "
                f"{beta_vv.mean():.2f} for 2D VV vs {beta_3d.mean():.2f} for 3D; their angular "
                "dependence (consolidating panel 6) flags where multiple-scattering / geometry "
                "corrections matter.\n")
    if has_ha:
        mode += K.high_angle_section_md(vv, suite, dep, "figures/swy3_highangle.png")
    _low = sorted([d for d in vv if d.metadata.get("path")], key=lambda d: d.q)[:4]
    _sm = K.slow_mode_stitched([os.path.dirname(d.metadata["path"]) for d in _low],
                               os.path.join(HERE, "figures", "swy3_slowmode.png"), label="SWy-3")
    mode += K.slow_mode_section_md(_sm, "figures/swy3_slowmode.png")
    mode += K.psd_weightings_section_md(
        K.fig_psd_weightings(suite["corenn"], float(np.median(suite["q"])), suite["T"],
                             suite["eta"], os.path.join(HERE, "figures", "swy3_weightings.png"),
                             label="SWy-3",
                             D_f=K._mass_fractal_dim((suite.get("regime") or {}).get("porod_morphology"))),
        "figures/swy3_weightings.png")
    _mz = K.fig_sizemax_sensitivity(os.path.dirname(_low[0].metadata["path"]),
                                    os.path.join(HERE, "figures", "swy3_sizemax.png"),
                                    label="SWy-3 (lowest angle)")
    mode += K.sizemax_section_md(_mz, "figures/swy3_sizemax.png")

    # ---- summary table ---------------------------------------------------
    cp = ", ".join(f"{p['radius_nm']:.0f} nm ({p['rel_amount']*100:.0f}%)" for p in suite["corenn"].peaks) if suite["corenn"] else "—"
    _b = suite["boot"]["R_h"]; _rho = suite["rho"]; _cd = suite["corenn_diag"]
    _mph = suite["regime"].get("porod_morphology") if suite.get("regime") else None
    _pp = suite["regime"]["porod"]["exponent"] if suite.get("regime") and suite["regime"].get("porod") else float("nan")
    _pse = suite["regime"]["porod"].get("exponent_se", float("nan")) if suite.get("regime") and suite["regime"].get("porod") else float("nan")
    _pqlo = suite["regime"]["porod"].get("q_min", float("nan")) if suite.get("regime") and suite["regime"].get("porod") else float("nan")
    _pqhi = suite["regime"]["porod"].get("q_max", float("nan")) if suite.get("regime") and suite["regime"].get("porod") else float("nan")
    _hq = suite["regime"]["porod"].get("high_q") if suite.get("regime") and suite["regime"].get("porod") else None
    m = []
    m.append("| quantity | value |")
    m.append("|---|---|")
    m.append(f"| angles × channels | {len(vv)} × (VH, VV, 3D) |")
    _rh = (f"{_b['median']*1e9:.0f} [68% {_b['p16']*1e9:.0f}–{_b['p84']*1e9:.0f}; "
           f"95% {_b['p2_5']*1e9:.0f}–{_b['p97_5']*1e9:.0f}]" if _b
           else f"{suite['gvq2']['R_h']*1e9:.0f}")
    _rhostr = (f"{_rho['rho']:.2f} ({_rho['model']})" if _rho["independent"]
               else "not determined (Porod regime)")
    m.append(f"| z-average R_h (all-q slope) [68%/95% CI] | {_rh} nm |")
    _gl = suite.get("gvq2_linear") or {}
    if _gl.get("n_linear") and np.isfinite(_gl.get("R_h", float("nan"))):
        _glr = (f"D={_gl['D']:.2e}±{_gl['D_se']:.1e} m²/s → R_h={_gl['R_h']*1e9:.0f} nm "
                f"(linear fit, {_gl['n_linear']}/{_gl['n_total']} angles"
                + (f", −{_gl['n_excluded']} hi-q rolloff at q≤{K._qnm(_gl['q_max_linear'])} nm⁻¹"
                   if _gl.get("n_excluded") else ", full range") + ")")
        m.append(f"| apparent D (linear Γ vs q², rolloff excl.) | {_glr} |")
    m.append(f"| shape ρ = R_g/R_h | {_rhostr} |")
    m.append(f"| apparent R_h low→high q | {suite['app']['R_h_low_q_median']*1e9:.0f}→{suite['app']['R_h_high_q_median']*1e9:.0f} nm |")
    _p_ci = f"{_pp:.2f}±{_pse:.2f}" if np.isfinite(_pse) else f"{_pp:.1f}"
    _p_rng = f", q={K._qrange_nm(_pqlo, _pqhi)}" if np.isfinite(_pqlo) and np.isfinite(_pqhi) else ""
    m.append(f"| static power-law I∝q^−p | p={_p_ci} (r²={suite['static_pl']['r2']:.3f}{_p_rng}) |")
    m.append(f"| implied morphology (Porod) | {K._morph_short(_mph) if _mph else '—'} |")
    m.append(f"| platelet radius a = (π/2)·R_h (oblate-disk SE) | primary ≈ {_a_prim:.0f} nm; "
             f"floc-eff ≈ {_a_floc:.0f} nm |")
    m.append(f"| Γ(q²) predictive R² (LOO-CV) | {suite['cv']['cv_r2']:.3f} (in-sample {suite['gvq2']['r2']:.3f}) |")
    m.append(f"| CORENN eff.DOF / cond | {_cd['eff_dof']:.0f}/{_cd['n_basis']}, κ={_cd['cond']:.0e} |" if _cd else "| CORENN eff.DOF / cond | — |")
    m.append(f"| CORENN multi-angle PSD | {cp} |")
    _bp = suite.get("boot_psd")
    m.append((f"| CORENN peak [68%/95% CI, bootstrap] | {_bp['median']:.0f} "
              f"[68% {_bp['p16']:.0f}–{_bp['p84']:.0f}; 95% {_bp['p2_5']:.0f}–{_bp['p97_5']:.0f}] nm |")
             if _bp else "| CORENN peak [CI, bootstrap] | — |")
    _rv = K.rotational_verdict(dep)
    m.append(f"| 6 D_r (VH−VV intercept) | {dep['six_Dr']:.1f} /s ({_rv['label']}) |")
    m.append(f"| depolarization ratio | {dep['depol_mean']*100:.2f} % |")
    m.append(f"| mean intercept β (2D / 3D) | {beta_vv.mean():.2f} / {beta_3d.mean():.2f} |")
    summary_table = m

    _morph_txt = (f"implies **{_mph['label']}** — a branched, mass-fractal floc rather than "
                  "compact spheres, so the sphere form factor in the PSD is only an effective "
                  "approximation" if _mph else
                  "is not cleanly resolved, so the sphere form factor is assumed, not confirmed")
    pk_nm = suite["corenn"].peaks[0]["radius_nm"] if suite["corenn"] and suite["corenn"].peaks else float("nan")
    # _R0nm (number-weighted primary) already computed for the platelet section above
    _qrg_min = (suite.get("regime") or {}).get("qRg_min", float("nan"))
    main_result = (
        "A flocculated Wyoming-montmorillonite clay suspension — anisotropic platelets aggregated "
        "into broad, mass-fractal flocs in water.\n\n"
        f"- **Size:** the scattering is dominated by ~**{pk_nm:.0f} nm** flocs (intensity-weighted); "
        f"by number the population centres near ~**{_R0nm:.0f} nm**; z-average R_h "
        f"**{suite['gvq2']['R_h']*1e9:.0f} nm**; broad / polydisperse (mean PDI "
        f"{suite['app'].get('PDI_mean', float('nan')):.2f}).\n"
        f"- **Structure:** the static Porod slope I ∝ q^−{_p_ci} ⇒ **{K._morph_short(_mph) if _mph else 'unresolved morphology'}** — "
        "a restructured / branched floc, not compact spheres, so the sphere PSD form factor is only "
        "an effective approximation and **ρ = R_g/R_h is not determined**.\n"
        f"- **Anisotropy:** a depolarized signal I_VH/I_VV ≈ {dep['depol_mean']*100:.1f}% confirms "
        f"plate-like, optically anisotropic scatterers; rotational diffusion is **{_rv['label']}** "
        "(carried only by the weak VH channel).\n"
        f"- **Measurement limits:** the lowest q still has q·R_g ≈ {_qrg_min:.1f} (> 1.3), so no "
        "Guinier regime is reached — R_g, aggregation number and ρ are not directly measurable; "
        "the toluene absolute-intensity scale is unverified.")

    ctx = {
        "title": "SWy-3 montmorillonite (1 %, 22 °C) — provisional polarized DLS analysis report",
        "root": ROOT, "data_path": os.path.relpath(DATA, ROOT),
        "script": "reports/swy3_report.py", "command": "python reports/swy3_report.py",
        "dataset_class": "multi-angle, single-temperature, polarized (VH/VV/3D)",
        "overview": (f"SWy-3 Wyoming montmorillonite clay in water at 22 °C by multi-angle "
                     f"PCS: {len(vv)} angles ({a_vv.min():.0f}–{a_vv.max():.0f}°), three "
                     "measurements per angle — depolarized VH, polarized VV (both 2D "
                     "pseudo-cross), and 3D modulated cross, 10 repetitions each. The full "
                     "suite runs on VV."),
        "solvent": {"n": 1.33, "lambda_nm": 632.8, "source": "water, 22 °C"},
        "eta_source": f"recorded export (water, {eta*1e3:.2f} mPa·s)",
        "eta_note": "water at 22 °C — physically reasonable recorded value",
        "static_scale": "absolute (toluene-referenced) — assumed, unverified",
        "particle_model": (f"mass-fractal clay floc (Porod q^−{_pp:.1f} ⇒ {K._morph_short(_mph)}); "
                           "PSD sphere form factor is an effective approximation"
                           if _mph else "homogeneous sphere (assumed; ρ not determined)"),
        "primary_model": "CORENN multi-angle PSD (VV) + depolarized VV/VH model",
        "next_action": ("fit a mass-fractal (and/or oblate-disk) form factor — the Porod slope "
                        "rules out compact spheres; confirm the toluene reference to lock the "
                        "absolute SLS scale."),
        "limitations": "sphere PSD form factor inconsistent with the mass-fractal Porod slope; toluene reference unverified",
        "provisional_flags": [
            "toluene/absolute-intensity reference not confirmable from the export",
            (f"lower-q Porod/fractal slope q^−{_p_ci} implies a mass-fractal / branched floc, not the sphere assumed by the PSD"
             if _mph else "sphere form factor assumed (ρ not determined)"),
            ("high-q SWy3 points flatten/upturn and are excluded from the slope as an end-of-scaling/background diagnostic"
             if _hq else "no high-q exclusion flagged"),
            "the clay is optically anisotropic (plate-shaped) — the sphere/translational model is an approximation"],
        "generated_files": ["SWy3.md", "SWy3.pdf", "figures/swy3.png", "figures/swy3_raw.png",
                            "figures/swy3_slowmode.png", "figures/swy3_weightings.png",
                            "figures/swy3_sizemax.png",
                            "figures/swy3_countrate.png", "figures/swy3_highangle.png",
                            "SWy3.sidecar.json"],
        "sidecar_path": os.path.join(HERE, "SWy3.sidecar.json"),
    }

    md_text = K.build_report(
        ctx, conditions, main_result=main_result, summary_table=summary_table,
        consolidating=("figures/swy3.png",
                       "Consolidating figure: 5 suite panels (VV) plus the depolarization "
                       "ratio and 2D/3D coherence β vs angle (raw-derived). Error bars are "
                       "1 SEM across repetitions where repetitions exist; the static exponent "
                       "is fit only on the accepted lower-q scaling window."),
        primary_results=primary, mode_specific=mode, raw_figs=raw_figs,
        extra_sections=K.sample_synthesis_section_md(suite, dep),
        countrate_fig=("figures/swy3_countrate.png",
                       "Scaled count rate (absolute Rayleigh ratio) vs angle, VV and VH "
                       "channels (raw data)."))
    with open(os.path.join(HERE, "SWy3.md"), "w") as fh:
        fh.write(md_text)
    md_to_pdf(md_text, os.path.join(HERE, "SWy3.pdf"), base_dir=HERE)
    print("Saved reports/SWy3.md and SWy3.pdf")


if __name__ == "__main__":
    main()
