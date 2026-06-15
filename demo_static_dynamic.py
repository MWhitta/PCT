"""
demo_static_dynamic.py -- Correlate accepted STATIC Guinier and DYNAMIC
(apparent R_h vs q) multi-angle observables for LiTFSI 20 molal.

Both are moments of the same q-dependent intensity weight w(R,q)=n(R)R^6 P(qR):
  static   I(q)       = int w dR                          (0th moment)
  dynamic  D_app(q)   = <D>_w ,  R_h,app(q)=1/<1/R>_w      (weighted-harmonic R)

For LiTFSI, the static data pass the physical two-population Guinier checks, so
the accepted static break and the R_h,app crossover can be compared as two
projections of one size distribution.  This script shows:
  (A) accepted segmented Guinier with the break q;
  (B) R_h,app(q) with the large/small plateaus and crossover q;
  (C) ln R_h,app vs ln q: the power law R_h,app ~ q^-alpha (vs semilog);
  (D) per-population shape ratio rho=R_g/R_h (recorded vs eta-corrected at 50C).

Run:  python demo_static_dynamic.py
"""

import os
import sys
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corenn as C

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LiTFSI", "20m")
G1_MIN = 0.60


def per_angle(tdir):
    q, g, m2, I = [], [], [], []
    T = eta = None
    for md in sorted(glob.glob(os.path.join(tdir, "Measurement *")),
                     key=lambda p: int(p.rsplit(" ", 1)[1])):
        try:
            ds, _ = C.load_measurement_replicates(md)
        except Exception:
            continue
        c = C.cumulant_fit(ds.tau, ds.g2m, beta=ds.beta, order=2, g1_min=G1_MIN)
        th = ds.metadata.get("angle_deg"); Ir = C.static_intensity_proxy(ds)
        if not np.isfinite(c["Gamma"]) or c["Gamma"] <= 0 or th is None or Ir is None:
            continue
        q.append(ds.q); g.append(c["Gamma"]); m2.append(c["mu2"])
        I.append(Ir)
        T, eta = ds.temperature, ds.viscosity
    return (np.array(q), np.array(g), np.array(m2), np.array(I), T, eta)


def main():
    data = {t: per_angle(os.path.join(ROOT, t)) for t in ("22C", "50C")}
    # same-sample corrected eta(50C) (anchored to 22C), for the shape ratio
    ser = [{"label": t, "T": data[t][4], "q": data[t][0], "gamma": data[t][1],
            "eta": data[t][5]} for t in ("22C", "50C")]
    ser_solve = [dict(s) for s in ser]; ser_solve[1]["eta"] = np.nan
    eta50_corr = C.joint_temperature_fit(ser_solve, reference_index=0)["eta_fit"][1]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    axA, axB, axC, axD = axes.ravel()
    bars = {}
    for temp, color in (("22C", "C0"), ("50C", "C3")):
        q, g, m2, I, T, eta = data[temp]
        r = C.static_dynamic_correlation(q, I, g, T, eta, mu2=m2)
        sg, an = r["segmented_guinier"], r["apparent_diffusion"]
        segments = (sg["two_segment"]["segments"] if sg["supports_two_segments"]
                    else [sg["one_segment"]])

        # (A) accepted Guinier diagnostic
        axA.plot(q**2*1e-14, np.log(I), "o", color=color, ms=4, label=temp)
        for seg in segments:
            xx = np.linspace(seg["q2_min"], seg["q2_max"], 20)
            rg = seg["R_g"]
            rg_label = f"R_g={rg*1e9:.0f} nm" if np.isfinite(rg) else "R_g undef"
            axA.plot(
                xx*1e-14,
                seg["slope"]*xx + seg["intercept"],
                "-",
                color=color,
                lw=1.3,
                label=f"{temp} {seg.get('name', 'fit')}: m={seg['slope']:.1e} m², {rg_label}",
            )
        if np.isfinite(r["static_break_q"]):
            axA.axvline((r["static_break_q"]**2)*1e-14, color=color, ls=":", lw=1)

        # (B) R_h,app vs q with crossover
        axB.plot(an["q"]*1e-6, an["R_h_app"]*1e9, "o-", color=color, ms=4,
                 label=f"{temp}: R_h {r['R_h_large']*1e9:.0f}/{r['R_h_small']*1e9:.0f} nm")
        axB.axvline(r["dynamic_crossover_q"]*1e-6, color=color, ls="-", lw=1, alpha=0.5)
        if np.isfinite(r["static_break_q"]):
            axB.axvline(r["static_break_q"]*1e-6, color=color, ls=":", lw=1.5)

        # (C) log-log linearity
        axC.plot(np.log(an["q"]), np.log(an["R_h_app"]*1e9), "o", color=color, ms=4,
                 label=f"{temp}: α={r['powerlaw_alpha']:.2f}, r²={r['r2_loglog']:.3f}")
        xx = np.array([np.log(an["q"]).min(), np.log(an["q"]).max()])
        # reconstruct the fit line: ln Rh(nm) = -alpha*ln q + b
        b = np.mean(np.log(an["R_h_app"]*1e9) + r["powerlaw_alpha"]*np.log(an["q"]))
        axC.plot(xx, -r["powerlaw_alpha"]*xx + b, "-", color=color, lw=1.2)

        # (D) rho per population, recorded vs eta-corrected (50C)
        rho_L_rec, rho_S_rec = r["rho_large"], r["rho_small"]
        if temp == "50C":
            scale = eta / eta50_corr     # R_h,corr = R_h,rec * eta_rec/eta_corr
            rho_L_cor = r["R_g_large"]/(r["R_h_large"]*scale)
            rho_S_cor = r["R_g_small"]/(r["R_h_small"]*scale)
            bars[temp] = (rho_L_rec, rho_S_rec, rho_L_cor, rho_S_cor)
        else:
            bars[temp] = (rho_L_rec, rho_S_rec, rho_L_rec, rho_S_rec)

        print(f"=== {temp} (T={T:.0f}K, eta_rec={eta*1e3:.0f} mPa.s) ===")
        print(f"  LARGE  R_g={r['R_g_large']*1e9:.0f} R_h={r['R_h_large']*1e9:.0f} nm  rho={rho_L_rec:.2f} [{r['shape_large']}]")
        print(f"  SMALL  R_g={r['R_g_small']*1e9:.0f} R_h={r['R_h_small']*1e9:.0f} nm  rho={rho_S_rec:.2f} [{r['shape_small']}]")
        print(f"  static break q={r['static_break_q']*1e-6:.1f}e6  dyn crossover q={r['dynamic_crossover_q']*1e-6:.1f}e6  (ratio {r['crossover_consistency']:.2f})")
        print(f"  R_h,app ~ q^-{r['powerlaw_alpha']:.2f}  r²(loglog)={r['r2_loglog']:.3f}  r²(semilog)={r['r2_semilog']:.3f} -> {r['linear_form']}")

    axA.set_xlabel(r"$q^2$ [$10^{14}$ m$^{-2}$]"); axA.set_ylabel(r"$\ln I(q)$")
    axA.set_title("(A) Two-slope Guinier (static); dotted = break q"); axA.legend(fontsize=8)
    axB.set_xlabel(r"$q$ [$10^6$ m$^{-1}$]"); axB.set_ylabel(r"apparent $R_h$ [nm]")
    axB.set_ylim(0, 800)
    axB.set_title("(B) R_h,app(q): solid=dyn crossover, dotted=static break (coincide)")
    axB.legend(fontsize=8)
    axC.set_xlabel(r"$\ln q$"); axC.set_ylabel(r"$\ln R_{h,app}$ [nm]")
    axC.set_title(r"(C) Power law $R_{h,app}\sim q^{-\alpha}$ (log-log linear)"); axC.legend(fontsize=8)

    labels = ["22C\nlarge", "22C\nsmall", "50C\nlarge", "50C\nsmall"]
    rec = [bars["22C"][0], bars["22C"][1], bars["50C"][0], bars["50C"][1]]
    cor = [bars["22C"][2], bars["22C"][3], bars["50C"][2], bars["50C"][3]]
    x = np.arange(4)
    axD.bar(x-0.2, rec, 0.4, label="recorded η", color="0.6")
    axD.bar(x+0.2, cor, 0.4, label="η-corrected (50C)", color="C2")
    for y, name in [(0.775, "sphere"), (1.0, "shell"), (1.5, "coil")]:
        axD.axhline(y, ls="--", lw=0.8, color="k", alpha=0.4)
        axD.text(3.5, y, name, fontsize=7, va="bottom", ha="right")
    axD.set_xticks(x); axD.set_xticklabels(labels, fontsize=8)
    axD.set_ylabel(r"$\rho = R_g/R_h$")
    axD.set_title("(D) Per-population shape ratio (needs correct η)"); axD.legend(fontsize=8)

    fig.tight_layout(); fig.savefig("static_dynamic_demo.png", dpi=110)
    print(f"\nsame-sample-corrected eta(50C) = {eta50_corr*1e3:.0f} mPa.s (recorded 34)")
    print("Saved static_dynamic_demo.png")


if __name__ == "__main__":
    main()
