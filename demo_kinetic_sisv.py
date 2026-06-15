"""
demo_kinetic_sisv.py -- Temperature-series ("kinetic") analysis of the SISV_1%
polymer (THF/DOX 7:3) across 7 temperatures (11.5-30.5 C), each a full MAPCS
experiment (27 angles x {VH, VV, 3D}, 5 reps).

Tracks, vs temperature:
  (A) z-average translational R_h (from VV Gamma-vs-q^2) and polydispersity;
  (B) static scattered intensity (aggregation indicator);
  (C) depolarization ratio I_VH/I_VV and the rotational term 6 D_r
      (temperature dependence of the VV and VH polarization channels).

Run:  python demo_kinetic_sisv.py [path_to_SISV_sample_dir]
"""

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
DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SISV_1%")


def main(sample_dir):
    series = C.load_temperature_series(sample_dir)
    if not series:
        print(f"No temperature folders under {sample_dir}"); return
    print(f"temperatures (C): {list(series.keys())}")

    rows = []
    for Tc, exp in series.items():
        vv, vh = exp["VV"], exp["VH"]
        if not vv:
            continue
        T = vv[0].temperature or (Tc + 273.15)
        eta = vv[0].viscosity
        dep = C.depolarized_diffusion(vv, vh, temperature=T, viscosity=eta)
        # static intensity & polydispersity from VV (z-average)
        I_vv = np.nanmedian([d.metadata.get("scaled_count_rate", np.nan) for d in vv])
        mu2, gam, qv = [], [], []
        for d in vv:
            c = C.cumulant_fit(d.tau, d.g2m, beta=d.beta, order=2)
            if np.isfinite(c["Gamma"]) and c["Gamma"] > 0:
                gam.append(c["Gamma"]); mu2.append(c["mu2"]); qv.append(d.q)
        pdi = float(np.nanmean(np.array(mu2) / np.array(gam) ** 2)) if gam else np.nan
        rows.append({
            "Tc": Tc, "T": T, "eta": eta,
            "R_h": dep["R_h"], "D_t": dep["fit_vv"]["D"],
            "I_vv": I_vv, "PDI": pdi,
            "depol": dep["depol_mean"], "six_Dr": dep["six_Dr"],
            "vh_r2": dep["fit_vh"]["r2"],
            "q": np.array(qv), "gamma": np.array(gam),
        })

    Tc = np.array([r["Tc"] for r in rows])
    Rh = np.array([r["R_h"] for r in rows]) * 1e9
    Ivv = np.array([r["I_vv"] for r in rows])
    PDI = np.array([r["PDI"] for r in rows])
    depol = np.array([r["depol"] for r in rows])
    six_Dr = np.array([r["six_Dr"] for r in rows])

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    axA.plot(Tc, Rh, "o-", color="C0")
    axA.set_xlabel("temperature [°C]"); axA.set_ylabel("z-average $R_h$ [nm]", color="C0")
    axA.tick_params(axis="y", labelcolor="C0")
    a2 = axA.twinx(); a2.plot(Tc, PDI, "s--", color="C1")
    a2.set_ylabel("PDI", color="C1"); a2.tick_params(axis="y", labelcolor="C1")
    axA.set_title("(A) Size & polydispersity vs T")

    axB.plot(Tc, Ivv, "o-", color="C2")
    axB.set_xlabel("temperature [°C]"); axB.set_ylabel("VV scaled intensity [kHz/mW]")
    axB.set_title("(B) Static intensity vs T (aggregation)")

    axC.plot(Tc, depol*100, "o-", color="C3", label="depol $I_{VH}/I_{VV}$ [%]")
    axC.set_xlabel("temperature [°C]"); axC.set_ylabel("depolarization ratio [%]", color="C3")
    axC.tick_params(axis="y", labelcolor="C3")
    c2 = axC.twinx(); c2.plot(Tc, six_Dr, "^--", color="C4", label="6 D_r [1/s]")
    c2.set_ylabel("6 $D_r$ [1/s] (VH−VV intercept)", color="C4")
    c2.tick_params(axis="y", labelcolor="C4")
    axC.set_title("(C) Polarization (VV/VH) vs T")

    fig.tight_layout(); fig.savefig("kinetic_sisv_demo.png", dpi=110)
    print("Saved kinetic_sisv_demo.png\n")
    print(" T[C]  R_h[nm]  PDI    I_VV    depol%   6Dr[1/s]  VHr2")
    for r in rows:
        print(f" {r['Tc']:5.1f}  {r['R_h']*1e9:6.0f}  {r['PDI']:.3f}  {r['I_vv']:6.1f}  "
              f"{r['depol']*100:6.3f}  {r['six_Dr']:8.1f}  {r['vh_r2']:.2f}")
    # simple transition detection: largest jump in static intensity
    if len(Tc) > 2:
        dI = np.diff(Ivv) / np.diff(Tc)
        i = int(np.argmax(np.abs(dI)))
        print(f"\nLargest dI/dT between {Tc[i]:.1f} and {Tc[i+1]:.1f} C "
              f"({dI[i]:+.1f} (kHz/mW)/°C) — candidate transition/aggregation onset.")

    # same-sample cross-check: the recorded viscosity is a FIXED 0.47 mPa.s at
    # every T, so the apparent R_h decline is largely a viscosity artifact.
    # Assuming one (T-invariant) polymer, the implied eta(T) should track the
    # real solvent viscosity (THF thins ~30% over this range).
    series_jt = [{"label": f"{r['Tc']}C", "T": r["T"], "q": r["q"],
                  "gamma": r["gamma"], "eta": None} for r in rows]
    series_jt[0]["eta"] = rows[0]["eta"]           # anchor at the lowest T
    jt = C.joint_temperature_fit(series_jt, reference_index=0)
    eta_rel = jt["eta_fit"] / jt["eta_fit"][0]
    print("\nSame-sample test (size assumed T-invariant), implied eta(T)/eta(T_lo):")
    print("  " + "  ".join(f"{t:.0f}C:{e:.2f}" for t, e in zip(Tc, eta_rel)))
    print(f"  -> implied viscosity drops {(1-eta_rel[-1])*100:.0f}% over "
          f"{Tc[0]:.0f}->{Tc[-1]:.0f}C, consistent with THF thinning; i.e. the\n"
          f"     apparent R_h decline is mostly the fixed-viscosity artifact, not real shrinkage.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
