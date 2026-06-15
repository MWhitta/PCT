"""
demo_multiangle.py -- Multi-angle DLS analysis on real angle-resolved data.

Demonstrates the multi-angle capabilities added to corenn.py:
  (A) per-angle cumulant decay rate vs q^2  ->  z-average D and R_h;
  (B) per-angle apparent hydrodynamic radius vs q  (polydispersity/aggregate
      diagnostic the single-angle measurement cannot reveal);
  (C) joint multi-angle CORENN PSD (single q-consistent I0(R) via form factor)
      compared to a single back-angle inversion.

Run:  python demo_multiangle.py [path_to_temperature_dir_with_Measurement_*]
"""

import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corenn as C

DEFAULT_DIR = ("/Users/mwhittaker/Library/CloudStorage/"
               "GoogleDrive-mwhittaker@lbl.gov/My Drive/PCT/Data/LiTFSI/20m/22C")


def main(base):
    meas_dirs = sorted(glob.glob(os.path.join(base, "Measurement *")),
                       key=lambda p: int(p.rsplit(" ", 1)[1]))
    datasets, replicates = [], []
    for md in meas_dirs:
        try:
            ds, reps = C.load_measurement_replicates(md)
        except Exception:
            continue
        datasets.append(ds); replicates.append(reps)
    if len(datasets) < 4:
        print("Not enough angles found; aborting."); return
    T, eta = datasets[0].temperature, datasets[0].viscosity

    # --- per-angle cumulant Gamma over the main decay -------------------
    qs, gammas, gsig, angles = [], [], [], []
    for ds, reps in zip(datasets, replicates):
        cum = C.cumulant_fit(ds.tau, ds.g2m, beta=ds.beta, order=2, tau_max=1.0)
        if not np.isfinite(cum["Gamma"]) or cum["Gamma"] <= 0:
            continue
        # per-rep curves live on the hardware lag grid (the dataset may be the
        # hybrid merge on a different grid); fit them on hardware_tau.
        hw_tau = ds.metadata.get("hardware_tau", ds.tau)
        gs = [C.cumulant_fit(hw_tau, r, beta=ds.beta, order=2, tau_max=1.0)["Gamma"]
              for r in reps]
        gs = [g for g in gs if np.isfinite(g) and g > 0]
        qs.append(ds.q); gammas.append(cum["Gamma"]); angles.append(ds.metadata.get("angle_deg"))
        gsig.append(max(np.std(gs) if len(gs) >= 2 else cum["Gamma"]*0.1, cum["Gamma"]*0.02))
    qs, gammas, gsig = map(np.array, (qs, gammas, gsig))

    fit = C.gamma_vs_q2(qs, gammas, sigma=gsig, temperature=T, viscosity=eta)
    Rapp = C.k_BOLTZMANN * T / (6 * np.pi * eta * (gammas / qs**2))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.6))

    # (A) Gamma vs q^2
    axA.errorbar(qs**2 * 1e-14, gammas, yerr=gsig, fmt="o", ms=4, label="cumulant Γ")
    qq = np.linspace(0, qs.max()**2, 50)
    axA.plot(qq * 1e-14, fit["D"] * qq + fit["intercept"], "C1-",
             label=f"D={fit['D']:.2e} m²/s\n$R_h$={fit['R_h']*1e9:.0f} nm, $r^2$={fit['r2']:.3f}")
    axA.set_xlabel(r"$q^2$  [$10^{14}$ m$^{-2}$]"); axA.set_ylabel(r"$\Gamma$ [1/s]")
    axA.set_title("(A) Decay rate vs q²  →  z-average D, $R_h$"); axA.legend(fontsize=8)

    # (B) apparent R_h vs q  (aggregate/polydispersity signature)
    axB.plot(qs * 1e-6, Rapp * 1e9, "o-", ms=4)
    axB.axhline(fit["R_h"] * 1e9, color="C1", ls="--", label=f"z-avg {fit['R_h']*1e9:.0f} nm")
    axB.set_xlabel(r"$q$ [$10^6$ m$^{-1}$]"); axB.set_ylabel(r"apparent $R_h$ [nm]")
    axB.set_title("(B) Apparent $R_h$ vs q\n(rises at low q → slow large-particle population)")
    axB.legend(fontsize=8)

    # (C) joint multi-angle vs single back-angle PSD
    sub = datasets[::3]
    res_ma = C.corenn_multi_angle(sub, radius_range=(2e-9, 2e-6), n_basis=22,
                                  n_R=140, temperature=T, viscosity=eta)
    ds90 = min(datasets, key=lambda d: abs((d.metadata.get("angle_deg") or 0) - 90))
    res90 = C.corenn_invert(ds90, radius_range=(2e-9, 2e-6), n_basis=22, n_R=140,
                            temperature=T, viscosity=eta, fit_baseline=True, lag_max=1.0)
    axC.plot(res_ma.R*1e9, res_ma.R*res_ma.intensity, "C0-", lw=2,
             label=f"multi-angle ({len(sub)} q)")
    axC.plot(res90.R*1e9, res90.R*res90.intensity, "C3-", lw=1.5, alpha=0.8,
             label="single angle (90°)")
    axC.set_xscale("log"); axC.set_xlabel("radius [nm]")
    axC.set_ylabel(r"$R\,I(R)$ (intensity / decade)")
    axC.set_title("(C) Multi-angle vs single-angle PSD"); axC.legend(fontsize=8)

    fig.tight_layout(); fig.savefig("multiangle_demo.png", dpi=110)
    print("Saved multiangle_demo.png")
    print(f"\nAngles used: {len(qs)} ({min(angles):.0f}°–{max(angles):.0f}°)")
    print(f"Gamma-vs-q^2: D={fit['D']:.3e} m^2/s, intercept={fit['intercept']:.2f} 1/s, "
          f"r2={fit['r2']:.3f}  =>  R_h(z-avg)={fit['R_h']*1e9:.0f} nm")
    print(f"Apparent R_h: {Rapp.min()*1e9:.0f} nm (high q) .. {Rapp.max()*1e9:.0f} nm (low q)")
    print(f"Multi-angle CORENN (intensity PSD): lambda={res_ma.lam:.2e}, "
          + ", ".join(f"{p['radius_nm']:.0f}nm({p['rel_amount']*100:.0f}%)" for p in res_ma.peaks))
    print("  (number/volume conversions amplify small-R noise by R^-6/R^-3 on broad\n"
          "   real PSDs and are only reliable for clean, well-separated modes.)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
