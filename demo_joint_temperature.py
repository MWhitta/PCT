"""
demo_joint_temperature.py -- Joint multi-temperature analysis of LiTFSI 20 molal
at 22 C and 50 C, assuming the two datasets are the *same sample*.

For one sample the hydrodynamic size is temperature-independent; only the
dynamics rescale as D_z(T) = kT / (6 pi eta(T) R_h,z).  The recorded solvent
viscosity is a fixed 34 mPa.s at both temperatures (a data-entry placeholder),
which is physically wrong for an electrolyte.  The same-sample constraint turns
this into a measurement: anchoring the size to 22 C, the 50 C diffusion implies
the true eta(50 C).

Run:  python demo_joint_temperature.py
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
CUMULANT_G1_MIN = 0.60


def angle_rates(temp_dir):
    q, gamma = [], []
    T = eta = None
    for md in sorted(glob.glob(os.path.join(temp_dir, "Measurement *")),
                     key=lambda p: int(p.rsplit(" ", 1)[1])):
        try:
            ds, _ = C.load_measurement_replicates(md)
        except Exception:
            continue
        cum = C.cumulant_fit(
            ds.tau,
            ds.g2m,
            beta=ds.beta,
            order=2,
            g1_min=CUMULANT_G1_MIN,
        )
        if np.isfinite(cum["Gamma"]) and cum["Gamma"] > 0:
            q.append(ds.q); gamma.append(cum["Gamma"])
            T, eta = ds.temperature, ds.viscosity
    return np.array(q), np.array(gamma), T, eta


def main():
    series = []
    for temp in ("22C", "50C"):
        q, g, T, eta = angle_rates(os.path.join(ROOT, temp))
        series.append({"label": temp, "T": T, "q": q, "gamma": g, "eta": eta})
        print(
            f"{temp}: {q.size} angles, T={T:.1f} K, "
            f"recorded eta={eta*1e3:.1f} mPa.s, "
            f"initial-cumulant g1_min={CUMULANT_G1_MIN:.2f}"
        )

    # (1) using the (fixed) recorded viscosity -> apparent same-sample test
    res_fixed = C.joint_temperature_fit(series)
    print("\n--- with recorded (fixed 34 mPa.s) viscosity ---")
    for lab, T, Dz, Rh in zip(res_fixed["labels"], res_fixed["T"],
                              res_fixed["D_z"], res_fixed["R_h"]):
        print(f"  {lab}: D_z={Dz:.3e} m^2/s  R_h(recorded eta)={Rh*1e9:.0f} nm")
    print(f"  shared R_h,z={res_fixed['R_h_z']*1e9:.0f} nm, R_h spread CV={res_fixed['consistency_cv']:.2f}")
    print(f"  (CV is large -> the recorded constant eta is inconsistent with one sample)")

    # (2) same-sample constraint: anchor to 22 C, SOLVE eta(50 C)
    series_solve = [dict(s) for s in series]
    series_solve[1]["eta"] = np.nan        # drop the (wrong) 50 C viscosity
    res_solve = C.joint_temperature_fit(series_solve, reference_index=0)
    eta22, eta50 = res_solve["eta_fit"][0], res_solve["eta_fit"][1]
    print("\n--- same-sample constraint (anchor 22 C, solve 50 C) ---")
    print(f"  shared R_h,z = {res_solve['R_h_z']*1e9:.0f} nm (from 22 C)")
    print(f"  D_z(22C)={res_solve['D_z'][0]:.3e}  D_z(50C)={res_solve['D_z'][1]:.3e} m^2/s")
    print(f"  implied eta(50C) = {eta50*1e3:.1f} mPa.s  (recorded {series[1]['eta']*1e3:.0f})")
    print(f"  eta(50C)/eta(22C) = {eta50/eta22:.2f}  (electrolyte thins on heating)")

    # plot
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    for s, color in zip(series, ("C0", "C3")):
        axA.plot(s["q"]**2*1e-14, s["gamma"], "o", color=color, ms=4, label=s["label"])
        qq = np.linspace(0, s["q"].max()**2, 40)
        Dz = C.gamma_vs_q2(s["q"], s["gamma"])["D"]
        axA.plot(qq*1e-14, Dz*qq, "-", color=color, lw=1)
    axA.set_xlabel(r"$q^2$ [$10^{14}$ m$^{-2}$]"); axA.set_ylabel(r"$\Gamma$ [1/s]")
    axA.set_title(
        "(A) Initial-cumulant decay rate vs q²\n"
        "50C uses the corrected early-decay Γ fit"
    )
    axA.legend(fontsize=8)

    axB.bar(["22 C\n(anchor)", "50 C\nrecorded", "50 C\nsame-sample"],
            [eta22*1e3, series[1]["eta"]*1e3, eta50*1e3],
            color=["C0", "0.7", "C3"])
    axB.set_ylabel("viscosity [mPa·s]")
    axB.set_title("(B) Recorded vs same-sample-implied η(50 C)")
    fig.tight_layout(); fig.savefig("joint_temperature_demo.png", dpi=120)
    print("\nSaved joint_temperature_demo.png")


if __name__ == "__main__":
    main()
