"""
demo_corenn_hybrid.py -- End-to-end demonstration and validation of the custom
CORENN inversion (corenn.py) and the hybrid hardware/software correlator
(hybrid_correlator.py).

Run:  python demo_corenn_hybrid.py [path_to_LiTFSI_measurement_dir]

Produces ``corenn_hybrid_demo.png`` with four panels:
  (A) synthetic bimodal recovery (ground-truth validation),
  (B) the hybrid g2-1 merge on real data (hardware + software + merged),
  (C) the inverse-variance blend weights and crossover,
  (D) recovered PSD from the hardware-only vs hybrid correlation.
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

DEFAULT_MEAS = ("/Users/mwhittaker/Library/CloudStorage/"
                "GoogleDrive-mwhittaker@lbl.gov/My Drive/PCT/Data/LiTFSI/"
                "20m/22C/Measurement 1")


def synthetic_bimodal():
    """Generate a synthetic bimodal curve and invert it (ground truth known)."""
    rng = np.random.default_rng(0)
    T, eta, n_sol = 295.0, 0.94e-3, 1.33
    q = C.q_from_angle(90.0, n_sol, 632.8)
    tau = np.geomspace(1.6e-6, 10.0, 200)
    R_true = np.array([20e-9, 200e-9])
    w = np.array([0.5, 0.5])
    D = C.diffusion_coefficient(R_true, T, eta)
    beta = 0.34
    reps = []
    for _ in range(5):
        g1 = sum(wi * np.exp(-q * q * d * tau) for wi, d in zip(w, D))
        reps.append(beta * g1**2 + 2e-3 * rng.standard_normal(tau.size))
    ds = C.DLSDataset(tau=tau, g2m=np.mean(reps, axis=0), q=q, beta=beta,
                      temperature=T, viscosity=eta, n_solvent=n_sol)
    res = C.corenn_invert(ds, radius_range=(2e-9, 1e-6), n_basis=22, n_R=140,
                          replicates=reps, uncertainty="bootstrap",
                          n_bootstrap=60)
    return res, R_true * 1e9


def main(meas_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axA, axB, axC, axD = axes.ravel()

    # On a log-radius axis the intensity *per log-decade*, R*I(R), integrates
    # to intensity fractions, so equal areas mean equal intensity (the standard
    # "intensity-weighted PSD vs log radius" convention).
    rw = lambda res, y=None: res.R * (res.intensity if y is None else y)

    # --- (A) synthetic validation ---------------------------------------
    res_s, truth_nm = synthetic_bimodal()
    axA.plot(res_s.R * 1e9, rw(res_s), "C0-", lw=2, label="recovered")
    if res_s.intensity_band is not None:
        axA.fill_between(res_s.R * 1e9, res_s.R * res_s.intensity_band[0],
                         res_s.R * res_s.intensity_band[1], color="C0", alpha=0.25,
                         label="95% band")
    for r in truth_nm:
        axA.axvline(r, color="k", ls="--", alpha=0.6)
    axA.set_xscale("log")
    axA.set_xlabel("radius [nm]"); axA.set_ylabel(r"$R\,I(R)$ (intensity / decade)")
    axA.set_title("(A) Synthetic bimodal recovery\n"
                  f"truth: {truth_nm[0]:.0f} & {truth_nm[1]:.0f} nm; "
                  f"found: " + ", ".join(f"{p['radius_nm']:.0f}nm" for p in res_s.peaks))
    axA.legend(fontsize=8)

    if not os.path.isdir(meas_dir):
        print(f"[!] measurement dir not found: {meas_dir}\n"
              "    Skipping real-data panels; saving synthetic-only figure.")
        for ax in (axB, axC, axD):
            ax.text(0.5, 0.5, "real data unavailable", ha="center", va="center")
        fig.tight_layout(); fig.savefig("corenn_hybrid_demo.png", dpi=130)
        print(res_s.summary())
        return

    # --- build hybrid + load hardware-only ------------------------------
    base, reps = C.load_measurement_replicates(meas_dir)
    ds_hy, merged = H.build_hybrid_dataset(meas_dir)
    hw = merged.components["hardware"]; sw = merged.components["software"]

    # --- (B) the merge ---------------------------------------------------
    axB.plot(hw["tau"], hw["g2m"], "C3.", ms=3, alpha=0.5, label="hardware")
    axB.plot(sw["tau"], sw["g2m"], "C2.", ms=6, alpha=0.7, label="software (count trace)")
    axB.plot(ds_hy.tau, ds_hy.g2m, "k-", lw=1.3, label="merged")
    axB.set_xscale("log")
    axB.set_xlabel(r"lag $\tau$ [s]"); axB.set_ylabel(r"$g_2-1$")
    axB.set_title("(B) Hybrid correlation merge (real LiTFSI)")
    axB.legend(fontsize=8)

    # --- (C) blend weights ----------------------------------------------
    W = merged.components["weights"]
    frac_sw = W["w_sw"] / (W["w_hw"] + W["w_sw"])
    axC.plot(W["tau"], 1 - frac_sw, "C3-", label="hardware weight")
    axC.plot(W["tau"], frac_sw, "C2-", label="software weight")
    cross = W["tau"][np.argmin(np.abs(frac_sw - 0.5))]
    axC.axvline(cross, color="k", ls=":", label=f"crossover {cross:.2f} s")
    axC.set_xscale("log")
    axC.set_xlabel(r"lag $\tau$ [s]"); axC.set_ylabel("blend weight")
    axC.set_title("(C) Inverse-variance blend weights")
    axC.legend(fontsize=8)

    # --- (D) PSD: hardware-only vs hybrid -------------------------------
    res_hw = C.corenn_invert(base, radius_range=(C.DEFAULT_CORENN_RADIUS_RANGE[0], 5e-5), n_basis=30,
                             n_R=200, replicates=reps, fit_baseline=True)
    res_hy = C.corenn_invert(ds_hy, radius_range=(C.DEFAULT_CORENN_RADIUS_RANGE[0], 5e-5), n_basis=30,
                             n_R=200, fit_baseline=True)
    axD.plot(res_hw.R * 1e9, rw(res_hw), "C1-", lw=2, label="hardware only")
    axD.plot(res_hy.R * 1e9, rw(res_hy), "C0-", lw=2, label="hybrid")
    axD.axvline(197.9, color="k", ls="--", alpha=0.6, label="vendor CORENN ~198 nm")
    axD.set_xscale("log")
    axD.set_xlabel("radius [nm]"); axD.set_ylabel(r"$R\,I(R)$ (intensity / decade)")
    axD.set_title("(D) Recovered PSD: hardware-only vs hybrid")
    axD.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("corenn_hybrid_demo.png", dpi=130)
    print("Saved corenn_hybrid_demo.png")
    print("\n--- Synthetic ---"); print(res_s.summary())
    print("\n--- Hardware-only ---"); print(res_hw.summary())
    print("\n--- Hybrid ---"); print(res_hy.summary())
    print(f"\nmerge crossover ~ {cross:.3f} s; software/hardware scale = {merged.scale_sw:.3f}")


if __name__ == "__main__":
    meas = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MEAS
    main(meas)
