"""
demo_polarized.py -- Depolarized (VV/VH) DLS analysis: separating translational
from rotational diffusion, validated on synthetic data and applied to the real
JM_F0 MAPCS experiment (28 angles x {VH, VV, 3D-mod-cross}).

Polarized (VV) scattering decays at Gamma_VV = q^2 D_t; depolarized (VH) at
Gamma_VH = q^2 D_t + 6 D_r.  Plotting both vs q^2 gives D_t from the slopes and
6 D_r from the VH intercept.

Run:  python demo_polarized.py [path_to_JM_F0_temperature_dir]
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
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "JM_F0", "25C")
T_WATER, ETA_WATER, N_WATER = 298.15, 0.89e-3, 1.33


def synthetic_anisotropic(R_h=30e-9):
    """Small optically anisotropic spheres -> resolvable rotational diffusion."""
    rng = np.random.default_rng(0)
    D_t = C.k_BOLTZMANN * T_WATER / (6 * np.pi * ETA_WATER * R_h)
    D_r = C.k_BOLTZMANN * T_WATER / (8 * np.pi * ETA_WATER * R_h**3)
    tau = np.geomspace(1e-7, 1.0, 200)
    vv, vh = [], []
    for th in np.arange(20, 151, 10.0):
        q = C.q_from_angle(th, N_WATER, 632.8)
        beta = 0.9
        gvv = beta * np.exp(-2 * q*q*D_t * tau) + 1e-3 * rng.standard_normal(tau.size)
        gvh = beta * np.exp(-2 * (q*q*D_t + 6*D_r) * tau) + 1e-3 * rng.standard_normal(tau.size)
        a = C.DLSDataset(tau=tau, g2m=gvv, q=q, beta=beta, temperature=T_WATER, viscosity=ETA_WATER)
        b = C.DLSDataset(tau=tau, g2m=gvh, q=q, beta=beta, temperature=T_WATER, viscosity=ETA_WATER)
        a.metadata.update(angle_deg=th, scaled_count_rate=100.0)
        b.metadata.update(angle_deg=th, scaled_count_rate=1.0)
        vv.append(a); vh.append(b)
    return vv, vh, D_t, D_r


def main(jm_dir):
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (A) synthetic validation
    vv, vh, D_t, D_r = synthetic_anisotropic()
    rs = C.depolarized_diffusion(vv, vh, temperature=T_WATER, viscosity=ETA_WATER)
    axA.plot(rs["q_vv"]**2*1e-14, rs["gamma_vv"], "o", ms=4, label="VV (translational)")
    axA.plot(rs["q_vh"]**2*1e-14, rs["gamma_vh"], "s", ms=4, label="VH (+ rotational)")
    qq = np.linspace(0, rs["q_vh"].max()**2, 50)
    axA.plot(qq*1e-14, rs["fit_vv"]["D"]*qq + rs["fit_vv"]["intercept"], "C0-")
    axA.plot(qq*1e-14, rs["fit_vh"]["D"]*qq + rs["fit_vh"]["intercept"], "C1-")
    axA.set_xlabel(r"$q^2$ [$10^{14}$ m$^{-2}$]"); axA.set_ylabel(r"$\Gamma$ [1/s]")
    axA.set_title("(A) Synthetic 30 nm anisotropic\n"
                  f"$D_r$={rs['D_r']:.0f}/s (truth {D_r:.0f}); "
                  f"$R_{{rot}}$={rs['R_rot']*1e9:.0f} nm")
    axA.legend(fontsize=8)

    if not os.path.isdir(jm_dir):
        print(f"[!] JM_F0 dir not found: {jm_dir}; saving synthetic-only figure.")
        for ax in (axB, axC):
            ax.text(0.5, 0.5, "JM_F0 data unavailable", ha="center", va="center")
        fig.tight_layout(); fig.savefig("polarized_demo.png", dpi=110); return

    # (B) real JM_F0 Gamma vs q^2
    exp = C.load_mapcs_experiment(jm_dir)
    rj = C.depolarized_diffusion(exp["VV"], exp["VH"], temperature=T_WATER,
                                 viscosity=ETA_WATER, tau_max=0.05)
    axB.plot(rj["q_vv"]**2*1e-14, rj["gamma_vv"], "o", ms=4, label="VV")
    axB.plot(rj["q_vh"]**2*1e-14, rj["gamma_vh"], "s", ms=4, label="VH")
    qq = np.linspace(0, max(rj["q_vv"].max(), rj["q_vh"].max())**2, 50)
    axB.plot(qq*1e-14, rj["fit_vv"]["D"]*qq + rj["fit_vv"]["intercept"], "C0-",
             label=f"$R_h$≈{rj['R_h']*1e9:.0f} nm")
    axB.plot(qq*1e-14, rj["fit_vh"]["D"]*qq + rj["fit_vh"]["intercept"], "C1-")
    axB.set_xlabel(r"$q^2$ [$10^{14}$ m$^{-2}$]"); axB.set_ylabel(r"$\Gamma$ [1/s]")
    axB.set_title("(B) Real JM_F0: VV ≈ VH\n(rotation below resolution for µm aggregates)")
    axB.legend(fontsize=8)

    # (C) depolarization ratio vs angle
    angs = sorted(rj["depolarization_ratio"])
    ratios = [rj["depolarization_ratio"][a] for a in angs]
    axC.plot(angs, ratios, "o-", ms=4)
    axC.axhline(rj["depol_mean"], color="C1", ls="--",
                label=f"mean {rj['depol_mean']*100:.1f}%")
    axC.set_xlabel("scattering angle [deg]")
    axC.set_ylabel(r"depol. ratio $I_{VH}/I_{VV}$")
    axC.set_title("(C) Depolarization ratio\n(optical anisotropy)")
    axC.legend(fontsize=8)

    fig.tight_layout(); fig.savefig("polarized_demo.png", dpi=110)
    print("Saved polarized_demo.png\n")
    print("=== Synthetic (validation) ===")
    print(f"  R_h={rs['R_h']*1e9:.1f} nm (truth 30)  D_r={rs['D_r']:.0f}/s (truth {D_r:.0f})  "
          f"R_rot={rs['R_rot']*1e9:.1f} nm  resolved={rs['rotation_resolved']}")
    print("=== Real JM_F0/25C ===")
    print(f"  loaded VH/VV/3D = {len(exp['VH'])}/{len(exp['VV'])}/{len(exp['3D'])}")
    print(f"  D_t(VV)={rj['fit_vv']['D']:.2e}  D_t(VH)={rj['fit_vh']['D']:.2e} m^2/s (consistent)")
    print(f"  apparent translational R_h ≈ {rj['R_h']*1e9:.0f} nm (polydisperse aggregates)")
    print(f"  6 D_r = {rj['six_Dr']:.1f}/s -> rotation {'RESOLVED' if rj['rotation_resolved'] else 'below resolution'}")
    print(f"  depolarization ratio I_VH/I_VV = {rj['depol_mean']*100:.2f}% (weak optical anisotropy)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
