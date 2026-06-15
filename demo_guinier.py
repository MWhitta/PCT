"""
demo_guinier.py -- Static (Guinier) and dynamic (apparent-R_h-vs-q) multi-angle
analysis of LiTFSI 20 molal at 22 C and 50 C.

  (A) Guinier plot: ln I(q) vs q^2 from the angle-resolved static intensity
      (laser-power-scaled count rate) -> R_g.
  (B) Apparent R_h(q) = kT / (6 pi eta D_app(q)), D_app = Gamma(q)/q^2, with a
      D_app = D0(1 + C q^2) fit quantifying polydispersity (z-average R_h, the
      coefficient C, the high/low-q span) and structure (R_h,app monotonicity).

Run:  python demo_guinier.py
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


def load_angles(temp_dir):
    """Per-angle replicate-mean datasets, sorted by q."""
    out = []
    for md in sorted(glob.glob(os.path.join(temp_dir, "Measurement *")),
                     key=lambda p: int(p.rsplit(" ", 1)[1])):
        try:
            ds, reps = C.load_measurement_replicates(md)
        except Exception:
            continue
        out.append((ds, reps))
    out.sort(key=lambda dr: dr[0].q)
    return out


def per_angle(datasets, cumulant_g1_min=0.60):
    q, gamma, mu2, intensity, theta = [], [], [], [], []
    for ds, _ in datasets:
        cum = C.cumulant_fit(
            ds.tau,
            ds.g2m,
            beta=ds.beta,
            order=2,
            g1_min=cumulant_g1_min,
        )
        if not np.isfinite(cum["Gamma"]) or cum["Gamma"] <= 0:
            continue
        th = ds.metadata.get("angle_deg")
        Iraw = C.static_intensity_proxy(ds)
        if Iraw is None or th is None:
            continue
        q.append(ds.q); gamma.append(cum["Gamma"]); mu2.append(cum["mu2"])
        theta.append(th)
        intensity.append(Iraw)
    return (np.array(q), np.array(gamma), np.array(mu2),
            np.array(intensity), np.array(theta))


def guinier_segment_label(temp, seg):
    name = seg.get("name", "fit")
    if name == "low_q_large":
        regime = "low-q"
    elif name == "high_q_small":
        regime = "high-q"
    else:
        regime = "fit"
    return (
        f"{temp} {regime}: m={seg['slope']:.2e} m², "
        f"$R_g$={seg['R_g']*1e9:.0f} nm"
    )


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))
    summary_lines = []
    for temp, color in [("22C", "C0"), ("50C", "C3")]:
        tdir = os.path.join(ROOT, temp)
        if not os.path.isdir(tdir):
            print(f"[!] {tdir} missing"); continue
        dsets = load_angles(tdir)
        T = dsets[0][0].temperature
        eta = dsets[0][0].viscosity
        q, gamma, mu2, I, theta = per_angle(dsets)

        sg = C.segmented_guinier_fit(q, I, n_min=5)
        an = C.apparent_diffusion_analysis(q, gamma, T, eta, mu2=mu2)
        two = sg["two_segment"]
        segments = two["segments"] if sg["supports_two_segments"] else [sg["one_segment"]]
        low_seg = segments[0]
        high_seg = segments[-1]

        # (A) Guinier
        axA.plot(q**2 * 1e-14, np.log(I), "o", color=color, ms=4,
                 label=f"{temp} data")
        for seg in segments:
            q2_line = np.linspace(seg["q2_min"], seg["q2_max"], 30)
            lnI_line = seg["slope"] * q2_line + seg["intercept"]
            axA.plot(
                q2_line * 1e-14,
                lnI_line,
                "-",
                color=color,
                lw=1.2,
                label=guinier_segment_label(temp, seg),
            )

        # (B) apparent R_h vs q
        axB.plot(an["q"]*1e-6, an["R_h_app"]*1e9, "o-", color=color, ms=4,
                 label=(f"{temp}: low/high $R_h$="
                        f"{an['R_h_low_q_median']*1e9:.0f}/"
                        f"{an['R_h_high_q_median']*1e9:.0f} nm"))
        axB.axhline(an["R_h_z"]*1e9, color=color, ls=":", lw=1)

        break_text = "none"
        if sg["supports_two_segments"] and two is not None:
            break_text = (
                f"{two['break_q_low']*1e-6:.1f}-"
                f"{two['break_q_high']*1e-6:.1f}e6/m"
            )
        low_rg = low_seg["R_g"] * 1e9 if np.isfinite(low_seg["R_g"]) else float("nan")
        high_rg = high_seg["R_g"] * 1e9 if np.isfinite(high_seg["R_g"]) else float("nan")
        summary_lines.append(
            f"{temp}: T={T:.1f}K eta={eta*1e3:.1f}mPa.s | "
            f"static R_g low-q={low_rg:.0f}nm, "
            f"high-q={high_rg:.0f}nm "
            f"(DeltaBIC={sg['delta_bic']:.1f}, break={break_text}) | "
            f"R_h,z={an['R_h_z']*1e9:.0f}nm  C={an['C']:.2e}  "
            f"R_h low/high-q={an['R_h_low_q_median']*1e9:.0f}/"
            f"{an['R_h_high_q_median']*1e9:.0f}nm  "
            f"span={an['polydispersity_span']:.1f}  "
            f"PDI={an.get('PDI_mean', float('nan')):.3f}  "
            f"spearman(q,Rh)={an['spearman_q_Rh']:+.2f}")

    axA.set_xlabel(r"$q^2$ [$10^{14}$ m$^{-2}$]"); axA.set_ylabel(r"$\ln I(q)$")
    axA.set_title("(A) Segmented Guinier (static): two optically weighted sizes")
    axA.legend(fontsize=8)
    axB.set_xlabel(r"$q$ [$10^6$ m$^{-1}$]"); axB.set_ylabel(r"apparent $R_h$ [nm]")
    axB.set_title("(B) Initial-cumulant apparent $R_h$ vs q")
    axB.set_ylim(0, 1000)
    axB.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("guinier_demo.png", dpi=120)
    print("Saved guinier_demo.png\n")
    for ln in summary_lines:
        print(ln)
    print("\nNotes: DeltaBIC>10 only flags static curvature; two static populations")
    print("       are accepted only when both fitted Guinier slopes are negative and")
    print("       the low-q Rg is larger than the high-q Rg.")
    print("       Initial cumulants use g1_min=0.60 to avoid fitting late noisy tails;")
    print("       span>1 and spearman(q,Rh)<0 indicate a large slow population at low q.")
    print("       R_g (static) is viscosity-independent; the recorded eta is a FIXED")
    print("       34 mPa.s at both T (data-entry), so absolute 50C R_h still depends")
    print("       on correcting eta(50C) with demo_joint_temperature.py.")


if __name__ == "__main__":
    main()
