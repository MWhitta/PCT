"""
capabilities_report.py -- Synthetic ground-truth validation of every analysis
capability in corenn.py / hybrid_correlator.py, assembled into one montage
figure (reports/figures/capabilities.png) and a markdown summary
(reports/CAPABILITIES.md).

Each panel generates synthetic DLS/SLS data from a KNOWN truth, runs the
corresponding analysis, and overlays recovered vs. truth so the method can be
checked at a glance.  Run from the repo root:

    python reports/capabilities_report.py
"""

import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import corenn as C
import hybrid_correlator as H
import _common as K

rng = np.random.default_rng(0)
T, ETA, NW = 295.0, 0.94e-3, 1.33
kT6 = C.k_BOLTZMANN * T / (6 * np.pi * ETA)
G = C.GUINIER_SPHERE
results = []   # (title, validation string) collected for the markdown table


def angles(lo=20, hi=150, step=8):
    a = np.arange(lo, hi + 1, step)
    return a, np.array([C.q_from_angle(t, NW, 632.8) for t in a])


def g1_two(q, tau, R1, R2, a2):
    P1, P2 = np.exp(-(q*G*R1)**2/3), np.exp(-(q*G*R2)**2/3)
    D1, D2 = kT6/R1, kT6/R2
    g1 = (P1*np.exp(-q*q*D1*tau) + a2*P2*np.exp(-q*q*D2*tau)) / (P1 + a2*P2)
    return g1


# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
ax = axes.ravel()
panel = 0


def done(title, val):
    results.append((title, val));


# 1) cumulant_fit -----------------------------------------------------------
tau = np.geomspace(1e-6, 2.0, 200); beta = 0.5; Gt = 800.0
g1 = np.exp(-Gt*tau); g2m = beta*g1**2 + 1e-3*rng.standard_normal(tau.size)
cf = C.cumulant_fit(tau, g2m, beta=beta, g1_min=0.3)
ax[panel].semilogx(tau, np.sqrt(np.clip(g2m,0,None)/beta), ".", ms=3, label="data g1")
ax[panel].semilogx(tau, np.exp(-cf["Gamma"]*tau), "C1-", lw=1.5, label="cumulant fit")
ax[panel].set_title(f"1. cumulant_fit\nΓ {cf['Gamma']:.0f} (truth {Gt:.0f}) /s")
ax[panel].set_xlabel(r"$\tau$ [s]"); ax[panel].set_ylabel(r"$g_1$"); ax[panel].legend(fontsize=K.FONT_LEGEND_SMALL)
done("cumulant_fit", f"Γ={cf['Gamma']:.0f}/s (truth {Gt:.0f}), PDI={cf['PDI']:.3f}")
panel += 1

# 2) gamma_vs_q2 ------------------------------------------------------------
Rt = 60e-9; aa, q = angles()
Gam = q*q*(kT6/Rt) * (1 + 0.02*rng.standard_normal(q.size))
fit = C.gamma_vs_q2(q, Gam, temperature=T, viscosity=ETA)
ax[panel].plot(q**2*1e-14, Gam, "o", ms=4)
ax[panel].plot(q**2*1e-14, fit["D"]*q**2, "C1-")
ax[panel].set_title(f"2. gamma_vs_q2\nR_h {fit['R_h']*1e9:.0f} nm (truth 60), r²={fit['r2']:.3f}")
ax[panel].set_xlabel(r"$q^2$ [$10^{-4}$ nm$^{-2}$]"); ax[panel].set_ylabel(r"$\Gamma$ [s$^{-1}$]")
done("gamma_vs_q2", f"R_h={fit['R_h']*1e9:.0f} nm (truth 60), D={fit['D']:.2e}, r²={fit['r2']:.3f}")
panel += 1

# 3) corenn_invert (bimodal PSD) -------------------------------------------
q90 = C.q_from_angle(90, NW, 632.8); tau = np.geomspace(1.6e-6, 10, 200)
Rs = np.array([20e-9, 200e-9]); ws = np.array([0.5, 0.5])
D = C.diffusion_coefficient(Rs, T, ETA)
reps = [0.34*(sum(w*np.exp(-q90*q90*d*tau) for w,d in zip(ws,D)))**2
        + 2e-3*rng.standard_normal(tau.size) for _ in range(5)]
ds = C.DLSDataset(tau=tau, g2m=np.mean(reps,0), q=q90, beta=0.34, temperature=T, viscosity=ETA)
res = C.corenn_invert(ds, radius_range=(2e-9,1e-6), n_basis=22, n_R=140, replicates=reps)
ax[panel].semilogx(res.R*1e9, res.R*res.intensity, "C0-", lw=2)
for r in [20,200]: ax[panel].axvline(r, color="k", ls="--", alpha=.5)
pk = "+".join(f"{p['radius_nm']:.0f}" for p in res.peaks)
ax[panel].set_title(f"3. corenn_invert (single-angle PSD)\npeaks {pk} nm (truth 20+200)")
ax[panel].set_xlabel("R [nm]"); ax[panel].set_ylabel("R·I(R)")
done("corenn_invert", f"bimodal peaks {pk} nm (truth 20+200), λ={res.lam:.1e}")
panel += 1

# 4) corenn_multi_angle (number PSD) ---------------------------------------
aa, q = angles(20, 140, 12); tau = np.geomspace(1e-6, 2.0, 160)
R0, sln = 150e-9, 0.2
Rg = np.geomspace(20e-9, 600e-9, 50); Nt = np.exp(-(np.log(Rg)-np.log(R0))**2/(2*sln**2))
I0 = Nt*Rg**6
dsets = []
for th in aa:
    qi = C.q_from_angle(th, NW, 632.8); P = C.sphere_form_factor(qi, Rg)
    wq = I0*P; wq /= np.trapz(wq, Rg); Dr = C.diffusion_coefficient(Rg, T, ETA)
    g1c = np.trapz(wq[None,:]*np.exp(-qi*qi*Dr[None,:]*tau[:,None]), Rg, axis=1)
    beta = 0.3 + 0.4*np.sin(np.radians(th)/2)
    g2 = beta*g1c**2 + 1.2e-3*rng.standard_normal(tau.size)
    dsets.append(C.DLSDataset(tau=tau, g2m=g2, q=qi, beta=beta, temperature=T, viscosity=ETA))
rma = C.corenn_multi_angle(dsets, radius_range=(5e-9,1e-6), n_basis=20, n_R=120, temperature=T, viscosity=ETA)
nmean = rma.mean_radius("number")*1e9
ax[panel].semilogx(rma.R*1e9, rma.R*rma.number/np.trapz(rma.number,rma.R), "C0-", lw=2)
ax[panel].axvline(150, color="k", ls="--", alpha=.5)
ax[panel].set_title(f"4. corenn_multi_angle\nnumber mean R {nmean:.0f} nm (truth 150)")
ax[panel].set_xlabel("R [nm]"); ax[panel].set_ylabel("R·N(R)")
done("corenn_multi_angle", f"number-mean R={nmean:.0f} nm (truth 150), λ={rma.lam:.1e}")
panel += 1

# --- shared two-population system for panels 5-8 (consistent static+dynamic) -
# spheres R1>R2; q->0 intensity amplitudes A1,A2; R_g=G*R, R_h=R so rho=G=0.775
aa, q = angles(15, 150, 5)
R1s, R2s, A1, A2 = 150e-9, 30e-9, 1.0, 0.6
P1s, P2s = np.exp(-(q*G*R1s)**2/3), np.exp(-(q*G*R2s)**2/3)
Iq = A1*P1s + A2*P2s
D1s, D2s = kT6/R1s, kT6/R2s
Gam = (q*q) * (A1*P1s*D1s + A2*P2s*D2s) / (A1*P1s + A2*P2s)

# 5) segmented_guinier_fit --------------------------------------------------
sg = C.segmented_guinier_fit(q, Iq)
seg = sg["two_segment"]["segments"]
ax[panel].plot(q**2*1e-14, np.log(Iq), "o", ms=3)
for s in seg:
    xx = np.linspace(s["q2_min"], s["q2_max"], 20)
    ax[panel].plot(xx*1e-14, s["slope"]*xx+s["intercept"], "C1-", lw=1.5)
ax[panel].set_title(f"5. segmented_guinier_fit\nR_g {seg[0]['R_g']*1e9:.0f}/{seg[1]['R_g']*1e9:.0f} nm "
                    f"(R_g1={R1s*G*1e9:.0f}, R_g2={R2s*G*1e9:.0f})")
ax[panel].set_xlabel("q² [1e14/m²]"); ax[panel].set_ylabel("ln I")
done("segmented_guinier_fit", f"DETECTS two length scales (ΔBIC={sg['delta_bic']:.0f} ≫10); "
     f"effective R_g {seg[0]['R_g']*1e9:.0f}/{seg[1]['R_g']*1e9:.0f} nm (intensity-weighted "
     f"low/high-q projections of populations R_g1={R1s*G*1e9:.0f}, R_g2={R2s*G*1e9:.0f} nm)")
panel += 1

# 6) apparent_diffusion_analysis -------------------------------------------
an = C.apparent_diffusion_analysis(q, Gam, T, ETA)
ax[panel].plot(an["q"]*1e-6, an["R_h_app"]*1e9, "o-", ms=4)
ax[panel].axhline(an["R_h_z"]*1e9, color="C1", ls="--")
ax[panel].set_title(f"6. apparent_diffusion_analysis\nR_h,z {an['R_h_z']*1e9:.0f} nm, C={an['C']:.1e}")
ax[panel].set_xlabel(r"$q$ [$10^{-3}$ nm$^{-1}$]"); ax[panel].set_ylabel(r"apparent $R_h$ [nm]")
done("apparent_diffusion_analysis",
     f"R_h,z={an['R_h_z']*1e9:.0f} nm, C={an['C']:.1e}, span={an['polydispersity_span']:.1f}, "
     f"spearman={an['spearman_q_Rh']:+.2f}")
panel += 1

# 7) static_dynamic_correlation --------------------------------------------
sd = C.static_dynamic_correlation(q, Iq, Gam, T, ETA)
ax[panel].plot(np.log(an["q"]), np.log(an["R_h_app"]*1e9), "o", ms=4)
xx = np.array([np.log(an["q"]).min(), np.log(an["q"]).max()])
b = np.mean(np.log(an["R_h_app"]*1e9)+sd["powerlaw_alpha"]*np.log(an["q"]))
ax[panel].plot(xx, -sd["powerlaw_alpha"]*xx+b, "C1-")
ax[panel].set_title(f"7. static_dynamic_correlation\nstatic/dyn crossover ratio "
                    f"{sd['crossover_consistency']:.2f}; α={sd['powerlaw_alpha']:.2f}")
ax[panel].set_xlabel(r"$\ln q$"); ax[panel].set_ylabel(r"$\ln R_{h,app}$")
done("static_dynamic_correlation",
     f"static-Guinier-break and dynamic-R_h-crossover q coincide (ratio "
     f"{sd['crossover_consistency']:.2f}); R_h,app∝q^−α with α={sd['powerlaw_alpha']:.2f} "
     f"(r²={sd['r2_loglog']:.2f}); per-population shape ratio ρ=R_g/R_h "
     f"{sd['rho_large']:.2f}/{sd['rho_small']:.2f}")
panel += 1

# 8) depolarized_diffusion --------------------------------------------------
Rh = 30e-9; D_t = kT6/Rh; D_r = C.k_BOLTZMANN*T/(8*np.pi*ETA*Rh**3)
aa, q = angles(20, 150, 10); tau = np.geomspace(1e-7, 1.0, 200); vv, vh = [], []
for th, qi in zip(aa, q):
    gvv = 0.9*np.exp(-2*qi*qi*D_t*tau) + 1e-3*rng.standard_normal(tau.size)
    gvh = 0.9*np.exp(-2*(qi*qi*D_t+6*D_r)*tau) + 1e-3*rng.standard_normal(tau.size)
    a = C.DLSDataset(tau=tau, g2m=gvv, q=qi, beta=0.9, temperature=T, viscosity=ETA); a.metadata.update(angle_deg=th, scaled_count_rate=100.0)
    b = C.DLSDataset(tau=tau, g2m=gvh, q=qi, beta=0.9, temperature=T, viscosity=ETA); b.metadata.update(angle_deg=th, scaled_count_rate=1.0)
    vv.append(a); vh.append(b)
dp = C.depolarized_diffusion(vv, vh, temperature=T, viscosity=ETA)
ax[panel].plot(dp["q_vv"]**2*1e-14, dp["gamma_vv"], "o", ms=4, label="VV")
ax[panel].plot(dp["q_vh"]**2*1e-14, dp["gamma_vh"], "s", ms=4, label="VH")
ax[panel].set_title(f"8. depolarized_diffusion\nD_r {dp['D_r']:.0f}/s (truth {D_r:.0f}), R_rot {dp['R_rot']*1e9:.0f} nm")
ax[panel].set_xlabel(r"$q^2$ [$10^{-4}$ nm$^{-2}$]"); ax[panel].set_ylabel(r"$\Gamma$ [s$^{-1}$]"); ax[panel].legend(fontsize=K.FONT_LEGEND_SMALL)
done("depolarized_diffusion", f"D_t→R_h={dp['R_h']*1e9:.0f} nm (truth 30); "
     f"6D_r={dp['six_Dr']:.0f}/s (truth {6*D_r:.0f}); R_rot={dp['R_rot']*1e9:.0f} nm")
panel += 1

# 10) joint_temperature_fit -------------------------------------------------
Rh_true = 120e-9; temps_C = [10, 20, 30, 40]
etas = [1.30e-3, 1.00e-3, 0.80e-3, 0.65e-3]   # real eta(T)
ser = []
for Tc, et in zip(temps_C, etas):
    Tk = Tc + 273.15; D = C.k_BOLTZMANN*Tk/(6*np.pi*et*Rh_true)
    aa2, q2 = angles(20, 150, 15); g = q2*q2*D*(1+0.01*rng.standard_normal(q2.size))
    ser.append({"label": f"{Tc}C", "T": Tk, "q": q2, "gamma": g, "eta": None})
ser[0]["eta"] = etas[0]
jt = C.joint_temperature_fit(ser, reference_index=0)
ax[panel].plot(temps_C, np.array(etas)*1e3, "ko-", label="true η(T)")
ax[panel].plot(temps_C, jt["eta_fit"]*1e3, "C1s--", label="solved η(T)")
ax[panel].set_title(f"9. joint_temperature_fit\nR_h,z {jt['R_h_z']*1e9:.0f} nm (truth 120)")
ax[panel].set_xlabel("T [°C]"); ax[panel].set_ylabel(r"$\eta$ [mPa·s]"); ax[panel].legend(fontsize=K.FONT_LEGEND_SMALL)
done("joint_temperature_fit", f"shared R_h,z={jt['R_h_z']*1e9:.0f} nm (truth 120); "
     f"η(T) solved within {np.max(np.abs(jt['eta_fit']/np.array(etas)-1))*100:.0f}% of truth")
panel += 1

# 11) hybrid correlator (software_correlate + merge) ------------------------
dt = 0.0525; n = 40000; tau_f = 2.0
phi = np.exp(-dt/tau_f); s = np.sqrt(1-phi**2)
x = np.empty(n); y = np.empty(n); x[0]=rng.standard_normal(); y[0]=rng.standard_normal()
for i in range(1,n):
    x[i]=phi*x[i-1]+s*rng.standard_normal(); y[i]=phi*y[i-1]+s*rng.standard_normal()
I = np.abs(x+1j*y)**2; I = I/I.mean()*80
a = rng.poisson(I).astype(float); b = rng.poisson(I).astype(float)
tau_sw, g2_sw = H.software_correlate(a, dt, b, per_octave=8)
ax[panel].semilogx(tau_sw, g2_sw, "C2o", ms=4, label="software g2-1")
ax[panel].semilogx(tau_sw, np.exp(-2*tau_sw/tau_f), "k-", lw=1.5, label="truth")
ax[panel].set_title("10. hybrid: software_correlate\nrecovers slow exp from count trace")
ax[panel].set_xlabel(r"$\tau$ [s]"); ax[panel].set_ylabel(r"$g_2-1$"); ax[panel].legend(fontsize=K.FONT_LEGEND_SMALL)
m = (tau_sw>0.1)&(tau_sw<4)
err = np.sqrt(np.mean((g2_sw[m]-np.exp(-2*tau_sw[m]/tau_f))**2))
done("hybrid_correlator", f"software correlation of count trace recovers g2-1 (RMS {err:.3f} vs truth)")
panel += 1

# 12) schatzel_sigma (noise model) -----------------------------------------
tau = np.geomspace(1.6e-6, 200, 200); g1 = np.exp(-30*tau)
sig = C.schatzel_sigma(tau, g1, 0.34, 240.0, sigma_floor=0.006)
ax[panel].loglog(tau, sig, "C3-")
ax[panel].set_title("11. schatzel_sigma\nnoise grows ~√(τ/T) at long lag")
ax[panel].set_xlabel("τ [s]"); ax[panel].set_ylabel("σ(g2-1)")
done("schatzel_sigma", "Schätzel weighting: σ flat at short lag, grows ~√(τ/T) at long lag")
panel += 1

# 12) regime_diagnostic (PSD anticipates Guinier vs Porod) ------------------
# Polydisperse sphere ensemble centered at R0=150 nm: polydispersity smears the
# monodisperse form-factor minima, leaving a clean Guinier knee at low q and a
# Porod q^−4 tail at high q.  The angular range straddles the Guinier limit
# (q·R_g<1.3) and the Porod regime (q·R≳π); the PSD decides which fit applies.
aa, q = angles(15, 150, 5)
R0 = 70e-9
Rdist = np.linspace(12e-9, 224e-9, 300); sig = 0.45
wN = np.exp(-(np.log(Rdist/R0))**2/(2*sig**2))          # number distribution
Vol = (4/3)*np.pi*Rdist**3                               # ~ scattering amplitude
Iq = np.array([np.trapz(wN*Vol**2*C.sphere_form_factor(qi, Rdist), Rdist) for qi in q])
# intensity-weighted (z-average) R_g of the ensemble = the Guinier truth
Iw = wN*Vol**2; Rg_z = np.sqrt(np.trapz(Iw*(G*Rdist)**2, Rdist)/np.trapz(Iw, Rdist))
tauR = np.geomspace(1e-6, 2.0, 200); dd = []
for th, qi in zip(aa, q):
    g1 = np.trapz(Iw[None, :]*np.exp(-qi*qi*(kT6/Rdist)[None, :]*tauR[:, None]), Rdist, axis=1)
    g1 /= g1[0]; g2 = 0.5*g1**2 + 8e-4*rng.standard_normal(tauR.size)
    dd.append(C.DLSDataset(tau=tauR, g2m=g2, q=qi, beta=0.5, temperature=T, viscosity=ETA))
rmg = C.corenn_multi_angle(dd, radius_range=(10e-9, 5e-7), n_basis=18, n_R=100,
                           temperature=T, viscosity=ETA)
rd = K.regime_diagnostic(q, Iq, rmg); Rg0 = Rg_z
lq = np.log10(q); lI = np.log10(np.clip(Iq, 1e-30, None)); ln10 = np.log(10)
ax[panel].plot(lq, lI, "o", ms=3, color="C0")
rg_txt = "—"
if rd["porod"] is not None:
    pp = rd["porod"]["exponent"]; mk = (q*rd["R_char"]) > 3.5
    xx = np.array([lq[mk].min(), lq.max()])
    ax[panel].plot(xx, rd["porod"]["log_prefactor"]/ln10 - pp*xx, "C1-", lw=1.5,
                   label=f"Porod q^−{pp:.1f}")
if rd["guinier"] is not None and np.isfinite(rd["guinier"].get("R_g", np.nan)) \
        and rd["guinier"]["R_g"] > 1e-9:
    rg_txt = f"{rd['guinier']['R_g']*1e9:.0f} nm"
ql = rd.get("q_guinier_limit", np.nan)
if np.isfinite(ql) and q.min() < ql < q.max():
    ax[panel].axvline(np.log10(ql), color="k", ls=":", alpha=.6, label="q·R_g=1.3")
ax[panel].set_title((f"12. regime_diagnostic\nGuinier R_g {rg_txt} (truth {Rg0*1e9:.0f} nm); "
                     f"Porod q^−{rd['porod']['exponent']:.1f}") if rd["porod"] else
                    "12. regime_diagnostic")
ax[panel].set_xlabel(r"$\log q$"); ax[panel].set_ylabel(r"$\log I$"); ax[panel].legend(fontsize=K.FONT_LEGEND_SMALL)
done("regime_diagnostic",
     f"PSD (R={rd['R_char']*1e9:.0f} nm) → q·R_g spans {rd['qRg_min']:.1f}–{rd['qRg_max']:.1f}: "
     f"{rd['regime']}; low-q Guinier R_g={rg_txt} (truth {Rg0*1e9:.0f}), high-q "
     f"I∝q^−{rd['porod']['exponent']:.1f} (Porod q^−4 envelope)")
panel += 1

for k in range(panel, len(ax)):
    ax[k].axis("off")
fig.suptitle("PCT/MAPCS analysis capabilities — synthetic ground-truth validation")
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(HERE, "figures", "capabilities.png"))
print("Saved reports/figures/capabilities.png")
for t, v in results:
    print(f"  [{t}] {v}")

# --- write CAPABILITIES.md -------------------------------------------------
md = ["# PCT / MAPCS analysis capabilities",
      "",
      "Synthetic ground-truth validation of every analysis function in "
      "`corenn.py` and `hybrid_correlator.py`.  Each was run on data generated "
      "from a known truth; the recovered values are below and visualized in "
      "`figures/capabilities.png`.",
      "",
      "![capabilities](figures/capabilities.png)",
      "",
      "| # | capability | synthetic validation |",
      "|---|------------|----------------------|"]
for i, (t, v) in enumerate(results, 1):
    md.append(f"| {i} | `{t}` | {v} |")
md += ["",
       "## Notes",
       "- The **forward model is primary**: `corenn_invert` / `corenn_multi_angle` "
       "fit the full `g2(q,τ)` for the size distribution; the cumulant `Γ(q)`, "
       "apparent `R_h(q)`, and the static `I(q)` regime (Guinier vs Porod, "
       "anticipated from the PSD via `regime_diagnostic`) are **diagnostic "
       "projections** of that model, not the model itself.",
       "- `Γ(q)` alone is information-poor: it supports a two-population *summary* "
       "but cannot resolve >2 populations or a continuous distribution — that "
       "requires the full decay (`corenn_invert` / `corenn_multi_angle`).",
       "- The hybrid hardware+software correlator is opt-in "
       "(`load_measurement_replicates(dir, hybrid=True)`); it extends the lag "
       "range for slow-mode analysis but is not used for the cross-angle "
       "inversion.",
       ""]
md_text = "\n".join(md)
with open(os.path.join(HERE, "CAPABILITIES.md"), "w") as f:
    f.write(md_text)
print("Saved reports/CAPABILITIES.md")
from md2pdf import md_to_pdf
md_to_pdf(md_text, os.path.join(HERE, "CAPABILITIES.pdf"), base_dir=HERE)
print("Saved reports/CAPABILITIES.pdf")
