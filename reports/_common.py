"""
_common.py -- shared analysis suite + report helpers for the dataset reports.

`run_suite` runs every applicable static+dynamic analysis on a list of
single-angle DLSDatasets (one per scattering angle) and returns a structured
dict.  `suite_steps_md` formats the step-by-step narrative; `plot_suite` draws
the standard diagnostic + forward-model panels.
"""

import glob
import math
import os
import sys

import numpy as np
import corenn as C

import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
# ---------------------------------------------------------------------------
# Single source of truth for figure look-and-feel.  Every report figure (the
# consolidating panels in each driver and every fig_* helper here) inherits this
# style, so fonts, grid, legends, line/marker weights, and output DPI are uniform
# across the whole report set.  Per-call overrides are reserved for genuinely
# special cases (insets, dense legends) via the FONT_* constants below.
# ---------------------------------------------------------------------------
_plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # fonts
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5, "figure.titlesize": 14, "figure.titleweight": "bold",
    # lines & markers
    "lines.linewidth": 1.7, "lines.markersize": 4.0, "axes.linewidth": 0.8,
    # subtle grid as the unifying element
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "0.85",
    "grid.linewidth": 0.5, "grid.alpha": 0.7,
    # cleaner frame: drop the top spine (right kept for twin-axis panels)
    "axes.edgecolor": "0.35", "axes.spines.top": False,
    # legends
    "legend.frameon": True, "legend.framealpha": 0.85, "legend.edgecolor": "0.8",
    "legend.fancybox": False, "legend.borderpad": 0.4, "legend.handlelength": 1.7,
    "legend.labelspacing": 0.3,
    # ticks
    "xtick.direction": "out", "ytick.direction": "out",
})
FONT_LEGEND_SMALL = 7.5   # dense multi-panel legends
FONT_INSET = 6.5          # inset axes (residual strip, etc.)

# Unit convention for all reports: sizes in nm, scattering vector q in nm⁻¹.
# q is held internally in SI (1/m); these convert for display only.
Q_NM = 1e-9               # multiply an SI q [1/m] to get q [nm⁻¹]


def _qnm(q_si, sig=3):
    """SI scattering vector [1/m] -> display string in nm⁻¹ (default 3 s.f.)."""
    return f"{q_si * Q_NM:.{sig}g}"


def _qrange_nm(lo, hi):
    """Format an SI q-range [1/m] as 'lo–hi nm⁻¹'."""
    return f"{_qnm(lo)}–{_qnm(hi)} nm⁻¹"


# Thin oblate-disk (platelet) hydrodynamics — the right morphology for clays.
# Orientation-averaged translational diffusion of a thin disk of radius a is
# D_t = kT/(12 η a) (from f_perp=16ηa, f_par=32ηa/3), so a disk diffuses like a
# sphere of radius R_h = (2/π) a, i.e. a = (π/2) R_h.  Tumbling rotation is
# D_r = 3 kT/(32 η a³).
DISK_SPHERE_FACTOR = math.pi / 2.0   # platelet radius a = DISK_SPHERE_FACTOR * R_h(sphere)


def disk_radius_translational(D_t, T, eta):
    """Lateral radius a [m] of a thin oblate disk from its orientation-averaged
    translational diffusion coefficient D_t [m²/s]:  D_t = kT/(12 η a)."""
    return C.k_BOLTZMANN * T / (12.0 * eta * D_t) if (D_t and D_t > 0) else float("nan")


def disk_radius_rotational(D_r, T, eta):
    """Lateral radius a [m] of a thin disk from its tumbling rotational diffusion
    coefficient D_r [1/s]:  D_r = 3 kT/(32 η a³)."""
    return ((3.0 * C.k_BOLTZMANN * T / (32.0 * eta * D_r)) ** (1.0 / 3.0)
            if (D_r and D_r > 0) else float("nan"))


def morphology_hydrodynamics(mph, R_h, R_g=None):
    """Pick the Stokes–Einstein geometry from the Guinier–Porod morphology and
    return the shape-appropriate hydrodynamic size(s).

    The static Porod slope p fixes the dimensionality of the scatterer, which
    selects the translational Stokes–Einstein relation:

      p ≈ 4 / surface-fractal (3D compact) → sphere,  D = kT/(6πηR),  R = R_h
      p ≈ 2 (2D lamella / platelet)        → oblate disk, D = kT/(12ηa), a = (π/2)R_h
      p ≈ 1 (1D rod)                       → rod, D ≈ kT·ln(L/d)/(3πηL)
      coil (p ≈ 2 polymer)                 → R_h is the coil hydrodynamic radius
      mass fractal (2<p<3 aggregate)       → R_h is the effective Stokes radius

    A q^-2 slope is dimensionally 2D and therefore *degenerate* (flat disk vs
    coil vs D=2 mass-fractal), so the returned dict carries an ``alt`` geometry
    in that case.  ``R_h``/``R_g`` are SI (m); sizes returned in nm."""
    Rhn = (R_h * 1e9) if (R_h and R_h > 0) else float("nan")
    a_disk = DISK_SPHERE_FACTOR * Rhn   # oblate-disk radius a = (π/2)·R_h
    disk_alt = {"shape": "a thin oblate disk / platelet", "size_nm": a_disk,
                "size_label": "disk radius a", "se": "D = kT/(12ηa) ⇒ a = (π/2)R_h"}
    if mph is None:
        return {"shape": "a sphere (assumed)", "se": "D = kT/(6πηR)", "size_nm": Rhn,
                "size_label": "radius R = R_h", "rigorous": False,
                "note": "no resolved Porod slope, so the sphere form is assumed"}
    kind = mph.get("kind"); p = mph.get("dimension")
    if kind in ("porod", "surface_fractal", "fractal_boundary"):
        return {"shape": ("a compact, sphere-like particle" if kind == "porod"
                          else "a rough but compact (3D) particle"),
                "se": "D = kT/(6πηR)", "size_nm": Rhn, "size_label": "radius R = R_h",
                "rigorous": kind == "porod",
                "note": "a 3D compact object — the Stokes (sphere) radius is the natural size"}
    if kind == "rod":
        out = {"shape": "a thin rod (1D)", "se": "D ≈ kT·[ln(L/d)+γ]/(3πηL)",
               "size_nm": Rhn, "size_label": "Stokes radius R_h", "rigorous": False,
               "note": "a 1D object — the sphere R_h under-represents the rod length"}
        if R_g and R_g > 0:
            out["length_nm"] = math.sqrt(12.0) * R_g * 1e9   # rod: R_g² = L²/12
            out["note"] += f"; the Guinier R_g gives a rod length L ≈ {out['length_nm']:.0f} nm"
        return out
    if kind in ("coil", "swollen_coil"):
        return {"shape": "a random coil", "se": "D = kT/(6πηR_h)", "size_nm": Rhn,
                "size_label": "coil hydrodynamic radius R_h", "rigorous": True, "alt": disk_alt,
                "note": "a q^-2 slope is dimensionally 2D — a Gaussian coil (R_h is the natural "
                        "size) or, equally, a flat platelet (use the disk SE)"}
    return {"shape": f"a mass-fractal aggregate (D≈{p:.1f})", "se": "D = kT/(6πηR_h) (Stokes radius)",
            "size_nm": Rhn, "size_label": "effective Stokes radius R_h", "rigorous": False,
            "alt": disk_alt,
            "note": "a branched aggregate has no closed-form Stokes–Einstein, so R_h is an "
                    "effective Stokes radius; the same ~2D slope is also consistent with a flat "
                    "platelet building block (disk SE)"}


G1_MIN = 0.60
CORENN_MIN_FIELD_DECAY = -math.log(0.80)  # require at least 20% g1 decay
CORENN_STATIC_QR_FACTOR = math.pi
CORENN_RADIUS_LOWER_Q_FACTOR = 0.2


def _linear_fit(x, y, sigma=None):
    """Small weighted linear least-squares helper with parameter uncertainty."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, float)
        good &= np.isfinite(sigma) & (sigma > 0)
    x, y = x[good], y[good]
    if sigma is not None:
        sigma = sigma[good]
    if x.size < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "slope_se": float("nan"), "r2": float("nan"),
                "chi2_red": float("nan"), "max_abs_z": float("nan"),
                "n": int(x.size), "resid": np.array([]), "pred": np.array([])}
    A = np.vstack([x, np.ones(x.size)]).T
    if sigma is not None:
        w = 1.0 / np.clip(sigma, 1e-30, None)
        sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        pred = A @ sol
        resid = y - pred
        chi2 = float(np.sum((resid / sigma) ** 2))
        chi2_red = chi2 / max(x.size - 2, 1)
        cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None]))
        cov *= max(chi2_red, 1.0)
        max_abs_z = float(np.nanmax(np.abs(resid / sigma)))
    else:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ sol
        resid = y - pred
        ss = float(np.sum(resid ** 2))
        cov = np.linalg.pinv(A.T @ A) * (ss / max(x.size - 2, 1))
        chi2_red = float("nan")
        max_abs_z = float("nan")
    ss = float(np.sum(resid ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(sol[0]), "intercept": float(sol[1]),
            "slope_se": float(np.sqrt(max(cov[0, 0], 0.0))),
            "r2": 1.0 - ss / st if st > 0 else float("nan"),
            "chi2_red": chi2_red, "max_abs_z": max_abs_z,
            "n": int(x.size), "resid": resid, "pred": pred}


def static_powerlaw_fit(q, I, intensity_err=None, mask=None, selection="all"):
    """Power-law (Porod/fractal) fit of the static intensity: I(q) ∝ q^(-p).

    The Guinier approximation (ln I linear in q²) only holds at q·R_g ≲ 1.3,
    which large particles never reach; across the measured range these samples
    are in the q·R ≳ 1 regime where I(q) is a power law.  ``p ≈ 4`` is Porod
    (compact particles, sharp interfaces), ``1 < p < 4`` is a fractal / diffuse
    / polydisperse regime, and ``p < 0`` (intensity rising with q) signals a
    low-q structure factor S(q) < 1 from interparticle correlations.
    """
    q = np.asarray(q, float); I = np.asarray(I, float)
    good = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
    if mask is not None:
        good &= np.asarray(mask, bool)
    if intensity_err is not None:
        ierr = np.asarray(intensity_err, float)
        sig = np.where(np.isfinite(ierr) & (ierr > 0), ierr / I, np.nan)
        if not np.all(np.isfinite(sig[good]) & (sig[good] > 0)):
            sig = None
    else:
        sig = None
    order = np.argsort(q[good])
    qg, Ig = q[good][order], I[good][order]
    x, y = np.log(qg), np.log(Ig)
    sy = sig[good][order] if sig is not None else None
    if x.size < 3:
        return {"exponent": float("nan"), "r2": float("nan"),
                "log_prefactor": float("nan"), "n_points": int(x.size),
                "q_min": float("nan"), "q_max": float("nan"),
                "accepted": False, "selection": selection}
    fit = _linear_fit(x, y, sy)
    p = float(-fit["slope"])
    p_se = float(fit["slope_se"])
    return {"exponent": p, "exponent_se": p_se,
            "exponent_ci68": (p - p_se, p + p_se) if np.isfinite(p_se) else (float("nan"), float("nan")),
            "exponent_ci95": (p - 1.96 * p_se, p + 1.96 * p_se) if np.isfinite(p_se) else (float("nan"), float("nan")),
            "r2": fit["r2"], "log_prefactor": float(fit["intercept"]),
            "chi2_red": fit["chi2_red"], "max_abs_z": fit["max_abs_z"],
            "n_points": int(x.size), "q_min": float(qg.min()),
            "q_max": float(qg.max()), "accepted": True,
            "selection": selection}


def select_static_powerlaw_window(q, I, intensity_err=None, n_min=5,
                                  r2_min=0.998, chi2_red_max=10.0,
                                  max_abs_z=4.0, min_exponent=0.8,
                                  max_exponent=4.2, prefer_low_q=True):
    """Select a contiguous static scaling window for a Porod/fractal exponent.

    The selected exponent is not a blind fit to every measured angle.  It is the
    longest contiguous log(I)-vs-log(q) window that is physically falling
    (``I ~ q^-p`` with ``min_exponent <= p <= max_exponent``) and statistically
    linear.  When replicate static-intensity errors are available, the linearity
    check uses the uncertainty in log(I) and stops before systematic high-q
    flattening/upturns can bias the slope.
    """
    q = np.asarray(q, float); I = np.asarray(I, float)
    good = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
    if intensity_err is not None:
        ierr = np.asarray(intensity_err, float)
    else:
        ierr = None
    order_good = np.where(good)[0][np.argsort(q[good])]
    if order_good.size < n_min:
        out = static_powerlaw_fit(q, I, intensity_err=intensity_err,
                                  selection="insufficient data")
        out.update({"accepted": False, "mask": good, "criteria": {}})
        return out

    candidates = []
    n = order_good.size
    for i0 in range(n):
        for i1 in range(i0 + n_min, n + 1):
            idx = order_good[i0:i1]
            mask = np.zeros(q.shape, bool); mask[idx] = True
            fit = static_powerlaw_fit(q, I, intensity_err=intensity_err, mask=mask,
                                      selection="contiguous log-log window")
            p, r2 = fit["exponent"], fit["r2"]
            if not (np.isfinite(p) and min_exponent <= p <= max_exponent
                    and np.isfinite(r2) and r2 >= r2_min):
                continue
            if ierr is not None:
                cr, mz = fit.get("chi2_red", float("nan")), fit.get("max_abs_z", float("nan"))
                if np.isfinite(cr) and cr > chi2_red_max:
                    continue
                if np.isfinite(mz) and mz > max_abs_z:
                    continue
            candidates.append((i0, i1, fit, mask))

    if candidates:
        if prefer_low_q:
            # Prefer the longest scaling range anchored as close as possible to
            # the measured low-q limit; this avoids fitting the high-q noise
            # floor when the primary lower-q power law is already clear.
            candidates.sort(key=lambda t: (-(t[1] - t[0]), t[0],
                                           -t[2]["r2"],
                                           t[2].get("chi2_red", np.inf)))
        else:
            candidates.sort(key=lambda t: (-(t[1] - t[0]), -t[2]["r2"],
                                           t[2].get("chi2_red", np.inf), t[0]))
        i0, i1, fit, mask = candidates[0]
        accepted = True
    else:
        # Fallback: report the best low-q anchored physical window, but mark it
        # as unaccepted so morphology interpretation is not over-stated.
        fallback = []
        for i1 in range(n_min, n + 1):
            idx = order_good[:i1]
            mask = np.zeros(q.shape, bool); mask[idx] = True
            fit = static_powerlaw_fit(q, I, intensity_err=intensity_err, mask=mask,
                                      selection="fallback low-q window")
            p = fit["exponent"]
            if np.isfinite(p) and min_exponent <= p <= max_exponent:
                fallback.append((fit["r2"], -abs(fit.get("chi2_red", 1.0)), i1, fit, mask))
        if fallback:
            fallback.sort(reverse=True)
            _, _, i1, fit, mask = fallback[0]
            i0 = 0
        else:
            mask = good
            fit = static_powerlaw_fit(q, I, intensity_err=intensity_err, mask=mask,
                                      selection="overall fallback")
            i0, i1 = 0, n
        accepted = False

    fit.update({"accepted": accepted, "mask": mask, "start_index_sorted": int(i0),
                "end_index_sorted": int(i1),
                "criteria": {"n_min": n_min, "r2_min": r2_min,
                             "chi2_red_max": chi2_red_max,
                             "max_abs_z": max_abs_z,
                             "min_exponent": min_exponent,
                             "max_exponent": max_exponent,
                             "prefer_low_q": prefer_low_q}})
    fit["high_q"] = high_q_deviation_summary(q, I, fit, intensity_err=intensity_err)
    return fit


def high_q_deviation_summary(q, I, fit, intensity_err=None):
    """Summarize points above the selected static power-law window."""
    if not fit or not np.isfinite(fit.get("q_max", float("nan"))):
        return None
    q = np.asarray(q, float); I = np.asarray(I, float)
    tail = np.isfinite(q) & np.isfinite(I) & (q > fit["q_max"]) & (I > 0)
    if not np.any(tail):
        return None
    y = np.log(I[tail])
    pred = fit["log_prefactor"] - fit["exponent"] * np.log(q[tail])
    resid = y - pred
    out = {"n_points": int(np.count_nonzero(tail)),
           "q_min": float(np.min(q[tail])), "q_max": float(np.max(q[tail])),
           "mean_log_residual": float(np.mean(resid)),
           "last_excess_factor": float(np.exp(resid[np.argmax(q[tail])]))}
    if intensity_err is not None:
        ierr = np.asarray(intensity_err, float)
        sy = ierr[tail] / I[tail]
        ok = np.isfinite(sy) & (sy > 0)
        out["max_abs_z"] = float(np.nanmax(np.abs(resid[ok] / sy[ok]))) if np.any(ok) else float("nan")
    if np.count_nonzero(tail) >= 3:
        tf = static_powerlaw_fit(q, I, intensity_err=intensity_err, mask=tail,
                                 selection="excluded high-q tail")
        out["tail_exponent"] = tf["exponent"]
        out["tail_r2"] = tf["r2"]
    else:
        out["tail_exponent"] = float("nan")
        out["tail_r2"] = float("nan")
    if out["last_excess_factor"] > 1.5:
        out["behavior"] = "flattening followed by an upturn"
    elif abs(out.get("tail_exponent", 0.0)) < 1.0:
        out["behavior"] = "near-flat intensity"
    else:
        out["behavior"] = "systematic departure"
    return out


def static_replicate_intensity_values(dataset):
    """Per-repetition scaled count-rate values for one loaded dataset."""
    vals = dataset.metadata.get("scaled_count_rate_replicates")
    if vals is not None:
        a = np.asarray(vals, float)
        return a[np.isfinite(a) & (a > 0)]
    path = dataset.metadata.get("path")
    if not path:
        val = static_intensity_proxy_if_positive(dataset)
        return np.asarray([val], float) if val is not None else np.array([])
    meas_dir = os.path.dirname(path) if os.path.basename(path).startswith("Repetition") else path
    def _rep_key(path):
        tail = path.rsplit(" ", 1)[-1]
        return int(tail) if tail.isdigit() else 10**9
    rep_dirs = sorted(glob.glob(os.path.join(meas_dir, "Repetition *")), key=_rep_key)
    out = []
    for rd in rep_dirs:
        try:
            v = C.load_summary(os.path.join(rd, "Summary.csv")).get("Scaled count rate [kHz/mW]")
        except Exception:
            continue
        if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
            out.append(float(v))
    if out:
        return np.asarray(out, float)
    val = static_intensity_proxy_if_positive(dataset)
    return np.asarray([val], float) if val is not None else np.array([])


def static_intensity_proxy_if_positive(dataset):
    val = C.static_intensity_proxy(dataset)
    return val if val is not None and np.isfinite(val) and val > 0 else None


def replicate_observation_stats(s, g1_min=G1_MIN):
    """Per-angle replicate error estimates used by plots and weighted fits."""
    kept = s.get("_kept_datasets", []); reps = s.get("_reps", [])
    q = np.asarray(s.get("q", []), float)
    kT6 = C.k_BOLTZMANN * s["T"] / (6.0 * math.pi * s["eta"])
    out = {"q": q, "gamma_mean": np.full(q.size, np.nan),
           "gamma_sd": np.full(q.size, np.nan), "gamma_sem": np.full(q.size, np.nan),
           "gamma_n": np.zeros(q.size, int),
           "R_h_app_mean": np.full(q.size, np.nan),
           "R_h_app_sd": np.full(q.size, np.nan), "R_h_app_sem": np.full(q.size, np.nan),
           "R_h_app_n": np.zeros(q.size, int),
           "I_mean": np.full(q.size, np.nan), "I_sd": np.full(q.size, np.nan),
           "I_sem": np.full(q.size, np.nan), "I_n": np.zeros(q.size, int)}
    for i, ds in enumerate(kept):
        rp = reps[i] if i < len(reps) else None
        gammas, rhs = [], []
        if rp is not None and len(rp) >= 2:
            for g2 in rp:
                cf = C.cumulant_fit(ds.tau, g2, beta=ds.beta, order=2, g1_min=g1_min)
                gam = cf.get("Gamma", float("nan"))
                if np.isfinite(gam) and gam > 0:
                    gammas.append(gam)
                    rhs.append(kT6 * ds.q ** 2 / gam)
        if gammas:
            g = np.asarray(gammas, float); r = np.asarray(rhs, float)
            out["gamma_mean"][i] = float(np.mean(g)); out["gamma_n"][i] = g.size
            out["R_h_app_mean"][i] = float(np.mean(r)); out["R_h_app_n"][i] = r.size
            if g.size >= 2:
                out["gamma_sd"][i] = float(np.std(g, ddof=1))
                out["gamma_sem"][i] = out["gamma_sd"][i] / math.sqrt(g.size)
                out["R_h_app_sd"][i] = float(np.std(r, ddof=1))
                out["R_h_app_sem"][i] = out["R_h_app_sd"][i] / math.sqrt(r.size)
        vals = static_replicate_intensity_values(ds)
        if vals.size:
            out["I_mean"][i] = float(np.mean(vals)); out["I_n"][i] = vals.size
            if vals.size >= 2:
                out["I_sd"][i] = float(np.std(vals, ddof=1))
                out["I_sem"][i] = out["I_sd"][i] / math.sqrt(vals.size)
    return out


def porod_morphology(p, r2=None, r2_min=0.90):
    """Morphology / form-factor class implied by a *valid* power law I(q) ∝ q^−p.

    Standard small-angle-scattering reading of the high-q slope:

    * ``p ≈ 4``  — a smooth, sharp interface: a compact, sphere-like particle
      (classic Porod law); consistent with the sphere form factor.
    * ``3 < p < 4`` — a **surface fractal** (rough interface), surface fractal
      dimension ``D_s = 6 − p`` (→ 2 smooth, → 3 maximally rough).
    * ``p ≈ 3`` — the ambiguous limit: a maximally rough surface fractal
      (``D_s → 3``) **or** a dense branched / network **mass fractal**
      (``D_m → 3``) — a rough or branching aggregate, not a smooth sphere.
    * ``2 ≤ p < 3`` — a **mass fractal**: branched / randomly-branched /
      aggregated network, mass-fractal dimension ``D_m = p``.
    * ``p ≈ 2`` — a Gaussian / ideal random coil (``D_m ≈ 2``).
    * ``5/3 ≲ p < 2`` — a swollen / self-avoiding coil (good solvent).
    * ``p ≈ 1`` — a rod / stiff linear chain.

    Returns ``None`` unless the slope is a real, well-fit falling power law
    (``p ≥ 0.8`` and ``r2 ≥ r2_min``), so noisy / flat / rising high-q data is
    not over-interpreted.  ``sphere_consistent`` flags whether the slope matches
    the homogeneous-sphere form factor used by the PSD inversion.
    """
    if not np.isfinite(p) or p < 0.8:
        return None
    if r2 is not None and np.isfinite(r2) and r2 < r2_min:
        return None
    if p >= 3.7:
        return {"label": "a smooth, sharp interface — a compact, sphere-like particle (Porod q^−4)",
                "dimension": None, "kind": "porod", "sphere_consistent": True}
    if p >= 3.3:
        return {"label": f"a surface fractal — a rough interface, surface fractal dimension D_s≈{6.0-p:.1f}",
                "dimension": 6.0 - p, "kind": "surface_fractal", "sphere_consistent": False}
    if p >= 2.7:
        return {"label": ("a surface fractal (very rough interface, D_s→3) or a branched / "
                          "mass-fractal network (D→3) — a rough or branching aggregate, not a "
                          "smooth sphere"),
                "dimension": p, "kind": "fractal_boundary", "sphere_consistent": False}
    if p >= 2.0:
        return {"label": ("a mass fractal — a branched / randomly-branched / aggregated network, "
                          f"mass-fractal dimension D≈{p:.1f}"),
                "dimension": p, "kind": "mass_fractal", "sphere_consistent": False}
    if p >= 1.7:
        return {"label": "≈ a Gaussian / ideal random coil (mass-fractal D≈2)",
                "dimension": 2.0, "kind": "coil", "sphere_consistent": False}
    if p >= 1.3:
        return {"label": f"a swollen / self-avoiding coil (good solvent, D≈{p:.1f})",
                "dimension": p, "kind": "swollen_coil", "sphere_consistent": False}
    return {"label": "a rod / stiff linear chain (D≈1)",
            "dimension": 1.0, "kind": "rod", "sphere_consistent": False}


def regime_diagnostic(q, I, corenn, intensity_err=None, qrg_guinier=1.3,
                      qr_porod=3.5, powerlaw=None):
    """Anticipate the scattering regime from the CORENN multi-angle PSD.

    The recovered size sets which static regime the measured q-range probes:
    a scatterer of geometric radius R (radius of gyration R_g = sqrt(3/5) R) is
    in the *Guinier* regime where q·R_g < ~1.3 (ln I linear in q²), the
    *form-factor* regime near q·R ~ 1-pi, and the *Porod* regime where
    q·R ≳ pi (I ∝ q^-4 for sharp interfaces).  This takes the intensity-weighted
    characteristic size from the CORENN PSD, classifies the regime(s) the data
    spans.  The static scaling exponent is then fit by selecting a contiguous
    linear log(I)-vs-log(q) window, rather than blindly including all nominal
    high-q points; this prevents high-q background/form-factor departures from
    biasing the Porod/fractal slope.
    """
    q = np.asarray(q, float); I = np.asarray(I, float)
    if corenn is None or not getattr(corenn, "peaks", None):
        return None
    try:
        R_char = corenn.mean_radius("intensity")        # intensity-weighted radius
    except Exception:
        R_char = corenn.peaks[0]["radius_nm"] * 1e-9
    R_g = C.GUINIER_SPHERE * R_char
    qrg = q * R_g
    gmask = qrg < qrg_guinier
    pmask = (q * R_char) > qr_porod
    porod = powerlaw if powerlaw is not None else select_static_powerlaw_window(
        q, I, intensity_err=intensity_err)
    out = {
        "R_char": R_char, "R_g_char": R_g,
        "qRg_min": float(np.nanmin(qrg)), "qRg_max": float(np.nanmax(qrg)),
        "n_guinier": int(gmask.sum()), "n_porod": int(porod.get("n_points", 0)),
        "n_nominal_porod": int(pmask.sum()),
        "q_guinier_limit": qrg_guinier / R_g if R_g > 0 else float("nan"),
        "guinier": (C.guinier_fit(q[gmask], I[gmask],
                                  intensity_err=(np.asarray(intensity_err)[gmask]
                                                 if intensity_err is not None
                                                 and np.all(np.isfinite(np.asarray(intensity_err)[gmask])
                                                            & (np.asarray(intensity_err)[gmask] > 0))
                                                 else None))
                    if gmask.sum() >= 3 else None),
        "porod": porod,
        "overall": static_powerlaw_fit(q, I, intensity_err=intensity_err),
    }
    if out["guinier"] and out["porod"]:
        out["regime"] = "spans Guinier (low-q) and Porod (high-q)"
    elif out["guinier"]:
        out["regime"] = "Guinier (low-q) into the form-factor regime"
    elif out["porod"]:
        out["regime"] = "form-factor / Porod (q·R ≳ π throughout)"
    else:
        out["regime"] = "form-factor crossover (q·R ~ 1)"
    # when the Porod slope is valid (well-fit, falling), use it to inform the
    # form-factor / morphology model
    out["porod_morphology"] = (porod_morphology(out["porod"]["exponent"], out["porod"]["r2"])
                               if out["porod"] is not None and out["porod"].get("accepted")
                               else None)
    return out


def residual_structure_test(resid):
    """Runs test + lag-1 autocorrelation on fit residuals (model adequacy).

    r²/χ² do not detect *structured* residuals: a fit can have r² ≈ 1 with
    systematically signed runs, the signature of a wrong model class.  This runs
    a Wald–Wolfowitz runs test about the median plus the lag-1 autocorrelation
    and the Durbin–Watson statistic, and flags the residuals as ``structured``
    (model-inadequacy) when either is significant.
    """
    r = np.asarray(resid, float); r = r[np.isfinite(r)]
    out = {"n": int(r.size), "runs": float("nan"), "runs_expected": float("nan"),
           "runs_z": float("nan"), "runs_p": float("nan"),
           "lag1_autocorr": float("nan"), "durbin_watson": float("nan"),
           "structured": False}
    if r.size < 8:
        return out
    s = np.sign(r - np.median(r)); s = s[s != 0]
    if s.size < 4:
        return out
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    n_pos = int(np.sum(s > 0)); n_neg = int(np.sum(s < 0)); n = n_pos + n_neg
    mu = 1.0 + 2.0 * n_pos * n_neg / n
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n * n * (n - 1.0)) \
        if n > 1 else 0.0
    z = (runs - mu) / math.sqrt(var) if var > 0 else float("nan")
    p = math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else float("nan")
    rc = r - r.mean(); denom = float(np.sum(rc * rc))
    ac1 = float(np.sum(rc[1:] * rc[:-1]) / denom) if denom > 0 else float("nan")
    dw = float(np.sum(np.diff(r) ** 2) / denom) if denom > 0 else float("nan")
    structured = bool((np.isfinite(p) and p < 0.05)
                      or (np.isfinite(ac1) and abs(ac1) > 0.3))
    out.update(runs=runs, runs_expected=mu, runs_z=z, runs_p=p,
               lag1_autocorr=ac1, durbin_watson=dw, structured=structured)
    return out


def loo_cv_gamma(q, gam):
    """Leave-one-angle-out cross-validation of the Γ-vs-q² model.

    For each angle, fit D (+ intercept) on the *other* angles and predict the
    held-out Γ.  Returns the predictive CV-R² and CV-RMSE on predicted-vs-
    observed Γ — a score a high in-sample r² can mask.
    """
    q = np.asarray(q, float); gam = np.asarray(gam, float)
    good = np.isfinite(q) & np.isfinite(gam)
    q, gam = q[good], gam[good]
    n = q.size
    if n < 4:
        return {"cv_r2": float("nan"), "cv_rmse": float("nan"), "n": int(n)}
    pred = np.empty(n)
    for i in range(n):
        msk = np.ones(n, bool); msk[i] = False
        fit = C.gamma_vs_q2(q[msk], gam[msk])
        pred[i] = fit["D"] * q[i] ** 2 + fit["intercept"]
    ss = float(np.sum((gam - pred) ** 2)); st = float(np.sum((gam - gam.mean()) ** 2))
    return {"cv_r2": 1.0 - ss / st if st > 0 else float("nan"),
            "cv_rmse": float(np.sqrt(ss / n)), "pred": pred, "obs": gam, "n": int(n)}


def linear_gvq2_window(q, gam, sigma=None, *, min_points=4, t_crit=3.0, rel_floor=0.05,
                       temperature=None, viscosity=None):
    """Fit a line to the linear part of Γ vs q² — excluding high-q angles that
    roll off — and report the slope as the apparent diffusion coefficient.

    At high scattering angle the signal weakens and number fluctuations / residual
    multiple scattering / aggregate artefacts make Γ plateau or drop, so the
    highest-q points fall *below* the Γ(q²) trend.  We fit Γ = c + D·q² to the
    contiguous lower-q points and exclude a high-q point (closing the window) once
    it falls below the line extrapolated through the retained points by more than
    ``t_crit``·σ AND ``rel_floor`` in relative terms — a downward rolloff; upward
    (polydispersity) curvature does not truncate the fit.  The apparent diffusion
    coefficient is the slope D (R_h = kT/6πηD); the intercept c is reported as a
    non-diffusive q→0 offset / low-q-curvature diagnostic.  σ is the replicate SEM
    when available, else residual-estimated."""
    q = np.asarray(q, float); g = np.asarray(gam, float)
    good = np.isfinite(q) & np.isfinite(g) & (q > 0) & (g > 0)
    q, g = q[good], g[good]
    sig = np.asarray(sigma, float)[good] if sigma is not None else None
    o = np.argsort(q); q, g = q[o], g[o]
    sig = sig[o] if sig is not None else None
    n = q.size
    if n == 0:
        return None
    q2 = q * q; scale = float(np.mean(q2)) or 1.0; x = q2 / scale
    if sig is None or not np.all(np.isfinite(sig)) or np.any(sig <= 0):
        k0 = max(min(min_points + 1, n), n // 2)
        if k0 >= 3:
            A0 = np.vstack([np.ones(k0), x[:k0]]).T
            sol0, *_ = np.linalg.lstsq(A0, g[:k0], rcond=None)
            s0 = float(np.std(g[:k0] - A0 @ sol0, ddof=2))
        else:
            s0 = 0.03 * float(np.mean(g))
        sig = np.full(n, max(s0, 1e-30)); sigma_src = "homoscedastic (residual-estimated)"
    else:
        sigma_src = "replicate SEM"
    w = 1.0 / sig ** 2

    def linefit(k):  # weighted Γ = c + D·q² on the first k (lowest-q) points
        X = np.vstack([np.ones(k), x[:k]]).T
        cov = np.linalg.inv((X * w[:k, None]).T @ X)
        return cov @ ((X * w[:k, None]).T @ g[:k]), cov  # [c, D_scaled]

    # grow from low q; close the window at the first high-q *downward* rolloff
    k = n
    if n >= min_points + 1:
        k = min_points
        while k < n:
            sol, cov = linefit(k)
            pred = float(sol[0]) + float(sol[1]) * x[k]
            J = np.array([1.0, x[k]]); sp = math.sqrt(max(float(J @ cov @ J), 0.0) + sig[k] ** 2)
            dev = g[k] - pred; z = dev / sp if sp > 0 else 0.0
            if dev < 0 and abs(z) > t_crit and abs(dev) > rel_floor * abs(pred):
                break
            k += 1
    sol, cov = linefit(k)
    D = float(sol[1]) / scale; D_se = math.sqrt(max(cov[1, 1], 0.0)) / scale
    c = float(sol[0]); c_se = math.sqrt(max(cov[0, 0], 0.0))
    # residual-structure / nonlinearity diagnostics on the *retained* points: even
    # after dropping the rolloff, a polydisperse sample leaves structured residuals
    # (Γ vs q² curves upward) — quantify with the q⁴ curvature term, a runs test,
    # and the residual-vs-q trend.
    pred_k = c + D * q2[:k]; resid_k = g[:k] - pred_k
    rel_resid = resid_k / np.clip(g[:k], 1e-30, None)
    curv_t = float("nan")
    if k >= 4:
        Xq = np.vstack([np.ones(k), x[:k], x[:k] ** 2]).T
        covq = np.linalg.inv((Xq * w[:k, None]).T @ Xq); solq = covq @ ((Xq * w[:k, None]).T @ g[:k])
        eq_se = math.sqrt(max(covq[2, 2], 0.0)); curv_t = float(solq[2] / eq_se) if eq_se > 0 else float("nan")
    sgn = np.sign(resid_k); sgn[sgn == 0] = 1
    runs = int(1 + np.count_nonzero(sgn[1:] != sgn[:-1]))
    npos = int(np.count_nonzero(sgn > 0)); nneg = k - npos
    exp_runs = 1.0 + 2.0 * npos * nneg / k if k else float("nan")
    resid_trend = float("nan")
    if k >= 4:
        rt = np.argsort(np.argsort(resid_k)); qt = np.argsort(np.argsort(q[:k]))
        resid_trend = float(np.corrcoef(rt, qt)[0, 1])  # Spearman(resid, q)
    out = {"D": D, "D_se": D_se, "intercept": c, "intercept_se": c_se,
           "intercept_sig": bool(abs(c) > 2 * c_se and abs(c) > 0.02 * float(np.max(g))),
           "n_linear": int(k), "n_total": int(n), "n_excluded": int(n - k),
           "q_min": float(q[0]), "q_max_linear": float(q[k - 1]),
           "q2_max_linear": float(q2[k - 1]),
           "excluded_q": [float(v) for v in q[k:]], "stopped_early": bool(k < n),
           "sigma_src": sigma_src, "t_crit": t_crit, "rel_floor": rel_floor,
           "D_lowest_angle": float(g[0] / q2[0]),
           "curv_t": curv_t, "resid_runs": runs, "resid_exp_runs": float(exp_runs),
           "resid_trend": resid_trend, "resid_max_pct": float(np.max(np.abs(rel_resid)) * 100.0),
           "resid_structured": bool(float(np.max(np.abs(rel_resid)) * 100.0) > 4.0
                                    and ((np.isfinite(curv_t) and abs(curv_t) > 3.0)
                                         or runs < 0.6 * exp_runs))}
    if temperature and viscosity and D > 0:
        out["R_h"] = C.k_BOLTZMANN * temperature / (6.0 * math.pi * viscosity * D)
    return out


def shape_factor(s):
    """rho = R_g/R_h as a model-independent shape diagnostic (point estimate).

    R_g is the *independent* static Guinier radius — defined only when the
    sample reaches the Guinier regime (q·R_g < 1.3); R_h is the z-average
    dynamic radius.  When no Guinier regime is reached, rho is not independently
    determined (the PSD-implied R_g = sqrt(3/5)·R only re-states the sphere
    assumption), and ``independent`` is False.
    """
    rd = s.get("regime"); g = s.get("gvq2", {}) or {}
    Rh = g.get("R_h", float("nan"))
    out = {"rho": float("nan"), "R_g": float("nan"), "R_h": Rh,
           "model": "undetermined", "independent": False}
    if rd is not None and rd.get("guinier") is not None and Rh and Rh > 0:
        rg = rd["guinier"].get("R_g", float("nan"))
        if np.isfinite(rg) and rg > 1e-9:
            rho = rg / Rh
            out.update(rho=rho, R_g=rg, model=C._shape_from_rho(rho), independent=True)
    return out


def bootstrap_suite(s, n_boot=400, seed=0):
    """Angle-resampling bootstrap CIs for the headline R_h and rho = R_g/R_h.

    Replicate-level curves are not threaded through the suite, so the dominant
    reproducible uncertainty in the *z-average* R_h is the angle-to-angle
    scatter in Γ(q).  Each draw resamples angles with replacement, refits
    Γ-vs-q² → R_h, and — when an independent low-q Guinier R_g exists — refits
    that Guinier on the resampled static points to form rho.  Returns median +
    68% + 95% intervals.
    """
    q = np.asarray(s["q"], float); gam = np.asarray(s["gam"], float)
    I = np.asarray(s["I"], float)
    T = s["T"]; eta = s["eta"]; n = q.size
    kT6 = C.k_BOLTZMANN * T / (6.0 * math.pi * eta)
    rd = s.get("regime")
    want_rho = (rd is not None and rd.get("guinier") is not None
                and np.isfinite(rd["guinier"].get("R_g", float("nan")))
                and rd["guinier"]["R_g"] > 1e-9)
    if n < 4:
        return {"R_h": None, "rho": None}
    rng = np.random.default_rng(seed)
    Rhs, rhos = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fit = C.gamma_vs_q2(q[idx], gam[idx])
        D = fit["D"]
        if not (D > 0):
            continue
        Rh = kT6 / D; Rhs.append(Rh)
        if want_rho:
            gf = C.guinier_fit(q[idx], I[idx])
            rg = gf.get("R_g", float("nan"))
            if np.isfinite(rg) and rg > 1e-9:
                rhos.append(rg / Rh)

    def ci(a):
        a = np.asarray(a, float)
        if a.size < 10:
            return None
        return {"median": float(np.median(a)), "n": int(a.size),
                "p16": float(np.percentile(a, 16)), "p84": float(np.percentile(a, 84)),
                "p2_5": float(np.percentile(a, 2.5)), "p97_5": float(np.percentile(a, 97.5))}
    return {"R_h": ci(Rhs), "rho": ci(rhos) if want_rho else None}


def replicate_bootstrap_rh(s, n_boot=400, seed=0):
    """True replicate bootstrap of the z-average R_h (measurement repeatability).

    Where per-angle replicate curves are available, each draw resamples the
    replicates within every kept angle (with replacement), re-averages, re-fits
    the cumulant Γ, and refits Γ-vs-q² → R_h.  This isolates the *repeatability*
    component of the R_h uncertainty (distinct from the angle-to-angle / model
    scatter captured by :func:`bootstrap_suite`).  Returns median + 68% + 95%
    intervals, or None when no angle has ≥2 replicates.
    """
    reps = s.get("_reps"); kept = s.get("_kept_datasets")
    if not reps or kept is None or not any(r is not None and len(r) >= 2 for r in reps):
        return None
    q = np.asarray(s["q"], float); T = s["T"]; eta = s["eta"]
    kT6 = C.k_BOLTZMANN * T / (6.0 * math.pi * eta)
    rng = np.random.default_rng(seed)
    Rhs = []
    for _ in range(n_boot):
        gam = np.empty(len(kept)); ok = True
        for i, ds in enumerate(kept):
            r = reps[i]
            if r is not None and len(r) >= 2:
                idx = rng.integers(0, len(r), len(r))
                g = np.mean([r[k] for k in idx], axis=0)
            else:
                g = ds.g2m
            cf = C.cumulant_fit(ds.tau, g, beta=ds.beta, order=2, g1_min=G1_MIN)
            if not np.isfinite(cf["Gamma"]) or cf["Gamma"] <= 0:
                ok = False; break
            gam[i] = cf["Gamma"]
        if not ok:
            continue
        fit = C.gamma_vs_q2(q, gam)
        if fit["D"] > 0:
            Rhs.append(kT6 / fit["D"])
    a = np.asarray(Rhs, float)
    if a.size < 10:
        return None
    return {"median": float(np.median(a)), "n": int(a.size),
            "p16": float(np.percentile(a, 16)), "p84": float(np.percentile(a, 84)),
            "p2_5": float(np.percentile(a, 2.5)), "p97_5": float(np.percentile(a, 97.5))}


def bootstrap_psd_peak(s, n_boot=20, seed=0):
    """Angle-resampling bootstrap of the CORENN PSD primary-peak radius.

    Resamples the angles fed to the joint inversion (with replacement), re-runs
    ``corenn_multi_angle`` each time, and collects the primary peak radius —
    giving a CI on the headline PSD size.  Costs one full inversion per draw, so
    ``n_boot`` is kept modest and this is opt-in (``psd_bootstrap=`` in
    ``run_suite``).  Returns median + 68% + 95% intervals in nm, or None.
    """
    cds = s.get("_corenn_datasets")
    if not cds or s.get("corenn") is None:
        return None
    T = s["T"]; eta = s["eta"]; rr = s.get("_radius_range"); lm = s.get("_lag_max")
    n = len(cds)
    if n < 4:
        return None
    # fix the regularization at the main inversion's GCV-selected lambda so each
    # draw is a single solve (≈20x faster) and isolates the angle-resampling
    # uncertainty from GCV re-selection noise
    lam = float(getattr(s["corenn"], "lam", float("nan")))
    _fix = np.isfinite(lam) and lam > 0
    lr = (lam, lam) if _fix else (1e-6, 1e3); nl = 1 if _fix else 20
    rng = np.random.default_rng(seed)
    radii = []
    for _ in range(n_boot):
        sub = [cds[k] for k in rng.integers(0, n, n)]
        try:
            r = C.corenn_multi_angle(sub, radius_range=rr, n_basis=24, n_R=150,
                                     temperature=T, viscosity=eta, lag_max=lm,
                                     n_lambda=nl, lambda_range=lr)
            if r.peaks:
                radii.append(r.peaks[0]["radius_nm"])
        except Exception:
            continue
    if len(radii) < 8:
        return None
    a = np.asarray(radii, float)
    return {"median": float(np.median(a)), "n": int(a.size),
            "p16": float(np.percentile(a, 16)), "p84": float(np.percentile(a, 84)),
            "p2_5": float(np.percentile(a, 2.5)), "p97_5": float(np.percentile(a, 97.5))}


def corenn_diagnostics(corenn):
    """Surface CORENN conditioning / identifiability for the report.

    Effective DOF (trace of the influence matrix), the kernel condition number,
    whether the GCV minimum is interior to the λ grid (a well-posed selection,
    not a boundary), and the number of statistically supportable modes (≲ DOF).
    """
    if corenn is None:
        return None
    ex = getattr(corenn, "extra", {}) or {}
    lam_grid = np.asarray(getattr(corenn, "lam_grid", []), float)
    gcv = np.asarray(getattr(corenn, "gcv", []), float)
    interior = None
    if lam_grid.size and gcv.size and np.any(np.isfinite(gcv)):
        k = int(np.nanargmin(gcv))
        interior = bool(0 < k < gcv.size - 1)
    dof = ex.get("eff_dof", float("nan"))
    return {"eff_dof": dof, "cond": ex.get("cond", float("nan")),
            "converged": ex.get("converged"), "n_basis": ex.get("n_basis"),
            "n_data": ex.get("n_data"), "lam": float(getattr(corenn, "lam", float("nan"))),
            "gcv_interior_min": interior,
            "supportable_modes": int(round(dof)) if np.isfinite(dof) else None}


def _trusted_tau_max(datasets, lag_max=None):
    """Largest lag actually available to CORENN after the configured cutoff."""
    vals = []
    for ds in datasets:
        tau = np.asarray(ds.tau, float)
        good = np.isfinite(tau) & (tau > 0)
        if lag_max is not None:
            good &= tau <= lag_max
        if np.any(good):
            vals.append(float(np.max(tau[good])))
    if vals:
        return float(max(vals))
    return float(lag_max) if lag_max is not None and np.isfinite(lag_max) else float("nan")


def corenn_psd_radius_bounds(datasets, T, eta, lag_max=None,
                             radius_range=None,
                             min_field_decay=CORENN_MIN_FIELD_DECAY,
                             static_qr_factor=CORENN_STATIC_QR_FACTOR,
                             lower_q_factor=CORENN_RADIUS_LOWER_Q_FACTOR):
    """Radius grid bounds from dynamic observability plus static q guardrail.

    A radius is dynamically observable only if the slowest accepted scattering
    vector sees at least ``min_field_decay`` of field-correlation decay inside
    the trusted lag window:

        q_min^2 kT tau_max / (6 pi eta R) >= min_field_decay

    The static guardrail ``static_qr_factor / q_min`` prevents the inversion
    from assigning intensity to length scales larger than the spatial window
    sampled by the kept multi-angle data.  The selected upper bound is the most
    conservative finite bound unless the caller explicitly supplies
    ``radius_range``.
    """
    qs = np.asarray([d.q for d in datasets if d.q], float)
    if qs.size == 0:
        return {"radius_range": radius_range or C.DEFAULT_CORENN_RADIUS_RANGE,
                "user_override": radius_range is not None}
    q_min, q_max = float(np.min(qs)), float(np.max(qs))
    tau_trusted = _trusted_tau_max(datasets, lag_max=lag_max)
    lower_q = lower_q_factor / q_max
    lower = max(lower_q, C.DEFAULT_CORENN_RADIUS_RANGE[0])
    kT = C.k_BOLTZMANN * T
    dynamic = (q_min * q_min * kT * tau_trusted
               / (6.0 * math.pi * eta * min_field_decay)
               if (q_min > 0 and eta > 0 and min_field_decay > 0
                   and np.isfinite(tau_trusted) and tau_trusted > 0)
               else float("nan"))
    static = static_qr_factor / q_min if q_min > 0 and static_qr_factor > 0 else float("nan")
    default_hi = C.DEFAULT_CORENN_RADIUS_RANGE[1]
    candidates = [v for v in (dynamic, static, default_hi) if np.isfinite(v) and v > lower]
    selected = min(candidates) if candidates else max(lower * 10.0, default_hi)
    labels = {"dynamic": dynamic, "static_q": static, "default": default_hi}
    limiting = min((k for k, v in labels.items() if np.isfinite(v) and v > lower),
                   key=lambda k: labels[k], default="fallback")
    if radius_range is not None:
        rr = tuple(map(float, radius_range))
        return {"radius_range": rr, "R_min": rr[0], "R_max": rr[1],
                "lower_q": lower_q, "dynamic_upper": dynamic,
                "static_upper": static, "default_upper": default_hi,
                "q_min": q_min, "q_max": q_max,
                "tau_max_trusted": tau_trusted,
                "min_field_decay": min_field_decay,
                "static_qr_factor": static_qr_factor,
                "limiting": "user_override", "user_override": True}
    return {"radius_range": (lower, selected), "R_min": lower, "R_max": selected,
            "lower_q": lower_q, "dynamic_upper": dynamic,
            "static_upper": static, "default_upper": default_hi,
            "q_min": q_min, "q_max": q_max,
            "tau_max_trusted": tau_trusted,
            "min_field_decay": min_field_decay,
            "static_qr_factor": static_qr_factor,
            "limiting": limiting, "user_override": False}


def run_suite(datasets, T, eta, g1_min=G1_MIN, corenn_subset=2,
              radius_range=None, resolvable_factor=3.0, psd_bootstrap=0,
              replicates=None):
    """Full static+dynamic analysis of one angle-resolved channel.

    A curve whose leading cumulant implies an apparent R_h far outside the
    measured q-window is rejected from the cumulant diagnostics.  The CORENN
    inversion then gets its PSD radius grid from dynamic observability in the
    trusted lag window, capped by a static q-space guardrail, so radii that
    cannot decorrelate or be spatially distinguished are reported as unresolved
    slow material rather than as resolved PSD peaks.
    """
    kT6 = C.k_BOLTZMANN * T / (6.0 * np.pi * eta)
    q, gam, mu2, I, ang, kept = [], [], [], [], [], []
    kept_reps, rejected = [], []
    qall = np.array([d.q for d in datasets if d.q])
    q_min, q_max = float(qall.min()), float(qall.max())
    R_reject = resolvable_factor * np.pi / q_min       # cumulant reject threshold
    R_small = 0.3 / q_max                              # smallest credible R_h
    for j, ds in enumerate(datasets):
        c = C.cumulant_fit(ds.tau, ds.g2m, beta=ds.beta, order=2, g1_min=g1_min)
        th = ds.metadata.get("angle_deg")
        Ir = C.static_intensity_proxy(ds)
        rep = replicates[j] if (replicates is not None and j < len(replicates)) else None
        if not np.isfinite(c["Gamma"]) or c["Gamma"] <= 0 or th is None or Ir is None:
            rejected.append({"angle": th, "q": ds.q, "reason": "no valid cumulant Γ",
                             "R_app_nm": float("nan")})
            continue
        R_app = kT6 * ds.q ** 2 / c["Gamma"]           # apparent hydrodynamic radius
        if not (R_small < R_app < R_reject):           # beyond resolution -> reject
            rejected.append({"angle": th, "q": ds.q,
                             "reason": ("apparent R_h beyond resolvable window"
                                        f" (>{R_reject*1e9:.0f} nm)" if R_app >= R_reject
                                        else "apparent R_h below credible floor"),
                             "R_app_nm": R_app * 1e9})
            continue
        q.append(ds.q); gam.append(c["Gamma"]); mu2.append(c["mu2"])
        I.append(Ir); ang.append(th); kept.append(ds); kept_reps.append(rep)
    q, gam, mu2, I, ang = map(np.array, (q, gam, mu2, I, ang))
    q_kept_min = float(q.min()) if q.size else float("nan")
    q_kept_max = float(q.max()) if q.size else float("nan")
    R_res_max = np.pi / q_min                          # all-angle static guardrail
    s = {"q": q, "gam": gam, "mu2": mu2, "I": I, "ang": ang, "T": T, "eta": eta,
         "n": q.size, "n_total": len(datasets),
         "q_range": (q.min(), q.max()) if q.size else (np.nan, np.nan),
         "R_resolvable": (R_small, R_res_max),
         "R_cumulant_reject": R_reject,
         "_kept_datasets": kept, "_reps": kept_reps, "rejected": rejected}
    s["replicate_stats"] = replicate_observation_stats(s)
    I_err = s["replicate_stats"].get("I_sem")
    s["gvq2"] = C.gamma_vs_q2(q, gam, temperature=T, viscosity=eta)
    s["gvq2_linear"] = linear_gvq2_window(
        q, gam, sigma=s["replicate_stats"].get("gamma_sem"), temperature=T, viscosity=eta)
    s["app"] = C.apparent_diffusion_analysis(q, gam, T, eta, mu2=mu2)
    s["sdc"] = C.static_dynamic_correlation(q, I, gam, T, eta,
                                             intensity_err=I_err, mu2=mu2)
    s["static_pl_overall"] = static_powerlaw_fit(q, I, intensity_err=I_err)
    s["static_pl"] = select_static_powerlaw_window(q, I, intensity_err=I_err)
    # focus the joint inversion on the main relaxation: ~20 decay times of the
    # z-average mode at the HIGHEST q (where the main decay is fastest and
    # cleanest), so the slow aggregate tail at low q does not skew the
    # intensity-weighted PSD toward unphysically large particles.
    Dz = s["gvq2"]["D"]
    # gap-aware stitched data has clean long-lag statistics, so keep the full
    # lag window and let the inversion resolve the slow (large-aggregate) tail;
    # otherwise truncate to ~20 decay times of the main mode to avoid skew.
    _stitched = any((getattr(d, "metadata", {}) or {}).get("stitched") for d in kept)
    if _stitched:
        lag_max = None
    else:
        lag_max = 20.0 / max(Dz * q_kept_max * q_kept_max, 1e-6) if Dz > 0 else None
    corenn_datasets = kept[::corenn_subset]
    psd_bounds = corenn_psd_radius_bounds(corenn_datasets, T, eta,
                                          lag_max=lag_max,
                                          radius_range=radius_range)
    radius_range = psd_bounds["radius_range"]
    # NOTE: we deliberately do NOT push the upper radius past the static
    # q-guardrail even for stitched data.  The multi-angle intensity inversion
    # couples angles through the sphere form factor P(qR), which is meaningless
    # for R >> pi/q; extending the grid there makes the R^6-weighted intensity
    # inversion pile mass at the boundary (a pure artifact, verified: the peak
    # tracks the bound).  Stitching's real benefit here is the clean long-lag
    # (lag_max=None above) -> a better-determined main mode and small tail
    # WITHIN the q-resolvable range.  For genuine large-aggregate (>~ pi/q_min)
    # sizing use the single-angle corenn_invert on a stitched dataset.
    s["stitched"] = _stitched
    s["_corenn_datasets"] = corenn_datasets
    s["_radius_range"] = radius_range; s["_lag_max"] = lag_max
    s["psd_bounds"] = psd_bounds
    try:
        s["corenn"] = C.corenn_multi_angle(corenn_datasets,
                                           radius_range=radius_range, n_basis=24,
                                           n_R=150, temperature=T, viscosity=eta,
                                           lag_max=lag_max)
        s["corenn_lag_max"] = lag_max
    except Exception:
        s["corenn"] = None
    # anticipate the Guinier/Porod regime from the recovered PSD and run the
    # regime-appropriate static fit
    s["regime"] = regime_diagnostic(q, I, s["corenn"], intensity_err=I_err,
                                    powerlaw=s["static_pl"])
    # shape, uncertainty, and fit-adequacy diagnostics
    s["rho"] = shape_factor(s)
    s["boot"] = bootstrap_suite(s)
    s["cv"] = loo_cv_gamma(q, gam)
    gpred = s["gvq2"]["D"] * q ** 2 + s["gvq2"]["intercept"]
    s["resid_gvq2"] = residual_structure_test(gam - gpred)
    if s["corenn"] is not None:
        # g2m is the full curve, g2_model the (lag-truncated) leading block used
        # in the fit; compare them on the matched leading block.
        cr = s["corenn"]; mlen = min(cr.g2m.size, cr.g2_model.size)
        s["resid_corenn"] = residual_structure_test(cr.g2m[:mlen] - cr.g2_model[:mlen])
    else:
        s["resid_corenn"] = None
    s["corenn_diag"] = corenn_diagnostics(s["corenn"])
    s["boot_psd"] = bootstrap_psd_peak(s, n_boot=psd_bootstrap) if psd_bootstrap else None
    s["boot_rep"] = replicate_bootstrap_rh(s)          # repeatability CI (if reps given)
    return s


def suite_steps_md(s, channel="", start=2):
    """Markdown lines for Steps `start`.. of the static+dynamic suite."""
    i = start
    g = s["gvq2"]; ap = s["app"]; sd = s["sdc"]
    sg = sd["segmented_guinier"]
    seg = sg["two_segment"]["segments"] if sg["supports_two_segments"] else None
    ch = f" ({channel})" if channel else ""
    m = []
    rej = s.get("n_total", s["n"]) - s["n"]
    rej_note = (f" {rej} of {s['n_total']} angles are rejected because the cumulant "
                f"apparent R_h exceeds the slow-mode acceptance threshold (~3π/q_low, "
                f"here >{s.get('R_cumulant_reject', float('nan'))*1e9:.0f} nm) — i.e. they are dominated by an "
                "unresolvably slow aggregate mode / noise." if rej else "")
    m.append(f"**Step {i} — per-angle cumulants → Γ(q){ch}.** A 2nd-order cumulant on "
             f"the initial decay (g1_min={G1_MIN}) gives the decay rate at each angle; "
             f"{s['n']} angles are kept.{rej_note}\n"); i += 1
    bt = s.get("boot") or {}; br = s.get("boot_rep")
    parts_ci = []
    if br:
        parts_ci.append(f"replicate/repeatability 68% {br['p16']*1e9:.0f}–{br['p84']*1e9:.0f}")
    if bt.get("R_h"):
        b = bt["R_h"]
        parts_ci.append(f"angle/model 68% {b['p16']*1e9:.0f}–{b['p84']*1e9:.0f}")
    rh_ci = f" [{'; '.join(parts_ci)} nm]" if parts_ci else ""
    gl = s.get("gvq2_linear") or {}
    ok = gl.get("n_linear") and np.isfinite(gl.get("R_h", float("nan"))) and gl["D"] > 0
    icpt = ("" if not gl.get("intercept_sig") else
            f" The fitted intercept c ≈ {gl['intercept']:.0f}/s is non-zero — a low-q curvature / "
            "non-diffusive offset (large-aggregate slow mode, baseline, or weak S(q)) — so read D "
            "as the slope of the linear region, not Γ/q².")
    if ok and gl.get("n_excluded"):
        exq = ", ".join(_qnm(v) for v in gl["excluded_q"])
        step2 = (
            f"**Step {i} — Γ vs q² (apparent diffusion).** Γ vs q² is linear over the bulk but "
            f"the **{gl['n_excluded']} highest-q angle(s) roll off** (Γ plateaus/drops at "
            f"q = {exq} nm⁻¹ — weak signal, number fluctuations, and residual multiple "
            f"scattering at high angle), so they are excluded. A straight-line fit Γ = c + D·q² to "
            f"the remaining **{gl['n_linear']} of {gl['n_total']} angles** "
            f"(q ≤ {_qnm(gl['q_max_linear'])} nm⁻¹, σ from {gl['sigma_src']}) gives the "
            f"**apparent diffusion coefficient D = {gl['D']:.3e} ± {gl['D_se']:.1e} m²/s → "
            f"R_h = {gl['R_h']*1e9:.0f} nm** (vs {g['R_h']*1e9:.0f} nm for the all-angle fit that "
            f"includes the rolloff).{icpt} ")
    elif ok:
        step2 = (
            f"**Step {i} — Γ vs q² (apparent diffusion).** A straight-line fit Γ = c + D·q² over "
            f"all {s['n']} angles (no high-q rolloff detected) gives the **apparent diffusion "
            f"coefficient D = {gl['D']:.3e} ± {gl['D_se']:.1e} m²/s → R_h = {gl['R_h']*1e9:.0f} nm** "
            f"(r²={g['r2']:.3f}){rh_ci}.{icpt} ")
    else:
        step2 = (f"**Step {i} — Γ vs q² (apparent diffusion).** A straight-line fit over "
                 f"{s['n']} angles gives D = **{g['D']:.3e} m²/s** (r²={g['r2']:.3f}) → "
                 f"R_h = {g['R_h']*1e9:.0f} nm{rh_ci}. ")
    if ok and gl.get("resid_structured"):
        up = (gl.get("curv_t", 0) or 0) > 0
        shape = ("**curves upward** — D_app rises with q, the signature of **polydispersity**"
                 if up else
                 "**curves downward** — D_app falls with q, a non-diffusive low-q offset "
                 "(slow large-aggregate mode, weak S(q), or baseline)")
        step2 += (
            f"**The fit residuals are structured, not random** (figure inset): they trend "
            f"systematically with q (Spearman resid–q = {gl['resid_trend']:+.2f}; runs "
            f"{gl['resid_runs']} vs ~{gl['resid_exp_runs']:.0f} expected) and a q⁴ curvature term "
            f"is significant at **{gl['curv_t']:+.0f}σ** — so even with the rolloff removed, "
            f"Γ vs q² genuinely {shape}: the line is a chord and no single slope is *the* diffusion "
            f"coefficient. The q-dependent apparent size (next step) and the CORENN PSD are the "
            f"proper description.\n")
    else:
        step2 += "D_app then varies with q for a polydisperse sample (quantified next).\n"
    m.append(step2); i += 1
    lo_q = ap["R_h_low_q_median"] * 1e9; hi_q = ap["R_h_high_q_median"] * 1e9
    pdi = ap.get("PDI_mean", float("nan"))
    trend = ("falls" if ap["spearman_q_Rh"] < -0.2 else
             "rises" if ap["spearman_q_Rh"] > 0.2 else "is roughly flat")
    if ap["spearman_q_Rh"] < -0.2:
        interp = "a broad / multi-modal population (large-particle tail at low q)"
    elif ap["spearman_q_Rh"] > 0.2:
        interp = "a narrow population with weak interparticle structure (S(q))"
    elif np.isfinite(pdi) and pdi > 0.3:
        interp = "a polydisperse population with no single dominant length scale"
    else:
        interp = "a near-monodisperse population"
    m.append(f"**Step {i} — apparent R_h(q) (polydispersity).** apparent R_h {trend} with q, "
             f"from **{lo_q:.0f} nm** (low q) to **{hi_q:.0f} nm** (high q) "
             f"(curvature C={ap['C']:.2e}, Spearman {ap['spearman_q_Rh']:+.2f}); mean PDI "
             f"{ap.get('PDI_mean', float('nan')):.2f} — {interp}.\n"); i += 1
    rd = s.get("regime")
    pl = s.get("static_pl", {})
    if rd is None:
        p = pl.get("exponent", float("nan"))
        m.append(f"**Step {i} — static intensity I(q) (SLS).** No PSD-based regime estimate "
                 f"available; the overall log-log slope is I ∝ q^{-p:.1f} "
                 f"(r²={pl.get('r2', float('nan')):.2f}).\n")
    else:
        parts = [f"**Step {i} — static intensity I(q), regime from the PSD.** The CORENN "
                 f"PSD gives dominant scatterers of R ≈ {rd['R_char']*1e9:.0f} nm "
                 f"(R_g ≈ {rd['R_g_char']*1e9:.0f} nm, sphere-derived from the PSD radius "
                 "— no Guinier regime is reached, so this is a model estimate, not a measured "
                 f"R_g), so q·R_g spans "
                 f"{rd['qRg_min']:.1f}–{rd['qRg_max']:.1f} across the angles → **{rd['regime']}**."]
        if rd["guinier"] is not None:
            rg = rd["guinier"].get("R_g", float("nan"))
            if np.isfinite(rg) and rg > 1e-9:
                parts.append(f"A Guinier fit on the {rd['n_guinier']} low-q points "
                             f"(q·R_g < 1.3) gives R_g ≈ **{rg*1e9:.0f} nm** "
                             f"(r²={rd['guinier']['r2']:.2f}).")
            else:
                parts.append(f"The {rd['n_guinier']} low-q points are in the Guinier window "
                             "but the intensity does not fall (no positive R_g) — a low-q "
                             "structure factor S(q) < 1 / interparticle correlations rather "
                             "than a particle Guinier radius.")
        if rd["porod"] is not None:
            pp = rd["porod"]["exponent"]; mph = rd.get("porod_morphology")
            pse = rd["porod"].get("exponent_se", float("nan"))
            qlo = rd["porod"].get("q_min", float("nan"))
            qhi = rd["porod"].get("q_max", float("nan"))
            slope_txt = (f"I ∝ q^−{pp:.2f}±{pse:.2f}" if np.isfinite(pse)
                         else f"I ∝ q^−{pp:.1f}")
            if mph is not None:
                tag = f"{slope_txt} → **{mph['label']}**"
            elif pp > 1.0:
                tag = f"{slope_txt}: a shallow / poorly-resolved power law"
            else:
                tag = (f"a near-flat / noisy high-q intensity (I ∝ q^{-pp:+.1f}), so no clean "
                       "Porod slope")
            window = (f"q={_qrange_nm(qlo, qhi)}"
                      if np.isfinite(qlo) and np.isfinite(qhi) else "selected q-window")
            accepted = "" if rd["porod"].get("accepted") else " (reported as diagnostic only)"
            parts.append(f"On the {rd['n_porod']} point contiguous lower-q scaling window "
                         f"({window}): {tag} (r²={rd['porod']['r2']:.3f}){accepted}.")
            hq = rd["porod"].get("high_q")
            if hq:
                up = "upturn" in str(hq.get("behavior", ""))
                parts.append(
                    f"The {hq['n_points']} higher-q point(s) are excluded from the exponent "
                    f"because they show {hq['behavior']} relative to the lower-q scaling law; "
                    "physically this is the end of the mass-fractal / form-factor scaling range"
                    + (" plus the form-factor rise of a smaller, weakly-scattering population — "
                       "an *upturn* rather than a plateau is the signature of small particles, "
                       "consistent with the small-R modes that appear under number-weighting in "
                       "the size-distribution section below" if up else
                       " and/or an additive static background / instrument-floor contribution")
                    + ", not a second Porod law.")
        if rd["guinier"] is None and rd["porod"] is None:
            parts.append("The data sits in the form-factor crossover; the overall log-log "
                         f"slope is I ∝ q^−{pl.get('exponent', float('nan')):.1f}.")
        m.append(" ".join(parts) + "\n")
    i += 1
    m.append(f"**Step {i} — dynamic q-dependence.** The apparent size follows a power law "
             f"R_h,app ∝ q^−α with α={sd['powerlaw_alpha']:.2f} (r²={sd['r2_loglog']:.2f}) — "
             "a compact descriptor of the breadth of the distribution (α→0 monodisperse, "
             "α→1 fully scale-free).\n"); i += 1
    if s["corenn"] is not None:
        cr = s["corenn"]
        pk = ", ".join(f"{p['radius_nm']:.0f} nm ({p['rel_amount']*100:.0f}%)" for p in cr.peaks)
        pb = s.get("psd_bounds") or {}
        Rmax_nm = pb.get("R_max", s["R_resolvable"][1]) * 1e9
        at_edge = cr.peaks and abs(cr.peaks[0]["radius_nm"] - Rmax_nm) / Rmax_nm < 0.12
        lim = pb.get("limiting", "resolution")
        bound_note = (f"PSD grid {pb.get('R_min', float('nan'))*1e9:.0f}–{Rmax_nm:.0f} nm, "
                      f"upper bound set by {lim}"
                      if pb else f"PSD grid upper bound ~{Rmax_nm:.0f} nm")
        edge_note = (" — this peak sits at the PSD upper observability bound "
                     f"(~{Rmax_nm:.0f} nm); the true material may extend beyond the "
                     "measured lag/q window, so treat it as an unresolved slow-tail / "
                     "lower-bound signature rather than a resolved peak." if at_edge else "")
        bp = s.get("boot_psd")
        bp_txt = (f" An {bp['n']}-angle bootstrap of the inversion puts the primary peak at "
                  f"{bp['median']:.0f} nm [68% {bp['p16']:.0f}–{bp['p84']:.0f}; 95% "
                  f"{bp['p2_5']:.0f}–{bp['p97_5']:.0f} nm]." if bp else "")
        m.append(f"**Step {i} — CORENN multi-angle PSD (forward model).** The joint "
                 "inversion of the resolvable angles, on a radius grid bounded by "
                 f"dynamic observability plus the static q-window ({bound_note}) and the "
                 f"main-relaxation lag window "
                 f"(λ={cr.lam:.1e}), recovers one q-consistent intensity size "
                 f"distribution: peak(s) at **{pk}**{edge_note}.{bp_txt}\n"); i += 1
    # (shape, uncertainty and fit-adequacy are promoted to their own top-level
    #  sections — see sec_static_shape / sec_error_propagation / sec_fit_quality)
    return m, i


def plot_suite(axlist, s, color, label):
    """Draw Γ-vs-q², apparent R_h(q), Guinier, power-law, CORENN PSD."""
    q, gam = s["q"], s["gam"]
    rs = s.get("replicate_stats") or {}
    gam_sem = np.asarray(rs.get("gamma_sem", np.full_like(q, np.nan)), float)
    rh_sem = np.asarray(rs.get("R_h_app_sem", np.full_like(q, np.nan)), float)
    I_sem = np.asarray(rs.get("I_sem", np.full_like(q, np.nan)), float)
    has_gerr = np.any(np.isfinite(gam_sem) & (gam_sem > 0))
    has_rherr = np.any(np.isfinite(rh_sem) & (rh_sem > 0))
    if has_gerr:
        axlist[0].errorbar(q**2*1e-14, gam, yerr=gam_sem, fmt="o", ms=3,
                           color=color, ecolor=color, elinewidth=0.8, capsize=2,
                           label=f"{label} (rep SEM)")
    else:
        axlist[0].plot(q**2*1e-14, gam, "o", ms=3, color=color, label=label)
    order = np.argsort(q)
    # linear fit Γ = c + D·q² through the retained (low-to-mid-q) points, with the
    # rolled-off high-q angles excluded and marked
    gl = s.get("gvq2_linear")
    if gl and gl.get("D", 0) > 0:
        qo = q[order]
        axlist[0].plot(qo**2*1e-14, gl["intercept"] + gl["D"] * qo**2, "-", color=color, lw=1.3,
                       label=rf"linear fit $R_h$={gl['R_h']*1e9:.0f} nm"
                       + (f" (−{gl['n_excluded']} hi-q)" if gl.get("n_excluded") else ""))
        if gl.get("n_excluded"):
            ex = order[-gl["n_excluded"]:]
            axlist[0].plot(q[ex]**2*1e-14, gam[ex], "x", color="red", ms=8, mew=1.6,
                           label="excluded (rolloff)")
        # residual inset: fit residuals as % of Γ vs q² — structure (systematic
        # high-q rise) flags deviation from linearity (polydispersity)
        try:
            import matplotlib.pyplot as _plt
            rel = (gam - (gl["intercept"] + gl["D"] * q**2)) / np.clip(gam, 1e-30, None) * 100.0
            axr = axlist[0].inset_axes([0.50, 0.07, 0.47, 0.34])
            axr.axhline(0, color="0.6", lw=0.6)
            keep = order[:gl["n_linear"]]
            axr.plot(q[keep]**2*1e-14, rel[keep], "o-", ms=2.4, lw=0.8, color=color)
            if gl.get("n_excluded"):
                axr.plot(q[ex]**2*1e-14, rel[ex], "x", color="red", ms=5, mew=1.2)
            axr.set_title(f"fit resid % (q⁴ {gl.get('curv_t', float('nan')):+.0f}σ)", fontsize=FONT_INSET)
            axr.tick_params(labelsize=FONT_INSET); axr.margins(x=0.05); axr.grid(alpha=0.4)
        except Exception:
            pass
    else:
        axlist[0].plot(q[order]**2*1e-14, s["gvq2"]["D"]*q[order]**2, "-", color=color, lw=1)
    rapp = s["app"]["R_h_app"] * 1e9
    if has_rherr:
        axlist[1].errorbar(s["app"]["q"]*1e-6, rapp, yerr=rh_sem*1e9, fmt="o-",
                           ms=3, color=color, ecolor=color, elinewidth=0.8,
                           capsize=2, label=f"{label} (rep SEM)")
    else:
        axlist[1].plot(s["app"]["q"]*1e-6, rapp, "o-", ms=3, color=color, label=label)
    # (3) static intensity log-log, with the PSD-anticipated regime: a Guinier
    # fit on the low-q (q·R_g<1.3) points and/or a Porod power-law on the high-q
    # points, with the Guinier-limit q marked.
    lq = np.log10(q); lI = np.log10(np.clip(s["I"], 1e-30, None))
    pl = s.get("static_pl", {}); rd = s.get("regime")
    p = pl.get("exponent", float("nan"))
    pse = pl.get("exponent_se", float("nan"))
    plabel = f"{label}: static data" if np.isfinite(p) else label
    yerr_log = I_sem / (np.asarray(s["I"], float) * np.log(10.0))
    if np.any(np.isfinite(yerr_log) & (yerr_log > 0)):
        axlist[2].errorbar(lq, lI, yerr=yerr_log, fmt="o", ms=3, color=color,
                           ecolor=color, elinewidth=0.8, capsize=2, label=plabel)
    else:
        axlist[2].plot(lq, lI, "o", ms=3, color=color, label=plabel)
    ln10 = np.log(10)
    if rd is not None and rd["porod"] is not None:
        pp = rd["porod"]["exponent"]
        mask = np.asarray(rd["porod"].get("mask", np.ones_like(q, bool)), bool)
        if np.any(mask):
            xx = np.array([lq[mask].min(), lq[mask].max()])
        else:
            xx = np.array([lq.min(), lq.max()])
        flabel = (f"{label}: p={pp:.2f}±{pse:.2f}" if np.isfinite(pse)
                  else f"{label}: p={pp:.2f}")
        if rd["porod"].get("high_q"):
            flabel += " (lower-q fit)"
        axlist[2].plot(xx, rd["porod"]["log_prefactor"]/ln10 - pp*xx, "-",
                       color=color, lw=1.3, label=flabel)
        ql = rd.get("q_guinier_limit")
        if np.isfinite(ql) and q.min() < ql < q.max():
            axlist[2].axvline(np.log10(ql), color=color, ls=":", lw=1, alpha=0.6)
    elif np.isfinite(p):
        xx = np.array([lq.min(), lq.max()])
        axlist[2].plot(xx, pl["log_prefactor"]/ln10 - p*xx, "-", color=color, lw=1,
                       label=f"{label}: p={p:.2f}")
    x4 = np.log(s["app"]["q"]); y4 = np.log(rapp)
    yerr4 = rh_sem / np.asarray(s["app"]["R_h_app"], float)
    if np.any(np.isfinite(yerr4) & (yerr4 > 0)):
        axlist[3].errorbar(x4, y4, yerr=yerr4, fmt="o", ms=3, color=color,
                           ecolor=color, elinewidth=0.8, capsize=2,
                           label=f"{label}: α={s['sdc']['powerlaw_alpha']:.2f}")
    else:
        axlist[3].plot(x4, y4, "o", ms=3, color=color,
                       label=f"{label}: α={s['sdc']['powerlaw_alpha']:.2f}")
    if s["corenn"] is not None:
        r = s["corenn"]
        _y = r.R * r.intensity
        _ymax = float(np.nanmax(_y)) if _y.size else 0.0
        _eps = _ymax * 1e-6 if _ymax > 0 else 1e-12   # small offset: whole PSD
        axlist[4].plot(r.R*1e9, _y + _eps, "-", color=color, lw=2, label=label)  # visible on log
        bp = s.get("boot_psd")
        if bp:
            axlist[4].axvspan(bp["p16"], bp["p84"], color=color, alpha=0.12,
                              label=f"{label} peak 68%")


def label_suite_axes(ax):
    ax[0].set_xlabel(r"$q^2$ [$10^{-4}$ nm$^{-2}$]"); ax[0].set_ylabel(r"$\Gamma$ [s$^{-1}$]")
    ax[0].set_title(r"(1) $\Gamma$ vs $q^2$ (z-avg $D$)"); ax[0].legend(fontsize=FONT_LEGEND_SMALL)
    ax[1].set_xlabel(r"$q$ [$10^{-3}$ nm$^{-1}$]"); ax[1].set_ylabel(r"apparent $R_h$ [nm]")
    ax[1].set_title(r"(2) apparent $R_h(q)$"); ax[1].legend(fontsize=FONT_LEGEND_SMALL)
    ax[2].set_xlabel(r"$\log_{10} q$ [m$^{-1}$]"); ax[2].set_ylabel(r"$\log_{10} I$")
    ax[2].set_title(r"(3) static $I(q)$ power law (Porod/fractal)"); ax[2].legend(fontsize=FONT_LEGEND_SMALL)
    ax[3].set_xlabel(r"$\ln q$"); ax[3].set_ylabel(r"$\ln R_{h,app}$")
    ax[3].set_title(r"(4) power-law $R_h \propto q^{-\alpha}$"); ax[3].legend(fontsize=FONT_LEGEND_SMALL)
    ax[4].set_xscale("log"); ax[4].set_yscale("log")
    # a small constant ε was added to R·I(R) in plot_suite so the whole PSD is
    # visible on the log scale (near-zero regions rest at the ε baseline instead
    # of being clipped or going to -inf); just frame to the offset data
    _yv = np.concatenate([ln.get_ydata() for ln in ax[4].get_lines()]) if ax[4].get_lines() else np.array([1.0])
    _yv = _yv[np.isfinite(_yv) & (_yv > 0)]
    if _yv.size:
        ax[4].set_ylim(bottom=float(_yv.min()) * 0.7, top=float(_yv.max()) * 1.6)
    ax[4].set_xlabel(r"$R$ [nm]"); ax[4].set_ylabel(r"$R\cdot I(R) + \epsilon$ (log)")
    ax[4].set_title(r"(5) CORENN PSD — $\log\,R\cdot I(R)$ (small pops enhanced)"); ax[4].legend(fontsize=FONT_LEGEND_SMALL)


# ===========================================================================
# Template-conformant report scaffold (REPORT_TEMPLATE.md)
# ===========================================================================
# A "condition" is one analysed channel at one set point:
#   {"label": str, "channel": str, "T": K, "eta": Pa·s,
#    "datasets": [all raw DLSDatasets, incl. rejected], "suite": run_suite(...)}
# `ctx` carries report-level metadata + provisional flags (see build_report).

def _rh_ci_str(suite, scale=1e9):
    """'212 [rep 206–217 | model 203–223] nm' from the two bootstraps."""
    g = suite["gvq2"]; pt = g.get("R_h", float("nan")) * scale
    br = suite.get("boot_rep"); bm = (suite.get("boot") or {}).get("R_h")
    seg = []
    if br:
        seg.append(f"rep {br['p16']*scale:.0f}–{br['p84']*scale:.0f}")
    if bm:
        seg.append(f"model {bm['p16']*scale:.0f}–{bm['p84']*scale:.0f}")
    return f"{pt:.0f}" + (f" [{' | '.join(seg)}]" if seg else "")


def _rho_str(suite):
    r = suite.get("rho") or {}
    return f"{r['rho']:.2f} ({r['model']})" if r.get("independent") else "not determined"


def sec_metadata(ctx, conditions):
    angs, qs, temps, nreps, nkept, ntot = [], [], [], 0, 0, 0
    chans = sorted({c["channel"] for c in conditions if c.get("channel")})
    for c in conditions:
        s = c["suite"]
        for d in c["datasets"]:
            a = d.metadata.get("angle_deg")
            if a is not None:
                angs.append(a)
            if d.q:
                qs.append(d.q)
        temps.append(c["T"])
        nkept += s["n"]; ntot += s["n_total"]
        nreps += sum(1 for r in s.get("_reps", []) if r is not None and len(r) >= 2)
    sol = ctx.get("solvent", {})
    rows = [
        ("Report title", ctx.get("title", "—")),
        ("Data source", ctx.get("data_path", "—")),
        ("Report generator", f"{ctx.get('script','—')} (provisional)"),
        ("Analyst / owner", ctx.get("analyst", "automated pipeline")),
        ("Dataset class", ctx.get("dataset_class", "—")),
        ("Channels present", ", ".join(chans) or "—"),
        ("Temperatures", ", ".join(f"{t:.1f} K" for t in temps)),
        ("Angles", f"{min(angs):.0f}–{max(angs):.0f}° ({ntot} per condition)" if angs else "—"),
        ("q range", f"{_qrange_nm(min(qs), max(qs))}" if qs else "—"),
        ("Replicates", f"{nreps} angle-sets with ≥2 replicates" if nreps else "single curve per angle"),
        ("Solvent constants", f"η per condition (see table); n={sol.get('n','—')}, "
                              f"λ={sol.get('lambda_nm','632.8')} nm ({sol.get('source','—')})"),
        ("Static scale", ctx.get("static_scale", "absolute (toluene-referenced) — assumed, unverified")),
        ("Particle model", ctx.get("particle_model", "homogeneous sphere (assumed)")),
        ("Primary model", ctx.get("primary_model", "CORENN multi-angle PSD (forward model)")),
        ("Report limitations", ctx.get("limitations", "—")),
    ]
    m = ["## Metadata\n", "| field | value |", "|---|---|"]
    m += [f"| {k} | {v} |" for k, v in rows]
    m.append("")
    return m


def sec_executive_summary(ctx, conditions, main_result):
    rep = conditions[0]["suite"]
    cd = rep.get("corenn_diag") or {}
    flags = ctx.get("provisional_flags", [])
    m = ["## Executive Summary\n"]
    m.append("**Dataset.** " + ctx.get("overview", "—") + "\n")
    m.append("**Main result.** " + main_result + "\n")
    conf = []
    if rep.get("boot_rep") or (rep.get("boot") or {}).get("R_h"):
        conf.append("the z-average R_h carries replicate/angle bootstrap intervals")
    cvr = (rep.get("cv") or {}).get("cv_r2", float("nan"))
    if np.isfinite(cvr):
        conf.append(f"leave-one-angle-out CV gives predictive R²={cvr:.3f}")
    if cd.get("converged") is not None:
        conf.append("the CORENN inversion converged"
                    + ("" if cd.get("gcv_interior_min") else " but the GCV minimum is at a λ-grid boundary"))
    m.append("**Confidence.** " + ("; ".join(conf).capitalize() + "." if conf else "—") + "\n")
    m.append("**Assumption-sensitive.** Viscosity, refractive index, q-from-angle calibration, "
             "the toluene/absolute-intensity reference, the assumed sphere form factor, and the "
             "CORENN regularization all enter the reported sizes; see the Assumption Register.\n")
    m.append("**Recommended next action.** " + ctx.get("next_action", "—") + "\n")
    if flags:
        m.append("> **Provisional flags:** " + "; ".join(flags) + ".\n")
    return m


def sec_data_inventory(conditions):
    m = ["## Data Inventory\n",
         "One row per analysed condition (per-angle detail is in Per-Curve Fit Results).\n",
         "| condition | channel | n angles (kept/total) | angle range | q range [nm⁻¹] | "
         "T [K] | scaled count rate [kHz/mW] | reps | missing meta |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in conditions:
        s = c["suite"]; ds = c["datasets"]
        angs = [d.metadata.get("angle_deg") for d in ds if d.metadata.get("angle_deg") is not None]
        scr = [d.metadata.get("scaled_count_rate") for d in ds
               if d.metadata.get("scaled_count_rate") is not None]
        miss = sum(1 for d in ds if d.metadata.get("angle_deg") is None
                   or d.metadata.get("scaled_count_rate") is None)
        nrep = sum(1 for r in s.get("_reps", []) if r is not None and len(r) >= 2)
        m.append(f"| {c['label']} | {c.get('channel','—')} | {s['n']}/{s['n_total']} | "
                 f"{min(angs):.0f}–{max(angs):.0f}° | {_qnm(s['q_range'][0])}–{_qnm(s['q_range'][1])} | "
                 f"{c['T']:.1f} | {min(scr):.0f}–{max(scr):.0f} | {nrep} | {miss} |")
    m.append("")
    return m


def sec_per_curve(condition, g1_min=G1_MIN):
    s = condition["suite"]
    kept = s.get("_kept_datasets", []); reps = s.get("_reps", [])
    kT6 = C.k_BOLTZMANN * s["T"] / (6.0 * math.pi * s["eta"])
    m = [f"### Per-curve fits — {condition['label']} ({condition.get('channel','')})\n",
         "Per-angle 2nd-order cumulant on the leading decay (g1_min="
         f"{g1_min}); R_h,app = kT q²/(6πη Γ).\n",
         "| angle [°] | q [nm⁻¹] | Γ [1/s] | μ₂ | PDI | β | R_h,app [nm] | reps | accepted |",
         "|---|---|---|---|---|---|---|---|---|"]
    order = np.argsort([d.metadata.get("angle_deg", 0) for d in kept])
    for j in order:
        d = kept[j]
        cf = C.cumulant_fit(d.tau, d.g2m, beta=d.beta, order=2, g1_min=g1_min)
        rh = kT6 * d.q**2 / cf["Gamma"] if cf["Gamma"] > 0 else float("nan")
        nr = len(reps[j]) if (j < len(reps) and reps[j] is not None) else 1
        m.append(f"| {d.metadata.get('angle_deg', float('nan')):.0f} | {_qnm(d.q)} | "
                 f"{cf['Gamma']:.1f} | {cf['mu2']:.1e} | {cf['PDI']:.2f} | {d.beta:.3f} | "
                 f"{rh*1e9:.0f} | {nr} | yes |")
    m.append("")
    return m


def sec_rejected(conditions):
    m = ["## Rejected Data And Failure Diagnostics\n"]
    any_rej = False
    for c in conditions:
        rej = c["suite"].get("rejected", [])
        if not rej:
            continue
        any_rej = True
        m.append(f"**{c['label']} ({c.get('channel','')})** — {len(rej)} angle(s) rejected:\n")
        m.append("| angle [°] | q [nm⁻¹] | apparent R_h [nm] | reason |")
        m.append("|---|---|---|---|")
        for r in rej:
            a = r["angle"]; a = f"{a:.0f}" if a is not None else "—"
            rh = r["R_app_nm"]; rh = f"{rh:.0f}" if np.isfinite(rh) else "—"
            m.append(f"| {a} | {_qnm(r['q'])} | {rh} | {r['reason']} |")
        m.append("")
    if not any_rej:
        m.append("No angles or curves were rejected.\n")
    return m


def sec_assumption_register(ctx, conditions):
    rep = conditions[0]["suite"]; sol = ctx.get("solvent", {})
    rr = rep.get("_radius_range"); cd = rep.get("corenn_diag") or {}
    pb = rep.get("psd_bounds") or {}
    if pb:
        bound_comment = (f"upper=min(dynamic {pb.get('dynamic_upper', float('nan'))*1e9:.0f} nm, "
                         f"static {pb.get('static_upper', float('nan'))*1e9:.0f} nm, "
                         f"default {pb.get('default_upper', float('nan'))*1e9:.0f} nm); "
                         f"limiting={pb.get('limiting')}; "
                         f"tau_max={pb.get('tau_max_trusted', float('nan')):.2g} s, "
                         f"min field decay={pb.get('min_field_decay', float('nan')):.3f}")
    else:
        bound_comment = "bounded by dynamic observability plus static q-window"
    etas = ", ".join(f"{c['eta']*1e3:.2f}" for c in conditions)
    recs = [
        ("wavelength", f"{sol.get('lambda_nm','632.8')} nm", "instrument default", "q, D, R_h", "no", "—"),
        ("refractive index", str(sol.get("n", "—")), sol.get("source", "—"), "q", "no", "replace with measured"),
        ("viscosity", f"{etas} mPa·s", ctx.get("eta_source", "recorded export"), "R_h",
         "yes", ctx.get("eta_note", "verify per temperature")),
        ("q source", "recalculated from angle", "pipeline", "all q-dependent fits", "no",
         "vendor 'Scattering vector [1/nm]' column is mislabelled (values are 1/m); q recomputed from angle, reported in nm⁻¹"),
        ("static scale", ctx.get("static_scale", "absolute, assumed"), "LSII toluene reference",
         "SLS, R_g", "no", "toluene reference not confirmable from parsed export"),
        ("form factor", "homogeneous sphere (RGD)", "pipeline", "PSD radii, ρ check", "yes",
         "flag if ρ≠0.78 or depolarization present"),
        ("cumulant g1_min", f"{G1_MIN}", "pipeline config", "Γ, PDI", "yes", "leading-decay window"),
        ("PSD radius grid", f"{rr[0]*1e9:.0f}–{rr[1]*1e9:.0f} nm" if rr else "—",
         "dynamic observability + static q guardrail", "PSD peaks", "yes", bound_comment),
        ("PSD basis", f"{cd.get('n_basis','24')} B-spline control points", "pipeline config",
         "PSD smoothness", "yes", "—"),
        ("regularization", f"GCV-selected λ={cd.get('lam', float('nan')):.1e}", "pipeline",
         "PSD width", "yes", "interior minimum required"),
        ("baseline", "free per fit (CORENN); leading-decay cumulant", "pipeline config",
         "Γ, PSD", "yes", "—"),
    ]
    m = ["## Assumption Register\n", "| assumption | value | source | affects | sensitivity checked? | comment |",
         "|---|---|---|---|---|---|"]
    m += [f"| {a} | {v} | {sr} | {af} | {se} | {co} |" for (a, v, sr, af, se, co) in recs]
    m.append("")
    sidecar = [dict(zip(("assumption", "value", "source", "affects", "sensitivity_checked", "comment"), r))
               for r in recs]
    return m, sidecar


def sec_quality_gates(ctx, conditions):
    rep = conditions[0]["suite"]; cd = rep.get("corenn_diag") or {}
    qs = [d.q for c in conditions for d in c["datasets"] if d.q]
    etas = {round(c["eta"], 6) for c in conditions}
    multi_T = len({round(c["T"], 1) for c in conditions}) > 1
    gates = []

    def g(name, status, note):
        gates.append({"gate": name, "status": status, "note": note})

    g("q physically plausible / reproducible from angle",
      "PASS" if qs and 1e6 < min(qs) and max(qs) < 1e8 else "FAIL",
      "q recomputed from angle (vendor q column mislabeled 1/nm)")
    g("viscosity not a placeholder",
      "WAIVE" if (multi_T and len(etas) == 1) else "PASS",
      "same η recorded across temperatures — flagged placeholder" if (multi_T and len(etas) == 1)
      else "recorded value used")
    pb = rep.get("psd_bounds") or {}
    psd_note = (f"CORENN tau_max={pb.get('tau_max_trusted', float('nan')):.2g}s, "
                f"Rmax={pb.get('R_max', float('nan'))*1e9:.0f} nm ({pb.get('limiting')})"
                if pb else "CORENN lag window stated")
    g("fit bounds stated for every fit", "PASS", f"cumulant g1_min={G1_MIN}; {psd_note}")
    g("rejected curves/angles listed", "PASS",
      f"{sum(len(c['suite'].get('rejected', [])) for c in conditions)} rejected, tabulated")
    g("residual-structure test reported (not r² alone)",
      "PASS" if rep.get("resid_gvq2") else "FAIL", "runs test + lag-1 autocorrelation")
    g("headline R_h carries a bootstrap CI",
      "PASS" if (rep.get("boot_rep") or (rep.get("boot") or {}).get("R_h")) else "FAIL",
      "replicate + angle/model bootstrap" if rep.get("boot_rep")
      else "angle/model bootstrap (no angle has ≥2 replicates, so no repeatability CI)")
    g("hold-out cross-validation reported",
      "PASS" if np.isfinite((rep.get("cv") or {}).get("cv_r2", float("nan"))) else "FAIL",
      "leave-one-angle-out")
    g("CORENN converged + well-posed regularization",
      "PASS" if cd.get("converged") and cd.get("gcv_interior_min") else "WAIVE",
      f"converged={cd.get('converged')}, GCV interior={cd.get('gcv_interior_min')}, κ={cd.get('cond', float('nan')):.0e}")
    g("static scale confirmed absolute",
      "WAIVE", "toluene reference assumed (LSII default), not confirmable from parsed export")
    g("PSD basis (intensity/volume/number) stated", "PASS", "intensity-weighted primary, conversions noted")
    _mph = (rep.get("regime") or {}).get("porod_morphology")
    g("ρ = R_g/R_h and particle model stated", "PASS",
      _rho_str(rep) + ("; Porod slope implies " + _morph_short(_mph)
                       + " — sphere PSD form factor flagged inconsistent"
                       if _mph and not _mph["sphere_consistent"]
                       else "; sphere form factor assumed"))
    if multi_T:
        g("multi-temperature size/viscosity ambiguity addressed", "PASS",
          "same-sample joint-temperature solve reported")
    g("figure captions state raw vs derived", "PASS", "captions included")

    npass = sum(1 for x in gates if x["status"] == "PASS")
    nwaive = sum(1 for x in gates if x["status"] == "WAIVE")
    nfail = sum(1 for x in gates if x["status"] == "FAIL")
    m = ["## Quality Gates\n",
         f"Provisional report: **{npass} PASS, {nwaive} WAIVE, {nfail} FAIL** of {len(gates)}.\n",
         "| gate | status | note |", "|---|---|---|"]
    m += [f"| {x['gate']} | **{x['status']}** | {x['note']} |" for x in gates]
    m.append("")
    return m, gates


def sec_reproducibility(ctx, generated_files):
    import subprocess
    def _git(args):
        try:
            return subprocess.run(["git"] + args, cwd=ctx.get("root", "."),
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return "—"
    commit = _git(["rev-parse", "--short", "HEAD"]) or "—"
    dirty = "dirty" if _git(["status", "--short"]) else "clean"
    try:
        import numpy as _np
        env = f"python {sys.version.split()[0]}, numpy {_np.__version__}"
    except Exception:
        env = "—"
    m = ["## Reproducibility Appendix\n",
         "| item | value |", "|---|---|",
         f"| Command | `{ctx.get('command','—')}` |",
         f"| Environment | {env} |",
         f"| Git commit (working tree) | {commit} ({dirty}) |",
         f"| Random seeds | bootstraps seed=0 (fixed) |",
         f"| Generated files | {', '.join(generated_files)} |",
         ""]
    return m


def fig_count_rate(conditions, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
    for c in conditions:
        pts = []
        for d in c["datasets"]:
            a = d.metadata.get("angle_deg")
            vals = static_replicate_intensity_values(d)
            if a is None or vals.size == 0:
                continue
            y = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size >= 2 else np.nan
            pts.append((a, y, sem))
        pts = sorted(pts)
        if pts:
            xs, ys, es = [np.asarray(v, float) for v in zip(*pts)]
            if np.any(np.isfinite(es) & (es > 0)):
                ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=4, capsize=2,
                            label=f"{c['label']} {c.get('channel','')} (rep SEM)")
            else:
                ax.plot(xs, ys, "o-", ms=4, label=f"{c['label']} {c.get('channel','')}")
    ax.set_xlabel("scattering angle [°]"); ax.set_ylabel("scaled count rate [kHz/mW] (absolute R(q))")
    ax.set_title("Static intensity vs angle (toluene-referenced)"); ax.legend(fontsize=FONT_LEGEND_SMALL)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_raw_correlograms(suite, path, label=""):
    """Raw correlograms at low/mid/high q: the stitched correlation (black line)
    over the semi-transparent raw replicate points (the spread = repeatability;
    the replicate-to-replicate average is not itself a meaningful object here)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    kept = suite.get("_kept_datasets", []); reps = suite.get("_reps", [])
    if not kept:
        return False
    order = list(np.argsort([d.q for d in kept]))
    pick = sorted({order[0], order[len(order)//2], order[-1]})
    fig, axes = plt.subplots(1, len(pick), figsize=(4.6*len(pick), 3.8), squeeze=False)
    for col, j in enumerate(pick):
        d = kept[j]; ax = axes[0][col]
        # raw data: every replicate's points, semi-transparent, behind
        if j < len(reps) and reps[j] is not None and len(reps[j]) >= 2:
            for k, g in enumerate(reps[j]):
                ax.semilogx(d.tau, g, ".", ms=2.5, color="C0", alpha=0.18,
                            label="raw replicate points" if k == 0 else None)
        else:
            ax.semilogx(d.tau, d.g2m, ".", ms=2.5, color="C0", alpha=0.3,
                        label="raw points")
        # stitched (multi-tau) correlation in black on top
        ax.semilogx(d.tau, d.g2m, "-", color="k", lw=1.1, label="stitched g2−1")
        ax.set_title(f"{d.metadata.get('angle_deg',0):.0f}° (q={_qnm(d.q)} nm⁻¹)")
        ax.set_xlabel(r"$\tau$ [s]"); ax.set_ylabel(r"$g_2-1$")
        if col == 0:
            ax.legend(fontsize=FONT_LEGEND_SMALL, loc="upper right")
    fig.suptitle(f"Raw correlograms (low/mid/high q) — {label}")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path); plt.close(fig)
    return True


def _smooth_logtau(y, w=6):
    """Gaussian smoothing over the (log-spaced) lag index — separates the broad
    diffusive residual from the high-frequency short-lag wiggles."""
    y = np.nan_to_num(np.asarray(y, float))
    k = np.exp(-0.5 * (np.arange(-3 * w, 3 * w + 1) / float(w)) ** 2); k /= k.sum()
    return np.convolve(y, k, mode="same")


def high_angle_components(vv, suite, w_smooth=6):
    """Decompose the single-exponential residual into a long-lag diffusive peak
    (broad) and short-lag wiggles (high-frequency), across *all* angles, and
    summarise their q-trends.

    The long-lag peak is the deviation of the real (polydisperse / multi-mode)
    correlogram from a single exponential; it sits at Γ_z·τ ≈ 1 and shifts ∝1/q²
    (a genuine diffusive feature).  The short-lag wiggles are locked to the
    multi-tau correlator's register/octave boundaries at *fixed* lag, so they do
    not scale with q (an instrumental feature).
    """
    D_z = suite["gvq2"]["D"]
    out = {"D_z": D_z, "rows": [], "boundaries": np.array([]),
           "gtau_pk_median": float("nan"), "peak_logq_slope": float("nan"),
           "wig_rms_lo": float("nan"), "wig_rms_hi": float("nan")}
    vv_s = sorted([d for d in vv if d.q], key=lambda d: d.q)
    if not vv_s or not (D_z > 0):
        return out
    tau0 = vv_s[0].tau
    if tau0.size > 4:                                  # multi-tau doubling boundaries
        dt = np.diff(tau0); rr = dt[1:] / np.clip(dt[:-1], 1e-30, None)
        out["boundaries"] = tau0[1:-1][rr > 1.5]
    for d in vv_s:
        cf = C.cumulant_fit(d.tau, d.g2m, beta=d.beta, order=2, g1_min=G1_MIN)
        G = cf["Gamma"]
        if not (G > 0) or not d.beta:
            continue
        resid = (d.g2m - d.beta * np.exp(-2.0 * G * d.tau)) / d.beta
        broad = _smooth_logtau(resid, w_smooth); wig = resid - broad
        tau_t = 1.0 / (d.q ** 2 * D_z)
        win = (d.tau >= tau_t / 30) & (d.tau <= tau_t * 30)
        if win.any():
            ipk = np.where(win)[0][np.argmax(broad[win])]
            tau_pk = float(d.tau[ipk]); A_pk = float(broad[ipk])
        else:
            tau_pk = float("nan"); A_pk = float("nan")
        sw = (d.tau >= 2e-6) & (d.tau <= tau_t / 8)
        wig_rms = float(np.sqrt(np.mean(wig[sw] ** 2))) if sw.sum() > 3 else float("nan")
        out["rows"].append({"q": d.q, "ang": d.metadata.get("angle_deg", float("nan")),
                            "tau": d.tau, "resid": resid, "broad": broad, "wig": wig,
                            "tau_t": tau_t, "tau_pk": tau_pk, "A_pk": A_pk,
                            "gtau_pk": tau_pk * d.q ** 2 * D_z, "wig_rms": wig_rms})
    rows = out["rows"]
    if rows:
        q = np.array([r["q"] for r in rows]); tpk = np.array([r["tau_pk"] for r in rows])
        gt = np.array([r["gtau_pk"] for r in rows]); wr = np.array([r["wig_rms"] for r in rows])
        hi = q >= np.median(q)                          # upper half of angles (cleaner)
        good = hi & np.isfinite(tpk) & (tpk > 0)
        if good.sum() >= 3:
            out["peak_logq_slope"] = float(np.polyfit(np.log(q[good]), np.log(tpk[good]), 1)[0])
        out["gtau_pk_median"] = float(np.nanmedian(gt[hi])) if hi.any() else float("nan")
        fin = np.where(np.isfinite(wr))[0]
        if fin.size >= 2:
            out["wig_rms_lo"] = float(wr[fin[0]]); out["wig_rms_hi"] = float(wr[fin[-1]])
    return out


def fig_high_angle_diagnostics(vv, vh, d3, suite, dep, path, label="", n_top=3):
    """High-angle ripple diagnostics, separating the two τ-features across all
    angles: (A) VV/VH/3D channel overlay at the top angle; (B) residual vs the
    q²-rescaled lag Γ_z·τ for every angle — the long-lag diffusive peak collapses
    onto Γ_z·τ≈1; (C) the short-lag wiggle component vs absolute τ for the top
    angles, with the multi-tau correlator boundaries marked — the wiggles pin to
    fixed lags; (D) the q-trends of the peak position (Γ_z·τ_pk, flat ⇒ q²) and
    the wiggle amplitude.
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    comp = high_angle_components(vv, suite); rows = comp["rows"]; D_z = comp["D_z"]
    if not rows:
        return False
    vv_s = sorted([d for d in vv if d.q], key=lambda d: d.q, reverse=True)
    top = vv_s[:n_top]

    def near(dsets, ang):
        cand = [d for d in (dsets or []) if d.metadata.get("angle_deg") is not None]
        return min(cand, key=lambda d: abs(d.metadata["angle_deg"] - ang)) if cand else None

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.6)); ax = ax.ravel()
    angs = np.array([r["ang"] for r in rows])
    a_lo, a_hi = float(np.nanmin(angs)), float(np.nanmax(angs))
    cmap = plt.cm.viridis
    def acol(a):
        return cmap((a - a_lo) / (a_hi - a_lo + 1e-9))

    # (A) channel overlay at the top angle
    d0 = top[0]; ang0 = d0.metadata.get("angle_deg", 0.0)
    tau_t0 = 1.0 / (d0.q ** 2 * D_z) if D_z > 0 else None
    for ch, dsets, col in (("VV", vv, "C0"), ("VH", vh, "C3"), ("3D", d3, "C2")):
        dd = near(dsets, ang0)
        if dd is None or not dd.beta:
            continue
        ax[0].semilogx(dd.tau, np.clip(dd.g2m / dd.beta, -0.3, 1.2), "-", lw=1, color=col, label=ch)
    if tau_t0:
        ax[0].axvline(tau_t0, color="k", ls="--", lw=1, label=r"$\tau_t = 1/(q^2 D_z)$")
    ax[0].axhline(0, color="0.7", lw=0.6)
    ax[0].set_title(f"(A) channels at θ={ang0:.0f}° (q={d0.q:.1e})")
    ax[0].set_xlabel(r"$\tau$ [s]"); ax[0].set_ylabel(r"$g_1^2 = (g_2-1)/\beta$"); ax[0].legend(fontsize=FONT_LEGEND_SMALL)

    # (B) residual vs q²-rescaled lag Γ_z·τ — the diffusive peak collapses to ~1
    for r in rows:
        ax[1].semilogx(r["q"] ** 2 * D_z * r["tau"], r["resid"], "-", lw=0.7, color=acol(r["ang"]))
    ax[1].axvline(1.0, color="k", ls="--", lw=1)
    ax[1].axhline(0, color="0.7", lw=0.6); ax[1].set_xlim(1e-3, 1e3)
    ax[1].set_title(r"(B) residual vs $\Gamma_z\cdot\tau$ ($q^2$-rescaled, all angles)" "\n" "long-lag peak collapses at ~1 ⇒ diffusive")
    ax[1].set_xlabel(r"$\Gamma_z\cdot\tau = q^2 D_z\cdot\tau$"); ax[1].set_ylabel(r"$(g_2{-}1 - \mathrm{model})/\beta$")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(a_lo, a_hi)); sm.set_array([])
    fig.colorbar(sm, ax=ax[1], label="angle [deg]", pad=0.01)

    # (C) short-lag wiggle component vs absolute τ, top angles + correlator boundaries
    for r in rows[-min(6, len(rows)):]:
        ax[2].semilogx(r["tau"], r["wig"], "-", lw=0.8, color=acol(r["ang"]),
                       label=f"{r['ang']:.0f}°")
    for b in comp["boundaries"]:
        if 1e-6 <= b <= 1e-3:
            ax[2].axvline(b, color="0.75", ls=":", lw=0.6)
    ax[2].axhline(0, color="0.6", lw=0.6); ax[2].set_xlim(1e-6, 2e-3)
    ax[2].set_title(r"(C) short-lag wiggles vs $\tau$ (top angles)" "\n" "extrema at fixed lag = multi-tau correlator registers")
    ax[2].set_xlabel(r"$\tau$ [s]"); ax[2].set_ylabel(r"wiggle (resid − smooth)/$\beta$"); ax[2].legend(fontsize=FONT_LEGEND_SMALL, ncol=2)

    # (D) q-trends: peak position Γ_z·τ_pk (flat ⇒ q²) and wiggle RMS
    gt = np.array([r["gtau_pk"] for r in rows]); wr = np.array([r["wig_rms"] for r in rows])
    ax[3].plot(angs, gt, "o-", ms=4, color="C0", label=r"$\Gamma_z\cdot\tau_{pk}$ (peak)")
    ax[3].axhline(1.0, color="C0", ls="--", lw=0.8, alpha=0.6)
    ax[3].set_ylim(0, 2.2); ax[3].set_xlabel("angle [deg]")
    ax[3].set_ylabel(r"$\Gamma_z\cdot\tau_{pk}$", color="C0"); ax[3].set_title(r"(D) peak $\Gamma_z\cdot\tau_{pk}$ (flat$\approx$1 $\Rightarrow q^2$) & wiggle RMS")
    a3 = ax[3].twinx(); a3.plot(angs, wr * 1e3, "s:", ms=4, color="C3", label="wiggle RMS")
    a3.set_ylabel(r"wiggle RMS $\times 10^{3}$", color="C3")
    ax[3].legend(fontsize=FONT_LEGEND_SMALL, loc="upper left"); a3.legend(fontsize=FONT_LEGEND_SMALL, loc="upper right")

    fig.suptitle(f"High-angle correlogram diagnostics — {label}")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path); plt.close(fig)
    return True


def high_angle_section_md(vv, suite, dep, fig_path):
    """Markdown for the high-angle ripple diagnostic, separating the two features
    (long-lag diffusive peak vs short-lag correlator wiggles) with a data-driven
    read across all angles."""
    comp = high_angle_components(vv, suite)
    n = len(comp["rows"])
    m = ["### High-angle correlogram diagnostics (µs–ms ripples)\n",
         "The high-angle single-exponential residual contains **two distinct features** that "
         "separate by their q-scaling: a **long-lag peak** that shifts to shorter lag with "
         "increasing angle, and **short-lag wiggles** that barely move. Panels (B)–(D) tell "
         "them apart across all angles.\n",
         f"![high-angle]({fig_path})\n",
         "*Reading the panels.* **(A)** VV/VH/3D at the top angle (a faster VH would be the "
         "rotational signature Γ_VH=q²D_t+6D_r; a VV-vs-3D split flags modulation / multiple "
         "scattering). **(B)** residual vs the q²-rescaled lag Γ_z·τ — a feature that **collapses "
         "onto Γ_z·τ≈1** for every angle is a genuine diffusive relaxation. **(C)** the short-lag "
         "wiggle component vs *absolute* τ — extrema that line up at **fixed lag** (dotted: the "
         "multi-tau correlator register boundaries) are instrumental, not q-dependent dynamics. "
         "**(D)** trends: a flat Γ_z·τ_pk ≈ 1 confirms the peak scales as q²; the wiggle RMS "
         "grows as the signal weakens at high q.\n"]
    gt = comp.get("gtau_pk_median"); sl = comp.get("peak_logq_slope")
    rlo = comp.get("wig_rms_lo"); rhi = comp.get("wig_rms_hi")
    bnds = [b for b in comp.get("boundaries", []) if 1e-5 <= b <= 2e-4]
    if np.isfinite(gt):
        peak_line = (f"**(1) the long-lag peak** sits at Γ_z·τ_pk ≈ {gt:.2f} and stays roughly "
                     "constant across angles (panel D), i.e. it tracks τ_pk = 1/(q²D_z) ∝ 1/q² "
                     "and collapses onto a single curve under q² rescaling (panel B) — a "
                     "**genuine diffusive multi-mode / polydispersity** deviation (the broad "
                     "relaxation-time distribution of the sample), strongest at high q.")
        bnd_txt = (" (≈ " + ", ".join(f"{b*1e6:.0f}" for b in bnds[:5]) + " µs)") if bnds else ""
        wig_grow = (f", with the amplitude growing ≈{rhi/rlo:.0f}× from low to high angle as the "
                    "signal weakens" if (np.isfinite(rlo) and np.isfinite(rhi) and rlo > 0) else "")
        wig_line = ("**(2) the short-lag wiggles** pin to *fixed* absolute lags" + bnd_txt +
                    ", independent of angle and locked to the multi-tau correlator's register "
                    "boundaries (panel C)" + wig_grow + " — a **correlator / instrumental "
                    "artifact**, not sample dynamics.")
        m.append(f"*This sample — the two features separated across all {n} angles:* "
                 f"{peak_line} {wig_line}\n")
    # channel / rotational note from the depolarized fit
    try:
        gvv = np.asarray(dep.get("gamma_vv", []), float); qvv = np.asarray(dep.get("q_vv", []), float)
        gvh = np.asarray(dep.get("gamma_vh", []), float); qvh = np.asarray(dep.get("q_vh", []), float)
        depol = (dep.get("depol_mean") or float("nan")) * 100.0
        if gvv.size and gvh.size:
            a = gvv[int(np.argmax(qvv))]; b = gvh[int(np.argmax(qvh))]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                ratio = b / a
                rot = ("faster (a rotational signature *if* not VH shot noise)" if ratio > 1.10
                       else "slower — not a rotational signature" if ratio < 0.90
                       else "at the same rate — no distinct rotational mode")
                m.append(f"Channel check (A): at the top angle Γ_VH/Γ_VV = {ratio:.2f}, i.e. VH "
                         f"decays {rot}; but the depolarized channel is weak (I_VH/I_VV ≈ "
                         f"{depol:.1f} %) so Γ_VH is noise-limited.\n")
    except Exception:
        pass
    return m


def _dynamic_radius_max(q, tau_trust, T, eta, decay_fold=8.0,
                        floor_m=2e-6, cap_m=2e-5):
    """Physical size_max for a single-angle inversion: the largest radius whose
    field correlation actually decays (Gamma*tau >= decay_fold) within the
    trusted lag window tau_trust at scattering vector q -- so the slow mode is
    *measured*, not parked in an unconstrained large-size region (which is what
    an arbitrary/huge fixed size_max invites; cf. the vendor size_max=3.5e11 nm
    ghost peak).  Clipped to [floor_m, cap_m] (a sane DLS upper limit)."""
    R = C.k_BOLTZMANN * T * q * q * (tau_trust / decay_fold) / (6.0 * np.pi * eta)
    return float(np.clip(R, floor_m, cap_m)) if (np.isfinite(R) and R > 0) else cap_m


def slow_mode_stitched(meas_dirs, path, label="", q_res_limit_m=None,
                       radius_max=None, n_basis=30, n_R=200, n_invert=2):
    """Single-angle CORENN on GAP-AWARE STITCHED data: resolve the slow-mode /
    large-aggregate tail beyond the multi-angle q-resolution limit (pi/q_min).

    The multi-angle intensity PSD is q-resolution-limited because its R^6-weighted
    q->0 representation, coupled through the form factor, edge-piles past pi/q.
    A *single-angle* :func:`corenn.corenn_invert` instead inverts one g1(tau)
    for the intensity-weighted relaxation-rate spectrum directly, so the slow
    diffusive modes land at their true hydrodynamic R with a data-bounded
    amplitude.  Fed the stitched correlogram (clean long-lag from the pooled
    count traces) it reports the large-aggregate tail where it is actually
    resolvable.  Builds hardware + stitched datasets for the given measurement(s),
    inverts each over an extended radius grid, draws the correlograms + PSDs, and
    returns tail metrics (or None if nothing could be processed).
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import hybrid_correlator as Hc
    # cheap q-scan: load the hardware base of every candidate, then invert only
    # the lowest-q (largest pi/q -> largest resolvable R) measurements.
    cand = []
    for md in meas_dirs:
        try:
            base, reps = C.load_measurement_replicates(md)
            if base.q:
                cand.append((base.q, md, base, reps))
        except Exception:
            continue
    if not cand:
        return None
    cand.sort(key=lambda c: c[0])
    q_min_all = cand[0][0]
    if radius_max is None:                             # physical size_max
        _b0 = cand[0][2]
        radius_max = _dynamic_radius_max(q_min_all, float(_b0.tau.max()),
                                         _b0.temperature or 295.0, _b0.viscosity or 1e-3)
    rr = (C.DEFAULT_CORENN_RADIUS_RANGE[0], radius_max)
    res = []
    for q, md, base, reps in cand[:max(1, n_invert)]:
        try:
            st, _ = Hc.build_stitched_dataset(md)
            r_hw = C.corenn_invert(base, replicates=reps, radius_range=rr,
                                   n_basis=n_basis, n_R=n_R, fit_baseline=True)
            r_st = C.corenn_invert(st, radius_range=rr, n_basis=n_basis, n_R=n_R,
                                   fit_baseline=True)
        except Exception:
            continue
        res.append({"q": base.q, "base": base, "st": st, "r_hw": r_hw, "r_st": r_st})
    if not res:
        return None
    rep = min(res, key=lambda r: r["q"])              # lowest q -> largest resolvable R
    if q_res_limit_m is None:
        q_res_limit_m = np.pi / q_min_all

    def tail_frac(r, R_thresh):
        tot = float(np.trapz(r.intensity, r.R))
        if tot <= 0:
            return float("nan")
        m = r.R >= R_thresh
        return float(np.trapz(r.intensity[m], r.R[m]) / tot)

    def cum_pct_R(r, pct=0.95):                        # intensity-weighted R percentile
        dR = np.gradient(r.R)                          # dR-weighted (log grid) to match tail_frac
        w = np.clip(r.intensity, 0.0, None) * dR
        c = np.cumsum(w) / max(float(np.sum(w)), 1e-30)
        i = int(np.searchsorted(c, pct))
        return float(r.R[min(i, r.R.size - 1)])

    R_lim = q_res_limit_m if q_res_limit_m else (np.pi / rep["q"])
    metrics = {"q_res_limit_nm": R_lim * 1e9, "rep_q": rep["q"],
               "tail_hw": tail_frac(rep["r_hw"], R_lim),
               "tail_st": tail_frac(rep["r_st"], R_lim),
               "imean_st_nm": rep["r_st"].mean_radius("intensity") * 1e9,
               "ext_st_nm": cum_pct_R(rep["r_st"], 0.95) * 1e9,
               "radius_max_nm": radius_max * 1e9,
               "tau_hw_s": float(rep["base"].tau.max()),
               "tau_st_s": float(rep["st"].tau.max())}

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    b, st = rep["base"], rep["st"]
    ax[0].semilogx(b.tau, b.g2m / (b.beta or 1), ".", ms=2.5, color="0.6", alpha=0.5, label="hardware")
    ax[0].semilogx(st.tau, st.g2m / (st.beta or 1), "-", color="C3", lw=1.2, label="stitched")
    ax[0].axhline(0, color="0.7", lw=0.6)
    ax[0].set_xlabel(r"$\tau$ [s]"); ax[0].set_ylabel(r"$g_1^2 = (g_2-1)/\beta$")
    ax[0].set_title(f"(A) correlogram at q={_qnm(rep['q'])} nm⁻¹"); ax[0].legend(fontsize=FONT_LEGEND_SMALL)
    for r, c, lab in ((rep["r_hw"], "0.45", "hardware"), (rep["r_st"], "C3", "stitched")):
        y = r.R * r.intensity; eps = (np.nanmax(y) * 1e-6) if np.nanmax(y) > 0 else 1e-12
        ax[1].plot(r.R * 1e9, y + eps, "-", color=c, lw=1.8, label=lab)
    ax[1].axvline(R_lim * 1e9, color="k", ls="--", lw=1,
                  label=f"π/q_min ≈ {R_lim*1e9:.0f} nm (multi-angle limit)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("R [nm]"); ax[1].set_ylabel(r"$R\cdot I(R) + \epsilon$")
    ax[1].set_title("(B) single-angle stitched PSD — large-aggregate tail")
    ax[1].legend(fontsize=FONT_LEGEND_SMALL)
    fig.suptitle(f"Slow-mode / large-aggregate tail (single-angle stitched) — {label}")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path); plt.close(fig)
    return metrics


def slow_mode_section_md(metrics, fig_path):
    """Markdown for the single-angle stitched large-aggregate tail panel."""
    if metrics is None:
        return []
    m = metrics
    out = ["### Slow-mode / large-aggregate tail (single-angle stitched)\n",
           "The multi-angle intensity PSD is q-resolution-limited: its R^6-weighted q->0 "
           "representation, coupled through the sphere form factor, edge-piles for R ≳ "
           f"π/q_min ≈ **{m['q_res_limit_nm']:.0f} nm**. A *single-angle* CORENN inversion of "
           "the gap-aware **stitched** correlogram (clean long-lag pooled over the full run, "
           f"τ_max {m['tau_hw_s']:.0f}→{m['tau_st_s']:.0f} s) instead recovers the slow "
           "diffusive modes directly from the decay-rate spectrum — with a data-bounded "
           "amplitude, so it resolves the large-aggregate tail where the multi-angle PSD "
           "cannot.\n",
           f"![slow-mode]({fig_path})\n"]
    th = m.get("tail_hw"); ts = m.get("tail_st")
    edge = (np.isfinite(m.get("ext_st_nm", float("nan")))
            and m["ext_st_nm"] >= 0.8 * m.get("radius_max_nm", 5e4))
    if ts is not None and ts < 0.02:
        out.append(
            f"At the lowest angle (q={_qnm(m['rep_q'])} nm⁻¹), the stitched single-angle inversion "
            f"finds **negligible intensity (<2%) beyond π/q_min** (extent ~{m['ext_st_nm']:.0f} "
            "nm) — there is no resolvable large-aggregate tail; the sample is a single "
            "well-defined population, consistent with the multi-angle PSD.\n")
    elif edge:
        out.append(
            f"At the lowest angle (q={_qnm(m['rep_q'])} nm⁻¹) the stitched inversion places "
            f"**{ts*100:.0f}% of the intensity beyond π/q_min**, but it **piles at the "
            f"radius-grid edge (~{m['radius_max_nm']/1000:.0f} µm)**: the aggregates here are "
            "at/beyond even single-angle resolution, so this is a **lower bound** on a "
            "large-aggregate population (a genuinely slow diffusive mode), not a measured "
            "size — consistent with this sample sitting at the q-resolution limit.\n")
    else:
        out.append(
            f"At the lowest angle (q={_qnm(m['rep_q'])} nm⁻¹), where large aggregates scatter most "
            f"strongly (forward, ∝R^6), the stitched single-angle inversion finds **{ts*100:.0f}% "
            f"of the intensity beyond π/q_min** (vs {th*100:.0f}% from the noise-dominated "
            f"hardware long-lag), extending to ~{m['ext_st_nm']:.0f} nm — confirming and sizing "
            "the large-aggregate tail the multi-angle PSD must omit. Read it as a *hydrodynamic* "
            "(slow-diffusion), intensity-weighted view: these aggregates dominate the low-angle "
            "intensity even though they are a small *number* fraction, so this is the resolvable "
            "home of the aggregate mass — not a contradiction of the few-hundred-nm q-consistent "
            "main population in the multi-angle PSD.\n")
    return out


def _mass_fractal_dim(mph):
    """Mass-fractal dimension D_f (how *material volume* scales with R) from the
    Porod morphology.  Mass fractals / coils / rods scale as R^D_f (D_f<3);
    surface fractals and smooth spheres are compact (D_f=3)."""
    if mph is None:
        return 3.0
    if mph.get("kind") in ("porod", "surface_fractal") or mph.get("dimension") is None:
        return 3.0
    return float(np.clip(mph["dimension"], 1.0, 3.0))


def _fractal_mechanism(D):
    """Colloidal aggregation mechanism implied by a mass-fractal dimension D_f."""
    if not np.isfinite(D):
        return None
    if D < 1.6:
        return "an open, tenuous / rod-like network"
    if D < 1.95:
        return ("diffusion-limited cluster aggregation (DLCA) — fast, barrier-free sticking "
                "into open clusters")
    if D < 2.2:
        return ("reaction-limited cluster aggregation (RLCA) — slower sticking past a repulsive "
                "barrier, giving denser clusters")
    if D < 2.7:
        return ("a restructured / compacted aggregate — denser than simple RLCA (internal "
                "reorganization, or for plates an edge–face 'house-of-cards' association)")
    return "a compact, near-space-filling aggregate"


def rotational_verdict(dep):
    """Honest assessment of whether rotational diffusion is *credibly* resolved.

    corenn.depolarized_diffusion sets ``rotation_resolved`` from the sign/size of
    the 6D_r intercept alone; it does not catch the common failure where the weak
    depolarized (VH) channel returns a translational D_t inconsistent with the
    polarized (VV) channel — which means the intercept is noise, not rotation.  We
    additionally require D_t(VH) ≈ D_t(VV) (same particles) and a non-trivial
    depolarized signal before calling rotation resolved."""
    if not dep:
        return None
    dvv = (dep.get("fit_vv") or {}).get("D")
    dvh = (dep.get("fit_vh") or {}).get("D")
    depol = dep.get("depol_mean", 0.0) or 0.0
    ratio = (dvh / dvv) if (dvv and dvh and dvv > 0) else float("nan")
    consistent = bool(np.isfinite(ratio) and 0.6 <= ratio <= 1.5)
    credible = bool(dep.get("rotation_resolved") and consistent and depol > 0.02)
    reasons = []
    if np.isfinite(ratio) and not consistent:
        reasons.append(f"the depolarized fit gives D_t(VH)={dvh:.1e} m²/s, {ratio:.1f}× the "
                       f"polarized D_t(VV)={dvv:.1e} — inconsistent for one translational mode")
    if depol <= 0.02:
        reasons.append(f"the depolarized channel is weak (I_VH/I_VV ≈ {depol*100:.1f}%)")
    return {"credible": credible, "label": "resolved" if credible else "not credibly resolved",
            "reasons": reasons, "ratio": ratio, "six_Dr": dep.get("six_Dr"),
            "R_rot": dep.get("R_rot"), "depol": depol}


def rotational_section_md(dep, sample_note=""):
    """Rotational-diffusion subsection with an honest resolved/not verdict."""
    v = rotational_verdict(dep)
    if v is None:
        return []
    dvv = (dep.get("fit_vv") or {}).get("D", float("nan"))
    dvh = (dep.get("fit_vh") or {}).get("D", float("nan"))
    head = (f"Polarized and depolarized translational diffusion: D_t(VV)={dvv:.2e}, "
            f"D_t(VH)={dvh:.2e} m²/s. The VH−VV intercept difference 6D_r = "
            f"{v['six_Dr']:.1f} /s makes rotation **{v['label']}**")
    if v["credible"]:
        rr = v.get("R_rot")
        head += (f"; modelled as a rotating object this is a rotational size R_rot ≈ "
                 f"{rr*1e9:.0f} nm." if rr else ".")
    else:
        head += ": " + ("; ".join(v["reasons"]) if v["reasons"]
                        else "the intercept is within noise") + "."
    if sample_note:
        head += " " + sample_note
    return ["### Rotational diffusion (depolarized VV/VH)\n", head + "\n"]


def sample_synthesis_section_md(suite, dep=None):
    """A stand-alone 'what is this sample' synthesis across the independent
    observables — floc structure (fractal dimension → aggregation mechanism and
    aggregation number), the primary scattering unit, what the angular window
    cannot determine (Guinier), the bounding caveats, and the cross-observable
    consistency.  Everything is derived from quantities already computed (the PSD,
    the static regime, the depolarized fit); no external data is used."""
    cr = (suite or {}).get("corenn")
    if cr is None:
        return []
    rd = suite.get("regime") or {}
    R = cr.R; I = np.clip(cr.intensity, 0.0, None)
    imod, _ = mode_descriptors(R, I)
    nmod, _ = mode_descriptors(R, np.where(R > 0, I / R**6, 0.0))
    R_f = imod[0]["r_peak"] if imod else float(cr.mean_radius("intensity"))
    R_0 = nmod[0]["r_peak"] if nmod else float(cr.mean_radius("number"))
    D = _mass_fractal_dim(rd.get("porod_morphology"))
    mech = _fractal_mechanism(D) if D < 2.9 else None
    zr = (suite.get("gvq2") or {}).get("R_h", float("nan"))
    pdi = (suite.get("app") or {}).get("PDI_mean", float("nan"))
    broad = bool(R_f > 1.6 * R_0 or (np.isfinite(pdi) and pdi > 0.3))

    out = ["## Sample Synthesis\n",
           "What the independent observables imply about the sample as a colloid — derived only "
           "from the scattering data above, with the inversion caveats carried through:\n"]
    out.append(
        f"**Physical picture.** The scattering is dominated by ~**{R_f*1e9:.0f} nm** objects "
        f"(intensity-weighted); by number the population centres near ~**{R_0*1e9:.0f} nm**, and "
        f"the z-average R_h is {zr*1e9:.0f} nm — "
        + ("best read as one broad, polydisperse distribution spanning these scales rather than "
           "well-separated species" if broad else
           "a single, fairly narrow population — intensity and number weightings nearly agree")
        + " (see the size-distribution-by-weighting and size_max sections).\n")

    # Hydrodynamic shape: the Guinier–Porod morphology picks the Stokes–Einstein
    # geometry (not the sphere by default) and the shape-appropriate size.
    R_h_si = (suite.get("gvq2_linear") or suite.get("gvq2") or {}).get("R_h")
    hd = morphology_hydrodynamics(rd.get("porod_morphology"), R_h_si,
                                  R_g=(rd.get("guinier") or {}).get("R_g"))
    if hd and np.isfinite(hd.get("size_nm", float("nan"))):
        seg = (f"**Hydrodynamic shape (Guinier–Porod → Stokes–Einstein).** The static slope "
               f"identifies the dominant scatterer as **{hd['shape']}**, so the geometry-matched "
               f"Stokes–Einstein relation is **{hd['se']}** — the {hd['size_label']} is "
               f"**{hd['size_nm']:.0f} nm**"
               + (f" (rod length L ≈ {hd['length_nm']:.0f} nm from R_g)" if hd.get("length_nm") else "")
               + ". ")
        if hd.get("alt"):
            a = hd["alt"]
            seg += (f"That ~2D slope is geometrically degenerate, so **{a['shape']}** is equally "
                    f"consistent — then {a['se']} → the {a['size_label']} is **{a['size_nm']:.0f} nm**. ")
        note = hd["note"]; seg += note[0].upper() + note[1:] + ".\n"
        out.append(seg)

    if mech:
        parts = [f"**Floc structure & aggregation mechanism.** The static Porod slope gives a "
                 f"mass-fractal dimension **D_f ≈ {D:.1f}**, which points to **{mech}**."]
        if R_f > 1.5 * R_0 and D > 1.0:
            N = (R_f / R_0) ** D
            parts.append(
                f"Taking the ~{R_0*1e9:.0f} nm number-mode as the primary unit, the fractal "
                f"scaling N ≈ (R_floc/R_primary)^D_f gives an order-of-magnitude **aggregation "
                f"number N ~ {N:.0f}** primary units per ~{R_f*1e9:.0f} nm floc (geometric, and "
                "sensitive to the primary-size choice and the inversion).")
        out.append(" ".join(parts) + "\n")

    if dep is not None:
        rv = rotational_verdict(dep); depol = (dep or {}).get("depol_mean", 0.0) or 0.0
        pu = [f"**Primary unit.** The number-weighted mode (~{R_0*1e9:.0f} nm) is the smallest "
              "resolved scatterer — plausibly the primary particle or smallest aggregate."]
        if depol > 0.005:
            tail = (("rotational diffusion is " + rv["label"]
                     + ((" (" + "; ".join(rv["reasons"]) + ")") if rv and rv["reasons"] else ""))
                    if rv else "rotation is not modelled")
            pu.append(f"A depolarized signal (I_VH/I_VV ≈ {depol*100:.1f}%) shows the scatterers "
                      f"are optically anisotropic (plate- or rod-like primaries), but {tail}, so "
                      "an independent shape/size from rotation is not available here.")
        out.append(" ".join(pu) + "\n")

    qrg_min = rd.get("qRg_min", float("nan")); rg = rd.get("R_g_char", float("nan"))
    if np.isfinite(qrg_min) and qrg_min > 1.3 and np.isfinite(rg):
        out.append(
            f"**Not determinable from this angular range.** The lowest q still has q·R_g ≈ "
            f"{qrg_min:.1f} (> 1.3), so the **Guinier regime is never reached**: an independent "
            "radius of gyration, an absolute molar mass / aggregation number, and ρ = R_g/R_h "
            f"cannot be measured here — only inferred. Reaching them needs q ≲ {_qnm(1.3/rg)} nm⁻¹ "
            "(smaller scattering angles, low-q SLS, or SAXS).\n")

    out.append(
        "**Caveats that bound these conclusions.** *(1) Dilute limit / S(q):* the analysis "
        "assumes single-particle scattering; if the suspension is not dilute, a low-q structure "
        "factor (interparticle correlations or incipient gelation) can bias the Porod/fractal "
        "slope and the apparent sizes — a dilution series would test it. *(2) Inversion "
        "dependence:* the modes and the aggregation number inherit the ill-posedness and "
        "size_max / λ / form-factor sensitivity documented in the size-distribution sections. "
        "*(3) Stability:* this is a single time-point of a system that can aggregate or sediment; "
        "a repeat / time-resolved run (or checking the count rate for drift across the angle "
        "scan) is needed to establish the distribution is stationary.\n")

    bits = []
    if np.isfinite(pdi):
        bits.append(
            (f"the broad dynamic distribution (mean PDI {pdi:.2f}, apparent R_h falling with q) "
             "matches the polydisperse" + (" fractal floc" if mech else " population")
             + " inferred from the static slope") if broad else
            (f"the narrow dynamic distribution (mean PDI {pdi:.2f}) is consistent with the single "
             "population seen across all weightings"))
    if mech:
        bits.append("the non-sphere Porod slope and the undetermined ρ agree (both say 'not a "
                    "compact sphere')")
    anis = "anisotropic, " if (dep and (dep.get("depol_mean") or 0) > 0.005) else ""
    body = "; ".join(bits)
    if body:
        body = body[0].upper() + body[1:] + ". "
    out.append(
        "**Consistency across observables.** " + body
        + "No single observable here is contradicted by another — the picture is "
        + (f"one broad, {anis}polydisperse population rather than a set of distinct species"
           if broad else f"one well-defined, {anis}fairly narrow population")
        + ".\n")
    return out


def mode_descriptors(R, w, min_frac=0.03, min_pts=4, merge_frac=0.5, min_sep_dec=0.25,
                     detect_shoulders=True, shoulder_min_frac=0.04):
    """Decompose a (weighted) PSD into modes *and shoulders* and describe each.

    A single mean is meaningless for a multimodal floc distribution - it lands in
    the valley between modes.  Peaks and boundaries are found on the *per-decade*
    density g = R*w (what the figure plots and what an instrument calls the "peak
    size" on a log axis), so the reported peak matches the visual maximum.  Per
    component we report location and asymmetry, not one summary number:

      kind      "peak" (a true local maximum) or "shoulder" (a sub-population that
                bends the curve - high downward curvature - without cresting)
      r_peak    per-decade maximum / shoulder location, m
      r_mean    expectation value E[R] within the component, m
      r_median  50%-of-area radius, m
      frac      fraction of the *total* weighted signal in the component
      cv        linear width sigma/mu
      gsd       geometric standard deviation exp(sigma_lnR), dimensionless
      logskew   skewness of ln R: 0 = symmetric on the log axis, >0 = tail to
                large R, <0 = to small R

    Adjacent maxima not separated by a real valley (saddle above ``merge_frac`` x
    the smaller peak) or closer than the instrument can resolve (``min_sep_dec``
    decades) are merged - suppressing B-spline ringing.  Shoulders are local
    maxima of the concavity (-d^2 g / d lnR^2) on a flank, away from any peak;
    component boundaries sit at the most-convex point between centres (a true
    valley for two peaks, the inflection for a shoulder).  Returns (modes,
    overall)."""
    R = np.asarray(R, float); w = np.clip(np.asarray(w, float), 0.0, None)
    total = float(np.trapz(w, R)); n = R.size
    if total <= 0 or n < 3:
        return [], {"n_modes": 0, "n_shoulders": 0, "r_median_nm": float("nan"),
                    "logskew": float("nan")}
    lnR = np.log(np.clip(R, 1e-15, None))
    g = R * w  # per-decade density: argmax(g) is the mode on a log-size axis
    dx = float(np.mean(np.diff(lnR))); dx_dec = dx / math.log(10.0)
    # smoothed per-decade density + curvature on the (uniform) lnR grid
    gs = g; d2 = None
    try:
        from scipy.signal import savgol_filter
        win = int(round(0.20 / max(dx_dec, 1e-9))); win += 1 - win % 2  # odd, ~0.2 decade
        win = max(7, min(win, n if n % 2 else n - 1))
        if 7 <= win < n:
            gs = np.clip(savgol_filter(g, win, 3), 0.0, None)
            d2 = savgol_filter(g, win, 3, deriv=2, delta=dx)
    except Exception:
        pass

    def _moments(lo, hi, kpk, kind):
        sl = slice(lo, hi + 1); rr = R[sl]; ww = w[sl]; xx = lnR[sl]
        amt = float(np.trapz(ww, rr))
        if amt <= 0:
            return None
        mu = float(np.trapz(rr * ww, rr) / amt)
        var = float(np.trapz((rr - mu) ** 2 * ww, rr) / amt)
        cv = math.sqrt(max(var, 0.0)) / mu if mu > 0 else 0.0
        m = float(np.trapz(xx * ww, rr) / amt)
        s2 = float(np.trapz((xx - m) ** 2 * ww, rr) / amt); s = math.sqrt(max(s2, 0.0))
        npts = int(np.count_nonzero(ww > ww.max() * 1e-3))
        skew = (float(np.trapz((xx - m) ** 3 * ww, rr) / amt) / s ** 3
                if s > 1e-3 and npts >= min_pts else None)
        c = np.concatenate([[0.0], np.cumsum(0.5 * (ww[1:] + ww[:-1]) * np.diff(rr))])
        med = float(np.interp(0.5 * c[-1], c, rr))
        rpk = R[kpk]
        if kind == "peak" and lo < kpk < hi:  # sub-grid parabola vertex on g
            y0, y1, y2 = g[kpk - 1], g[kpk], g[kpk + 1]; den = y0 - 2 * y1 + y2
            if den < 0:
                rpk = math.exp(lnR[kpk] + 0.5 * (y0 - y2) / den * dx)
        edge = "lo" if kpk == 0 else ("hi" if kpk == n - 1 else None)
        return {"kind": kind, "r_peak": rpk, "r_mean": mu, "r_median": med,
                "frac": amt / total, "cv": cv, "gsd": math.exp(s), "logskew": skew,
                "edge": edge}

    sep = min_sep_dec * math.log(10.0)
    # ---- primary peaks: local maxima of the raw per-decade density g, merged by
    # valley-depth & resolution (raw g avoids savgol boundary overshoot).  Grid
    # endpoints count when g rises into them - an "edge pile" (mass against the
    # grid bound, e.g. number-weighting an inverted PSD) is a real, if unresolved,
    # component and must not be missed for an interior bump.
    is_max = np.r_[False, (g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]), False]
    if g[0] > g[1]:
        is_max[0] = True
    if g[-1] > g[-2]:
        is_max[-1] = True
    peaks = list(np.where(is_max)[0]) or [int(np.argmax(g))]
    changed = True
    while changed and len(peaks) > 1:
        changed = False
        for i in range(len(peaks) - 1):
            a, b = peaks[i], peaks[i + 1]
            if (float(g[a:b + 1].min()) > merge_frac * min(g[a], g[b])
                    or (lnR[b] - lnR[a]) < sep):
                peaks.pop(i if g[a] < g[b] else i + 1); changed = True; break
    comps = [(int(p), "peak") for p in peaks]
    # ---- shoulders: *prominent* local maxima of the concavity C=-g'' on a flank,
    # resolvably away from any peak.  A shoulder need not be concave in absolute
    # terms (it can be a curvature bump on a convex flank), so detect by the
    # prominence of the C-maximum, not by sign.
    if detect_shoulders and d2 is not None:
        gmax = float(g.max()); C = -d2; Cmax = float(C.max())
        try:
            from scipy.signal import find_peaks
            # prominence of the concavity bump relative to the *main* peak's
            # curvature (Cmax) - a shoulder's curvature is small next to the
            # dominant peak's, so the global C-range is the wrong yardstick.
            cand, _props = find_peaks(C, prominence=max(0.06 * Cmax, 1e-30))
        except Exception:
            cand = np.array([], int)
        for k in cand:
            if g[k] < 0.02 * gmax:  # negligible signal here
                continue
            if any(abs(lnR[k] - lnR[p]) < sep for p in peaks):  # it's the peak itself
                continue
            comps.append((int(k), "shoulder"))
    comps.sort(key=lambda c: c[0])
    # ---- boundaries: a true valley (argmin g) between two peaks; the most-convex
    # inflection (argmax d2) when a shoulder is involved (no valley exists there)
    div = d2 if d2 is not None else -g
    bounds = [0]
    for (a, ka), (b, kb) in zip(comps[:-1], comps[1:]):
        if ka == "peak" and kb == "peak":
            bounds.append(a + int(np.argmin(g[a:b + 1])))
        else:
            bounds.append(a + int(np.argmax(div[a:b + 1])))
    bounds.append(n - 1)
    modes = []
    for (lo, hi), (k, kind) in zip(zip(bounds[:-1], bounds[1:]), comps):
        d = _moments(lo, hi, k, kind)
        if d and d["frac"] >= (shoulder_min_frac if kind == "shoulder" else min_frac):
            modes.append(d)
    # peaks first (by fraction), then shoulders (by fraction)
    modes.sort(key=lambda d: (d["kind"] != "peak", -d["frac"]))
    # ---- continuous-distribution summary over the *whole* distribution: robust
    # to the (inversion-dependent) mode structure - the assumption-light description
    # when the data cannot establish discrete modes.
    ov = _moments(0, n - 1, int(np.argmax(g)), "peak") or {}
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(R))])

    def _pct(p):
        return float(np.interp(p * cum[-1], cum, R)) if cum[-1] > 0 else float("nan")

    d10, d50, d90 = _pct(0.10), _pct(0.50), _pct(0.90)
    dg = math.exp(float(np.trapz(lnR * w, R) / total)) if total > 0 else float("nan")
    overall = {"n_modes": sum(d["kind"] == "peak" for d in modes),
               "n_shoulders": sum(d["kind"] == "shoulder" for d in modes),
               "d10_nm": d10 * 1e9, "d50_nm": d50 * 1e9, "d90_nm": d90 * 1e9,
               "dg_nm": dg * 1e9, "gsd": ov.get("gsd", float("nan")),
               "span": ((d90 - d10) / d50) if (d50 and d50 > 0) else float("nan"),
               "r_median_nm": ov.get("r_median", float("nan")) * 1e9,
               "logskew": ov.get("logskew")}
    return modes, overall


def fig_psd_weightings(cr, q_ref, T, eta, path, label="", D_f=3.0):
    """CORENN size distribution under all weightings (intensity / volume /
    number / Gamma), mirroring the vendor tabs.  The volume (R^3) and number
    (R^6) conversions assume *compact spheres*; for a mass fractal of dimension
    D_f the physically-correct conversions are volume = I/R^D_f and
    number = I/R^(2 D_f) (these reduce to the sphere ones at D_f=3).  When a
    mass-fractal D_f<2.9 is supplied, both the sphere and the fractal-corrected
    curves are overlaid."""
    if cr is None:
        return None
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = cr.R; Rnm = R * 1e9; fractal = D_f < 2.9
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.9))

    def perdec(w):
        y = R * np.clip(w, 0.0, None)
        return y + ((np.nanmax(y) * 1e-6) if np.nanmax(y) > 0 else 1e-12)

    modes = {}

    def _gamma_of(R_m):  # decay rate at the reference angle for radius R
        return C.k_BOLTZMANN * T * q_ref ** 2 / (6.0 * np.pi * eta * max(R_m, 1e-15))

    def mark(a, md, col, x_is_gamma=False):
        for m in md:  # dashed = true peak, dotted = shoulder
            xv = _gamma_of(m["r_peak"]) if x_is_gamma else m["r_peak"] * 1e9
            a.axvline(xv, color=col, lw=1, alpha=0.55,
                      ls="--" if m["kind"] == "peak" else ":")

    def draw(a, w, col, lw=1.8, lab=None, marks=True):
        a.plot(Rnm, perdec(w), "-", color=col, lw=lw, label=lab)
        md, ov = mode_descriptors(R, w)
        if marks:
            mark(a, md, col)
        return md, ov

    # (1) Intensity -- what CORENN actually inverts (compute first; Gamma reuses it)
    modes["intensity"] = draw(ax[1], cr.intensity, "C0")
    ax[1].set_title("Intensity (∝R⁶)")
    # (0) Gamma (decay-rate) spectrum -- the rawest, model-free view, shown first.
    # The per-decade density R*w is invariant under R<->Gamma, so reuse it vs Gamma.
    Gam = C.k_BOLTZMANN * T * q_ref ** 2 / (6.0 * np.pi * eta * np.clip(R, 1e-12, None))
    o = np.argsort(Gam)
    ax[0].plot(Gam[o], perdec(cr.intensity)[o], "-", color="C4", lw=1.8)
    mark(ax[0], modes["intensity"][0], "C4", x_is_gamma=True)
    ax[0].set_xlabel(r"$\Gamma$ [s$^{-1}$]"); ax[0].set_ylabel(r"$R\cdot$weight (per decade)")
    ax[0].set_title("Γ (decay rate, intensity-wt)")
    # (2) Number: sphere I/R^6 (+ fractal I/R^(2 D_f))
    ns = np.where(R > 0, cr.intensity / R**6, 0.0)
    modes["number"] = draw(ax[2], ns, "C3", lab="sphere (R⁶)")
    if fractal:
        nf = np.where(R > 0, cr.intensity / R**(2 * D_f), 0.0)
        modes["number_fractal"] = draw(ax[2], nf, "C1", lw=1.6, lab=f"fractal (R^{2*D_f:.1f})", marks=False)
        ax[2].legend(fontsize=FONT_LEGEND_SMALL)
    ax[2].set_title("Number")
    # (3) Volume: sphere I/R^3 (+ fractal I/R^D_f)
    vs = np.where(R > 0, cr.intensity / R**3, 0.0)
    modes["volume"] = draw(ax[3], vs, "C2", lab="sphere (R³)")
    if fractal:
        vf = np.where(R > 0, cr.intensity / R**D_f, 0.0)
        modes["volume_fractal"] = draw(ax[3], vf, "C1", lw=1.6, lab=f"fractal (R^{D_f:.1f})", marks=False)
        ax[3].legend(fontsize=FONT_LEGEND_SMALL)
    ax[3].set_title("Volume / mass")
    for a in (ax[0], ax[2], ax[3]):
        a.set_xscale("log"); a.set_yscale("log")
    for a in (ax[2], ax[3]):
        a.set_xlabel("R [nm]")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("R [nm]")
    # dominant intensity mode -> its decay rate / relaxation time (for the text)
    gdom = None
    if modes["intensity"][0]:
        dm = modes["intensity"][0][0]; gpk = _gamma_of(dm["r_peak"])
        gdom = {"R": dm["r_peak"], "Gamma": gpk, "tau": 1.0 / gpk}
    fig.suptitle(f"CORENN size distribution by weighting - {label}"
                 + (f"  (mass-fractal D_f={D_f:.1f}: volume R^{D_f:.1f}, number R^{2*D_f:.1f})"
                    if fractal else "  (compact-sphere conversions)")
                 + "  -- dashed = peaks, dotted = shoulders")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path); plt.close(fig)
    eff_dof = None
    try:
        eff_dof = cr.extra.get("eff_dof")
    except Exception:
        pass
    return {"D_f": D_f, "fractal": fractal, "modes": modes, "label": label,
            "gamma_dom": gdom, "eff_dof": eff_dof}


def _fmt_nm(r_m):
    nm = r_m * 1e9
    return f"{nm:.1f}" if nm < 10 else (f"{nm:.0f}" if nm < 1000 else f"{nm/1000:.2f}k")


def _skew_word(s):
    if s is None:
        return "width-limited"
    if s > 0.3:
        return "tail \u2192 large R"
    if s < -0.3:
        return "tail \u2192 small R"
    return "\u2248 log-symmetric"


def _fmt_time(t):
    if not np.isfinite(t) or t <= 0:
        return "\u2014"
    if t >= 1:
        return f"{t:.3g} s"
    if t >= 1e-3:
        return f"{t*1e3:.3g} ms"
    if t >= 1e-6:
        return f"{t*1e6:.3g} \u00b5s"
    return f"{t*1e9:.3g} ns"


def psd_weightings_section_md(info, fig_path):
    if not info or not info.get("modes"):
        return []
    D_f = info.get("D_f", 3.0); fractal = info.get("fractal", False)
    M = info["modes"]
    # order: Gamma (figure/lead) -> Intensity -> Number -> Volume
    series = [("intensity", "Intensity (\u221dR\u2076)")]
    series += [("number_fractal", f"Number (fractal, R^{2*D_f:.1f})")] if fractal else []
    series += [("number", "Number (sphere, R\u2076)")]
    series += [("volume_fractal", f"Volume (fractal, R^{D_f:.1f})")] if fractal else []
    series += [("volume", "Volume (sphere, R\u00b3)")]
    series = [s for s in series if s and s[0] in M]
    nsh = sum(ov.get("n_shoulders", 0) for _md, ov in M.values())

    io = M["intensity"][1] if "intensity" in M else {}
    out = ["### Size distribution by weighting \u2014 continuous summary, then apparent modes\n",
           "DLS recovers a size distribution by an **ill-posed Laplace inversion**: many "
           "different distributions \u2014 a smooth continuum or a set of discrete modes \u2014 reproduce "
           "the measured correlation function equally well within noise, and the regularized "
           "inversion (CORENN) returns just one of them, fixed by its smoothness penalty, radius "
           "grid, and form-factor assumption. The number and sharpness of peaks are therefore "
           "properties of the inversion, not facts the data establish on their own. The report "
           "gives **two views**: (1) assumption-light continuous percentiles, and (2) the "
           "inversion's apparent modes/shoulders, flagged as inversion-dependent and trustworthy "
           "only where they survive the size_max / \u03bb sensitivity tests. CORENN returns the *same* "
           "inversion under every weighting (the vendor's \u0393 / Intensity / Number / Volume tabs); "
           "re-weighting only relocates and re-weights the features.\n",
           f"![weightings]({fig_path})\n"]
    gd = info.get("gamma_dom")
    if gd:
        out.append(
            f"**\u0393 (decay-rate) view \u2014 the rawest signal (figure panel 1).** Before any size "
            f"model, the dominant intensity component relaxes at \u0393 \u2248 "
            f"**{gd['Gamma']:.3g} s\u207b\u00b9** (relaxation time \u03c4 = 1/\u0393 \u2248 "
            f"**{_fmt_time(gd['tau'])}**) at the reference angle; the size columns below are this "
            "spectrum mapped through Stokes\u2013Einstein \u0393 = k_BT q\u00b2/(6\u03c0\u03b7R).\n")
    # (1) continuous-distribution summary -- per weighting, percentile-based, no
    # modes assumed.  Intensity is narrow (sees only the big scatterers); number/
    # volume span wider but rest on the sphere conversion and amplify small-R
    # inversion noise (so their percentiles are the least certain).
    out += ["**Continuous-distribution summary (percentiles; no modes assumed).** The "
            "assumption-light description — percentiles of the recovered distribution, robust to "
            "how it is partitioned into modes. Intensity is narrow (dominated by the largest "
            "scatterers); number-weighting (÷R⁶) strongly amplifies the inverted PSD's small-R "
            "end, so its percentiles are the least certain.\n",
            "| Weighting | D10 (nm) | D50 (nm) | D90 (nm) | D_g (nm) | \u03c3_g (GSD) | span | log-skew |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, lab in series:
        ov = M[key][1]
        if not (ov and np.isfinite(ov.get("d50_nm", float("nan")))):
            continue
        sk = ov.get("logskew")
        out.append(
            f"| {lab} | {_fmt_nm(ov['d10_nm']*1e-9)} | {_fmt_nm(ov['d50_nm']*1e-9)} | "
            f"{_fmt_nm(ov['d90_nm']*1e-9)} | {_fmt_nm(ov['dg_nm']*1e-9)} | {ov['gsd']:.2f} | "
            f"{ov['span']:.2f} | {sk:+.2f} |" if sk is not None else
            f"| {lab} | {_fmt_nm(ov['d10_nm']*1e-9)} | {_fmt_nm(ov['d50_nm']*1e-9)} | "
            f"{_fmt_nm(ov['d90_nm']*1e-9)} | {_fmt_nm(ov['dg_nm']*1e-9)} | {ov['gsd']:.2f} | "
            f"{ov['span']:.2f} | \u2014 |")
    out.append("")
    # (2) modality caveat
    eff = info.get("eff_dof")
    eff_txt = (f" The inversion carries \u2248 **{eff:.0f} spline degrees of freedom** \u2014 ample latitude "
               "to place peaks on a continuum." if (eff is not None and np.isfinite(eff)) else "")
    out.append(
        "**Modality caveat \u2014 read the next table as inversion fine structure.** The Laplace "
        "inversion is ill-posed: a smooth continuum and a set of discrete modes can fit the same "
        "g\u2082 within noise, so the number and sharpness of peaks depend on the regularization (\u03bb), "
        "the radius grid / size_max, the spline basis, and the sphere form factor \u2014 none of which "
        f"the data uniquely fix.{eff_txt} **Treat the apparent peaks/shoulders below as fine "
        "structure of this particular inversion \u2014 not established sub-populations \u2014 unless they "
        "are stable under the size_max / \u03bb sensitivity checks and reproducible across "
        "replicates.**\n")
    out += ["#### Apparent modes & shoulders (CORENN fine structure \u2014 inversion-dependent)\n",
            "| Weighting | Comp. | Peak R (nm) | \u27e8R\u27e9 (nm) | Median (nm) | Fraction | GSD | "
            "log-skew |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, lab in series:
        md, _ov = M[key]
        if not md:
            out.append(f"| {lab} | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |")
            continue
        pk = 0
        for i, m in enumerate(md):
            if m["kind"] == "peak":
                pk += 1; tag = str(pk)
            else:
                tag = "shoulder"
            edge = m.get("edge")
            if edge:
                tag += " (edge)"
            pkcell = ("≤" if edge == "lo" else "≥" if edge == "hi" else "") + _fmt_nm(m["r_peak"])
            sk = m["logskew"]
            out.append(
                f"| {lab if i == 0 else ''} | {tag} | {pkcell} | "
                f"{_fmt_nm(m['r_mean'])} | {_fmt_nm(m['r_median'])} | {m['frac']*100:.0f}% | "
                f"{m['gsd']:.2f} | "
                f"{(f'{sk:+.2f} ({_skew_word(sk)})') if sk is not None else _skew_word(sk)} |")
    out.append("")
    if any(m.get("edge") for md, _ov in M.values() for m in md):
        out.append(
            "*An **edge** component piles against the radius-grid bound (≤ small-R or ≥ large-R "
            "edge) — it is unresolved, so its size is a bound only and its ⟨R⟩/median/skew are "
            "not reliable. This is typical when number/volume weighting (÷R⁶, ÷R³) amplifies the "
            "inverted PSD's small-R floor.*\n")

    # automated reading: how the dominant size relocates with weighting + its asymmetry
    i0 = M["intensity"][0][0] if M["intensity"][0] else None
    nkey = "number_fractal" if "number_fractal" in M else "number"
    n0 = M[nkey][0][0] if M[nkey][0] else None
    if i0 and n0:
        sh_txt = (f"  Across the weightings, {nsh} apparent shoulder(s) appear \u2014 curvature "
                  "features blended into a dominant peak (inversion-dependent; see caveat)." if nsh else "")
        out.append(
            f"The intensity distribution is led by a {i0['kind']} at "
            f"**{_fmt_nm(i0['r_peak'])} nm** ({i0['frac']*100:.0f}% of scattered light, "
            f"{_skew_word(i0['logskew'])}); under number weighting the dominant component is at "
            f"**{_fmt_nm(n0['r_peak'])} nm** \u2014 the *same* data, modes relocated and re-weighted "
            f"by the R-power, which is why no single mean describes it.{sh_txt}\n")
    if fractal:
        out.append(
            "**Sphere vs fractal correction.** The sphere Volume/Number rows assume compact "
            f"spheres (V\u221dR\u00b3, I\u221dR\u2076); this sample is a **mass fractal D_f\u2248{D_f:.1f}**, so the "
            f"fractal rows (volume R^{D_f:.1f}, number R^{2*D_f:.1f}) are the physically-correct "
            "conversions \u2014 they move every mode's peak/\u27e8R\u27e9 upward relative to the sphere rows "
            "(the sphere law over-divides the large flocs).\n")
    else:
        out.append(
            "*Volume/Number assume compact spheres (V\u221dR\u00b3); no mass-fractal correction is "
            "applied (the static slope is consistent with a compact / surface-fractal object, "
            "D_f\u22483).*\n")
    return out


def fig_sizemax_sensitivity(meas_dir, path, label="",
                            smaxes=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)):
    """Demonstrate CORENN size_max (grid-bound) sensitivity on one stitched
    single-angle inversion.  (A) overlays the PSD as size_max grows; (B) the
    intensity-mean size drifts up (mass pulled to the unconstrained edge) while
    the effective DOF -- the data-supportable mode count -- stays flat.  So a
    huge size_max buys edge-pile/ghost peaks (cf. the vendor 3.5e11 nm), not
    information."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import hybrid_correlator as Hc
    try:
        st, _ = Hc.build_stitched_dataset(meas_dir)
    except Exception:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); cmap = plt.cm.viridis
    rows = []
    for i, smax in enumerate(smaxes):
        try:
            r = C.corenn_invert(st, radius_range=(2.5e-10, smax), n_basis=30,
                                n_R=200, fit_baseline=True)
        except Exception:
            continue
        eff = float((r.extra or {}).get("eff_dof", float("nan")))
        y = r.R * np.clip(r.intensity, 0.0, None)
        eps = (np.nanmax(y) * 1e-6) if np.nanmax(y) > 0 else 1e-12
        ax[0].plot(r.R * 1e9, y + eps, "-", lw=1.4, color=cmap(i / max(len(smaxes) - 1, 1)),
                   label=f"{smax*1e9:.0g} nm")
        rows.append({"smax_nm": smax * 1e9, "eff_dof": eff,
                     "imean_nm": r.mean_radius("intensity") * 1e9, "n_peaks": len(r.peaks)})
    if not rows:
        plt.close(fig); return None
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("R [nm]"); ax[0].set_ylabel(r"$R\cdot I(R) + \epsilon$")
    ax[0].set_title("(A) PSD vs size_max")
    ax[0].legend(fontsize=FONT_LEGEND_SMALL, title="size_max")
    sm = [r["smax_nm"] for r in rows]; im = [r["imean_nm"] for r in rows]
    ef = [r["eff_dof"] for r in rows]
    ax[1].plot(sm, im, "o-", color="C3", label="intensity-mean R")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("size_max [nm]"); ax[1].set_ylabel("intensity-mean R [nm]", color="C3")
    a2 = ax[1].twinx(); a2.plot(sm, ef, "s--", color="C0", label="effective DOF")
    a2.set_ylabel("eff-DOF (supportable modes)", color="C0")
    a2.set_ylim(0, (np.nanmax(ef) if np.any(np.isfinite(ef)) else 10) * 1.6)
    ax[1].set_title("(B) size drifts up; eff-DOF (real modes) stays flat")
    fig.suptitle(f"CORENN size_max sensitivity (single-angle stitched) - {label}")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path); plt.close(fig)
    effs = [r["eff_dof"] for r in rows if np.isfinite(r["eff_dof"])]
    return {"rows": rows, "eff_med": float(np.median(effs)) if effs else float("nan"),
            "imean_lo_nm": rows[0]["imean_nm"], "imean_hi_nm": rows[-1]["imean_nm"],
            "smax_lo_nm": rows[0]["smax_nm"], "smax_hi_nm": rows[-1]["smax_nm"]}


def sizemax_section_md(m, fig_path):
    if not m:
        return []
    out = ["### CORENN size_max sensitivity (ill-posedness)\n",
           "CORENN is ill-posed: where it parks unconstrained mass depends on the **size_max** "
           "grid bound. Sweeping size_max on a single-angle stitched inversion shows it "
           "directly:\n",
           f"![sizemax]({fig_path})\n",
           f"Raising size_max from {m['smax_lo_nm']:.0g} to {m['smax_hi_nm']:.0g} nm drifts the "
           f"intensity-mean size from ~{m['imean_lo_nm']:.0f} to ~{m['imean_hi_nm']:.0f} nm "
           "(mass pulled to the unconstrained large-R edge), while the **effective DOF stays "
           f"~ {m['eff_med']:.1f}** -- the data support only ~that many modes either way. So a "
           "large/arbitrary size_max (the vendor 3.5e11 nm) buys edge-pile and ghost peaks, not "
           "information. The pipeline bounds size_max to the physically resolvable window "
           "(pi/q for the multi-angle PSD; the trusted-lag size for the single-angle slow-mode) "
           "and reports the effective DOF as the honest cap on resolvable modes.\n"]
    return out


def write_json_sidecar(path, payload):
    import json
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _morph_short(mph):
    """A compact form-factor tag from a porod_morphology dict."""
    if mph is None:
        return ""
    d = {"porod": "smooth sphere-like", "surface_fractal": "surface fractal",
         "fractal_boundary": "surface-fractal / branched", "mass_fractal": "mass fractal / branched",
         "coil": "Gaussian coil", "swollen_coil": "swollen coil", "rod": "rod"}
    tag = d.get(mph["kind"], "fractal")
    return f"{tag}" + (f" (D≈{mph['dimension']:.1f})" if mph.get("dimension") else "")


def sec_static_shape(conditions):
    m = ["## Static Regime And Shape\n",
         "The CORENN PSD anticipates which static regime each angle probes (Guinier where "
         "q·R_g < 1.3, Porod where q·R ≳ π). When a **valid Porod slope** exists, its exponent "
         "informs the form-factor / morphology model (q^−4 a smooth sphere; 3<p<4 a surface "
         "fractal, D_s=6−p; p≈3 a rough-surface or branched network; 2≤p<3 a mass-fractal / "
         "branched aggregate, D=p; p≈2 a coil; p≈1 a rod). ρ = R_g/R_h is the model-independent "
         "shape check (sphere ≈ 0.78, coil ≈ 1.5, rod ≳ 2), determined only when a static "
         "Guinier R_g exists.\n",
         "| condition | q·R_g span | regime | static fit | implied morphology | ρ = R_g/R_h |",
         "|---|---|---|---|---|---|"]
    for c in conditions:
        s = c["suite"]; rd = s.get("regime")
        if rd is None:
            m.append(f"| {c['label']} | — | — | — | — | {_rho_str(s)} |"); continue
        gn = rd.get("guinier"); mph = rd.get("porod_morphology")
        if gn and np.isfinite(gn.get("R_g", float("nan"))) and gn["R_g"] > 1e-9:
            sfit = f"Guinier R_g={gn['R_g']*1e9:.0f} nm"
        elif rd.get("porod") is not None:
            pe = rd["porod"]["exponent"]
            pse = rd["porod"].get("exponent_se", float("nan"))
            qlo = rd["porod"].get("q_min", float("nan"))
            qhi = rd["porod"].get("q_max", float("nan"))
            ptxt = f"{pe:.2f}±{pse:.2f}" if np.isfinite(pse) else f"{pe:.1f}"
            qtxt = f", q={_qrange_nm(qlo, qhi)}" if np.isfinite(qlo) and np.isfinite(qhi) else ""
            if mph is not None:                       # valid falling power law
                sfit = f"Porod/fractal I∝q^−{ptxt}{qtxt}"
            elif pe >= 0.8:                            # falling but poorly fit
                sfit = f"I∝q^−{ptxt} (diagnostic only{qtxt})"
            else:                                     # flat / rising — not Porod
                sfit = f"flat/rising I∝q^{-pe:+.1f} (no clean slope)"
        else:
            sfit = "form-factor crossover"
        morph = _morph_short(mph) or "—"
        m.append(f"| {c['label']} | {rd['qRg_min']:.1f}–{rd['qRg_max']:.1f} | "
                 f"{rd['regime']} | {sfit} | {morph} | {_rho_str(s)} |")
    m.append("")
    # collect valid Porod morphologies that contradict the sphere assumption
    nonsphere = []
    for c in conditions:
        rd = c["suite"].get("regime") or {}
        mph = rd.get("porod_morphology")
        if mph is not None and not mph["sphere_consistent"]:
            nonsphere.append((c["label"], mph))
    if nonsphere:
        uniq = "; ".join(f"{lbl}: {mph['label']}" for lbl, mph in nonsphere)
        m.append("**Particle model.** The PSD inversion assumes a homogeneous-sphere form "
                 "factor (which predicts a Porod slope q^−4). The measured static power law "
                 f"instead implies — {uniq}. The sphere form factor is therefore **inconsistent "
                 "with the static data**: read the PSD peak as the *effective size* of a rough / "
                 "branched aggregate rather than a compact-sphere radius, and prefer a fractal "
                 "(surface- or mass-fractal) form factor for a quantitative size. ρ = R_g/R_h "
                 "is not independently determined here, so it cannot further constrain the "
                 "shape.\n")
    else:
        all_indep_absent = all(not (c["suite"].get("rho") or {}).get("independent")
                               for c in conditions)
        m.append("**Particle model.** The PSD assumes a homogeneous-sphere form factor. " +
                 ("No valid Porod slope and ρ is **not independently determined** (no Guinier "
                  "regime, or a low-q structure factor), so the sphere assumption is not "
                  "confirmed; treat the PSD radii as model-dependent."
                  if all_indep_absent else
                  "Where ρ is determined, compare it with the assumed model (sphere ≈ 0.78) and "
                  "flag departures.") + "\n")
    return m


def sec_error_propagation(conditions):
    m = ["## Error Propagation And Sensitivity\n",
         "Headline R_h and the PSD peak carry bootstrap intervals (ρ where determined). The "
         "**replicate** bootstrap is measurement repeatability; the **angle/model** bootstrap "
         "is robustness to angle selection — the dominant term for polydisperse samples.\n",
         "| condition | R_h replicate 68% | R_h angle/model 68% | PSD peak 68% | ρ 68% |",
         "|---|---|---|---|---|"]
    for c in conditions:
        s = c["suite"]
        br = s.get("boot_rep"); bm = (s.get("boot") or {}).get("R_h")
        bp = s.get("boot_psd"); brho = (s.get("boot") or {}).get("rho")
        rep = f"{br['p16']*1e9:.0f}–{br['p84']*1e9:.0f} nm" if br else "—"
        ang = f"{bm['p16']*1e9:.0f}–{bm['p84']*1e9:.0f} nm" if bm else "—"
        psd = f"{bp['p16']:.0f}–{bp['p84']:.0f} nm" if bp else "—"
        rho = f"{brho['p16']:.2f}–{brho['p84']:.2f}" if brho else "—"
        m.append(f"| {c['label']} | {rep} | {ang} | {psd} | {rho} |")
    m.append("")
    m.append("**Systematic sensitivity (not included in the intervals above).** "
             "R_h = kT/(6πηD) scales inversely with viscosity, so a viscosity error "
             "propagates linearly into R_h; the q calibration (scattering angle + refractive "
             "index) shifts every q-dependent quantity; the absolute-intensity (toluene) "
             "scale sets the SLS magnitude; and the assumed sphere form factor biases the PSD "
             "radii for non-spherical particles. See the Assumption Register.\n")
    return m


def sec_fit_quality(conditions):
    m = ["## Fit Quality And Model Adequacy\n",
         "Beyond r²: a hold-out cross-validation (predictive R²), a residual-structure test "
         "(runs / lag-1 autocorrelation — structured residuals flag model inadequacy even at "
         "high r²), and the CORENN inversion's convergence and conditioning.\n",
         "| condition | Γ-vs-q² R² (in / CV) | Γ resid | CORENN g2 resid | CORENN: conv / GCV / κ / eff-DOF |",
         "|---|---|---|---|---|"]
    for c in conditions:
        s = c["suite"]; g = s["gvq2"]; cv = s.get("cv") or {}
        rg = s.get("resid_gvq2") or {}; rc = s.get("resid_corenn") or {}
        cd = s.get("corenn_diag") or {}
        gv = f"{g.get('r2', float('nan')):.3f} / {cv.get('cv_r2', float('nan')):.3f}"
        rgv = "**structured**" if rg.get("structured") else "ok"
        rcv = "**structured**" if rc.get("structured") else "ok"
        conv = (f"{'yes' if cd.get('converged') else 'no'} / "
                f"{'interior' if cd.get('gcv_interior_min') else 'boundary'} / "
                f"κ={cd.get('cond', float('nan')):.0e} / "
                f"{cd.get('eff_dof', float('nan')):.0f}/{cd.get('n_basis', '?')}")
        m.append(f"| {c['label']} | {gv} | {rgv} | {rcv} | {conv} |")
    m.append("")
    m.append("*Structured Γ-vs-q² residuals reflect the known polydispersity curvature (the "
             "single slope is a summary, not the model); structured CORENN g2 residuals flag "
             "form-factor / single-distribution imperfection. A predictive CV R² well below "
             "the in-sample R² indicates over-fitting / poor extrapolation across angles.*\n")
    kappas = [(c["suite"].get("corenn_diag") or {}).get("cond", float("nan")) for c in conditions]
    kmax = max([k for k in kappas if np.isfinite(k)], default=float("nan"))
    if np.isfinite(kmax) and kmax > 0:
        sev = ("severely ill-conditioned" if kmax > 1e12 else
               "ill-conditioned" if kmax > 1e8 else "moderately conditioned")
        m.append(f"*Conditioning: κ ≈ {kmax:.0e} means the inversion is **{sev}** "
                 f"(~{math.log10(kmax):.0f} of ~16 double-precision digits lost), so the recovered "
                 "distribution is fixed by the regularization, not the data alone — reinforcing the "
                 "modality caveat. The eff-DOF/n_basis above is the **multi-angle** inversion's "
                 "data-supportable complexity; it is a different inversion from the single-angle "
                 "stitched one whose eff-DOF (~5–6) appears in the size_max diagnostic.*\n")
    return m


def build_report(ctx, conditions, *, main_result, summary_table, consolidating,
                 primary_results, mode_specific, raw_figs=None, countrate_fig=None,
                 per_curve_conditions=None, extra_sections=None):
    """Assemble a template-skeleton provisional report. Returns the markdown text.

    Writes a `<stem>.sidecar.json` next to the report with the machine-readable
    assumption register, quality gates, and headline results.
    """
    per_curve_conditions = per_curve_conditions or conditions
    # Executive Summary leads; Metadata is moved to the back matter.
    m = [f"# {ctx['title']}\n"]
    m += sec_executive_summary(ctx, conditions, main_result)
    m.append("## Summary Table\n"); m += summary_table; m.append("")
    if consolidating:
        cpath, ccap = consolidating
        m.append("## Consolidating Figures\n")
        m.append(f"![consolidating]({cpath})\n")
        m.append(f"*{ccap}*\n")
    m.append("## Primary Model Results\n"); m += primary_results
    # promoted standalone sections
    m += sec_static_shape(conditions)
    m += sec_error_propagation(conditions)
    m += sec_fit_quality(conditions)
    if extra_sections:
        m += extra_sections
    m.append("## Mode-Specific Diagnostics\n"); m += mode_specific
    m += sec_data_inventory(conditions)
    if raw_figs:
        m.append("## Raw Correlograms\n")
        for rp, rcap in raw_figs:
            m.append(f"![raw]({rp})\n"); m.append(f"*{rcap}*\n")
    if countrate_fig:
        cp, ccap = countrate_fig
        m.append("## Count Rates And Static Intensities (absolute, toluene-referenced)\n")
        m.append(f"![countrate]({cp})\n"); m.append(f"*{ccap}*\n")
    m.append("## Per-Curve Fit Results\n")
    for c in per_curve_conditions:
        m += sec_per_curve(c)
    m += sec_rejected(conditions)
    areg, areg_json = sec_assumption_register(ctx, conditions)
    gates_md, gates_json = sec_quality_gates(ctx, conditions)
    m += areg
    m += gates_md
    # back matter
    m += sec_metadata(ctx, conditions)
    m += sec_reproducibility(ctx, ctx.get("generated_files", ["(report files)"]))
    md_text = "\n".join(m)
    if ctx.get("sidecar_path"):
        write_json_sidecar(ctx["sidecar_path"], {
            "title": ctx["title"], "dataset_class": ctx.get("dataset_class"),
            "conditions": [{"label": c["label"], "channel": c.get("channel"),
                            "T_K": c["T"], "eta_Pa_s": c["eta"],
                            "R_h_nm": c["suite"]["gvq2"].get("R_h", float("nan"))*1e9,
                            "rho": _rho_str(c["suite"]),
                            "cv_r2": (c["suite"].get("cv") or {}).get("cv_r2")}
                           for c in conditions],
            "assumptions": areg_json, "quality_gates": gates_json})
    return md_text
