"""
corenn.py -- A custom, transparent implementation of the CORENN algorithm for
recovering particle-size distributions (PSDs) from dynamic light scattering
(DLS) intensity correlation functions, plus a set of extensions that relax
assumptions of the original method.

CORENN ("COrrelation REctification by Non-linear Numerics") was introduced by
A. Vaccaro (LS Instruments) and solves the ill-posed DLS inversion

    g2(q, tau_i) - 1 = beta * [ \\int I(R) exp(-q^2 D(R) tau_i) dR ]^2 ,
    D(R) = k_B T / (6 pi eta R)

for the intensity-weighted PSD I(R).  The defining ingredients of CORENN are:

  1. a smooth *cubic B-spline* discretization of the unknown I(R),
  2. *non-linear* weighted least squares on the un-linearized model
     (it does NOT take the square root of g2-1, which would distort the noise),
  3. *Tikhonov* regularization with a curvature (smoothness) penalty,
  4. data weights from a *precomputed* correlation-noise model, and
  5. *generalized cross validation* (GCV) to choose the regularization weight.

This module reproduces all five ingredients and adds:

  * EMPIRICAL per-lag uncertainties estimated from measurement replicates
    (more faithful than any parametric noise model when replicates exist);
  * hard NON-NEGATIVITY enforced inside the optimizer (cubic B-splines are
    non-negative, so x >= 0 guarantees I(R) >= 0 everywhere);
  * a P-spline penalty on log-spaced knots, which regularizes curvature in
    log(R) -- the natural scale for size distributions spanning decades;
  * UNCERTAINTY BANDS on the recovered PSD via replicate bootstrap or the
    linearized (Gauss-Newton) posterior covariance;
  * a MULTI-ANGLE joint inversion that solves for a single q-independent
    *number* distribution N(R) using the sphere form factor P(qR), relaxing
    the single-angle assumption that the intensity weighting is q-independent.

The module is self-contained (numpy + scipy) and can be driven either from raw
arrays or directly from LS Instruments "Correlation Function.csv" /
"Summary.csv" exports.

Author: Michael L. Whittaker (and Claude), 2026.
"""

from __future__ import annotations

import os
import re
import glob
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.interpolate import BSpline
    from scipy.optimize import least_squares, nnls
except ImportError as exc:  # pragma: no cover
    raise ImportError("corenn.py requires scipy (interpolate, optimize).") from exc


k_BOLTZMANN = 1.380649e-23  # J / K
DEFAULT_WAVELENGTH_NM = 632.8  # HeNe laser
DEFAULT_CORENN_RADIUS_RANGE = (1e-10, 5e-6)  # 1 Angstrom to 5 microns


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class DLSDataset:
    """One DLS correlation curve at a single scattering vector q.

    Parameters
    ----------
    tau : array
        Lag times in seconds (strictly positive, sorted ascending).
    g2m : array
        The measured ``g2(tau) - 1`` (the "Value" column of LS exports).
    q : float
        Scattering vector magnitude in 1/m.
    beta : float
        Coherence factor / intercept (``g2(0) - 1``).  Used as the known
        amplitude unless ``fit_beta`` is requested at inversion time.
    sigma : array, optional
        Per-lag 1-sigma uncertainty on ``g2m``.  When ``None`` a parametric
        model is used at inversion time.
    temperature : float, optional
        Absolute temperature [K].
    viscosity : float, optional
        Solvent viscosity [Pa s].
    n_solvent : float
        Solvent refractive index (for q from angle and for the form factor).
    metadata : dict, optional
        Free-form provenance (measurement id, repetition, angle, ...).
    """

    tau: np.ndarray
    g2m: np.ndarray
    q: float
    beta: float = 1.0
    sigma: Optional[np.ndarray] = None
    temperature: Optional[float] = None
    viscosity: Optional[float] = None
    n_solvent: float = 1.33
    duration: Optional[float] = None       # total acquisition time [s]
    count_rate: Optional[float] = None     # mean detected count rate [Hz]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.tau = np.asarray(self.tau, dtype=float).ravel()
        self.g2m = np.asarray(self.g2m, dtype=float).ravel()
        if self.tau.shape != self.g2m.shape:
            raise ValueError("tau and g2m must have the same length.")
        order = np.argsort(self.tau)
        self.tau = self.tau[order]
        self.g2m = self.g2m[order]
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float).ravel()[order]


@dataclass
class CorennResult:
    """Container for an inversion result."""

    R: np.ndarray                 # radius grid [m]
    intensity: np.ndarray         # intensity-weighted PSD I(R) (area-normalized)
    volume: np.ndarray            # volume-weighted PSD (area-normalized)
    number: np.ndarray            # number-weighted PSD (area-normalized)
    x: np.ndarray                 # B-spline coefficients
    beta: float
    lam: float                    # selected regularization weight
    lam_grid: np.ndarray
    gcv: np.ndarray               # GCV(lambda) over the grid
    tau: np.ndarray
    g2m: np.ndarray
    g2_model: np.ndarray          # model g2 - 1 at the data lags
    residual_norm: float
    peaks: List[Dict[str, float]]  # detected modes (intensity-weighted)
    intensity_band: Optional[Tuple[np.ndarray, np.ndarray]] = None
    extra: Dict[str, object] = field(default_factory=dict)

    # convenience -----------------------------------------------------------
    def mean_radius(self, weighting: str = "intensity") -> float:
        w = getattr(self, weighting)
        return float(np.trapz(self.R * w, self.R) / np.trapz(w, self.R))

    def summary(self) -> str:
        lines = [
            f"CORENN inversion: lambda={self.lam:.3e}, "
            f"residual_norm={self.residual_norm:.4f}, beta={self.beta:.4f}",
            f"  intensity-weighted mean R = {self.mean_radius('intensity')*1e9:.2f} nm",
        ]
        for i, pk in enumerate(self.peaks):
            lines.append(
                f"  peak {i+1}: R={pk['radius_nm']:.2f} nm  "
                f"amount={pk['rel_amount']*100:.1f}%  CV={pk['cv']:.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def q_from_angle(theta_deg: float, n_solvent: float,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> float:
    """Scattering vector magnitude [1/m] from angle, refractive index, lambda.

    q = 4 pi n sin(theta/2) / lambda0.  Note the refractive index factor that
    the legacy pct.py kernel omits.
    """
    lambda_m = wavelength_nm * 1e-9
    return 4.0 * math.pi * n_solvent * math.sin(math.radians(theta_deg) / 2.0) / lambda_m


def diffusion_coefficient(R: np.ndarray, temperature: float,
                          viscosity: float) -> np.ndarray:
    """Stokes-Einstein translational diffusion coefficient [m^2/s]."""
    return k_BOLTZMANN * temperature / (6.0 * math.pi * viscosity * np.asarray(R, float))


def sphere_form_factor(q: float, R: np.ndarray) -> np.ndarray:
    """Rayleigh-Gans-Debye form factor P(qR) for homogeneous spheres.

    P(x) = [3 (sin x - x cos x) / x^3]^2, with the x->0 limit equal to 1.
    """
    x = q * np.asarray(R, float)
    out = np.ones_like(x)
    nz = x > 1e-8
    xv = x[nz]
    out[nz] = (3.0 * (np.sin(xv) - xv * np.cos(xv)) / xv**3) ** 2
    return out


# ---------------------------------------------------------------------------
# Cumulant analysis and multi-angle diffusion
# ---------------------------------------------------------------------------
def cumulant_fit(tau: np.ndarray, g2m: np.ndarray, beta: Optional[float] = None,
                 order: int = 2, g1_min: float = 0.60,
                 tau_max: Optional[float] = None) -> Dict[str, float]:
    """Method-of-cumulants fit of one DLS curve.

    Fits ``ln g1(tau) = -Gamma tau + (mu2/2) tau^2 - ...`` over the *initial*
    decay (amplitude-weighted), where ``g1 = sqrt((g2-1)/beta)``.  The fit
    window is the contiguous leading segment down to ``g1_min`` (default 0.60,
    roughly the first half-decay) -- this is essential for polydisperse /
    aggregate-laden samples where a slow tail would otherwise bias the rate far
    too low.
    Returns the mean decay rate ``Gamma`` [1/s], the second cumulant ``mu2``,
    and the polydispersity index ``PDI = mu2/Gamma^2``.
    """
    tau = np.asarray(tau, float)
    g2m = np.asarray(g2m, float)
    if beta is None:
        beta = float(np.nanmax(g2m))
    ratio = np.clip(g2m / max(beta, 1e-12), 0.0, None)
    g1 = np.sqrt(ratio)
    # leading contiguous segment from the shortest lag down to g1_min
    valid = np.isfinite(g1)
    end = g1.size
    for i in range(g1.size):
        if not valid[i]:
            continue
        if g1[i] < g1_min:
            end = i
            break
    sel = np.zeros(g1.size, bool)
    sel[:end] = valid[:end]
    if tau_max is not None:
        sel &= tau <= tau_max
    if np.count_nonzero(sel) < order + 1:
        return {"Gamma": float("nan"), "mu2": float("nan"),
                "PDI": float("nan"), "n_points": int(np.count_nonzero(sel))}
    t = tau[sel]
    g1s = g1[sel]
    coeffs = np.polyfit(t, np.log(g1s), order, w=g1s)  # highest power first
    a = coeffs[::-1]                                    # a0 + a1 t + a2 t^2 + ...
    Gamma = float(-a[1])
    mu2 = float(2.0 * a[2]) if order >= 2 else 0.0
    PDI = mu2 / Gamma**2 if Gamma > 0 else float("nan")
    return {"Gamma": Gamma, "mu2": mu2, "PDI": PDI,
            "intercept": float(np.exp(2.0 * a[0]) * beta),
            "n_points": int(np.count_nonzero(sel))}


def guinier_fit(q: Sequence[float], intensity: Sequence[float],
                intensity_err: Optional[Sequence[float]] = None,
                qrg_max: float = 1.3, n_min: int = 4) -> Dict[str, float]:
    """Guinier fit of angle-resolved static intensity -> radius of gyration.

    In the Guinier regime ``ln I(q) = ln I0 - q^2 R_g^2 / 3`` for ``q R_g`` up to
    ~``qrg_max``.  The fit window is selected self-consistently: start from the
    lowest-q points, fit, recompute ``R_g``, and keep only points with
    ``q R_g <= qrg_max`` (at least ``n_min`` points).  Returns ``R_g``, ``I0``,
    the fit window size and ``r^2``.
    """
    q = np.asarray(q, float)
    I = np.asarray(intensity, float)
    order = np.argsort(q)
    q, I = q[order], I[order]
    good = np.isfinite(q) & np.isfinite(I) & (I > 0)
    q, I = q[good], I[good]
    if intensity_err is not None:
        ierr = np.asarray(intensity_err, float)[order][good]
    else:
        ierr = None
    if q.size < n_min:
        return {"R_g": float("nan"), "I0": float("nan"), "n_points": int(q.size),
                "r2": float("nan")}
    y = np.log(I)
    w = (I / ierr) if ierr is not None else None     # ln-space weight ~ I/dI
    Rg = None
    for _ in range(6):
        if Rg is None:
            sel = np.arange(min(n_min + 2, q.size))
        else:
            sel = np.where(q * Rg <= qrg_max)[0]
            if sel.size < n_min:
                sel = np.arange(n_min)
        x_sel = q[sel] ** 2
        x_scale = float(np.mean(np.abs(x_sel))) or 1.0
        A = np.vstack([x_sel / x_scale, np.ones(sel.size)]).T
        ww = w[sel] if w is not None else None
        if ww is not None:
            sol, *_ = np.linalg.lstsq(A * ww[:, None], y[sel] * ww, rcond=None)
        else:
            sol, *_ = np.linalg.lstsq(A, y[sel], rcond=None)
        slope, lnI0 = float(sol[0] / x_scale), float(sol[1])
        Rg_new = math.sqrt(max(-3.0 * slope, 0.0))
        if Rg is not None and abs(Rg_new - Rg) < 1e-3 * max(Rg, 1e-12):
            Rg = Rg_new
            break
        Rg = Rg_new
    pred = (q[sel] ** 2 / x_scale) * sol[0] + sol[1]
    ss_res = float(np.sum((y[sel] - pred) ** 2))
    ss_tot = float(np.sum((y[sel] - y[sel].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"R_g": float(Rg), "I0": float(math.exp(lnI0)),
            "n_points": int(sel.size), "qRg_max": float(q[sel].max() * Rg),
            "r2": r2}


def segmented_guinier_fit(
    q: Sequence[float],
    intensity: Sequence[float],
    intensity_err: Optional[Sequence[float]] = None,
    n_min: int = 5,
    bic_threshold: float = 10.0,
) -> Dict[str, object]:
    """Compare one-line Guinier and optional two-segment diagnostics.

    The default physical model is a single Guinier slope.  A two-line fit is
    reported as a *two-population Guinier* only when the BIC improvement is
    strong and both segments are physically meaningful Guinier slopes
    (negative slopes with finite radii, with the low-q radius larger than the
    high-q radius).  Otherwise the two-line fit is retained only as a curvature
    diagnostic: it may reflect structure factor, background, normalization, or
    other static-intensity artifacts rather than two particle populations.
    """
    q = np.asarray(q, float)
    I = np.asarray(intensity, float)
    order = np.argsort(q)
    q, I = q[order], I[order]
    good = np.isfinite(q) & np.isfinite(I) & (I > 0)
    q, I = q[good], I[good]
    if intensity_err is not None:
        ierr = np.asarray(intensity_err, float)[order][good]
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(ierr > 0, I / ierr, np.nan)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    else:
        weights = np.ones_like(I)

    x = q * q
    y = np.log(I)
    n = x.size

    def _fit_slice(start, stop):
        xs = x[start:stop]
        ys = y[start:stop]
        ws = weights[start:stop]
        x_scale = float(np.mean(np.abs(xs))) or 1.0
        A = np.vstack([xs / x_scale, np.ones(xs.size)]).T
        sol, *_ = np.linalg.lstsq(A * ws[:, None], ys * ws, rcond=None)
        slope, intercept = float(sol[0] / x_scale), float(sol[1])
        pred = A @ sol
        resid = ys - pred
        rss = float(np.sum((ws * resid) ** 2))
        ss_tot = float(np.sum((ws * (ys - np.average(ys, weights=ws))) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
        Rg = math.sqrt(-3.0 * slope) if slope < 0 else float("nan")
        return {
            "slope": slope,
            "intercept": intercept,
            "I0": float(math.exp(intercept)),
            "R_g": float(Rg),
            "n_points": int(xs.size),
            "q_min": float(q[start]),
            "q_max": float(q[stop - 1]),
            "q2_min": float(xs[0]),
            "q2_max": float(xs[-1]),
            "r2": r2,
            "rss": rss,
        }

    if n < max(2 * n_min, 2):
        return {
            "one_segment": None,
            "two_segment": None,
            "delta_bic": float("nan"),
            "bic_supports_two_segments": False,
            "supports_two_segments": False,
            "n_points": int(n),
        }

    one = _fit_slice(0, n)
    rss1 = max(one["rss"], 1e-300)
    bic1 = n * math.log(rss1 / n) + 2.0 * math.log(n)

    best = None
    for split in range(n_min, n - n_min + 1):
        low = _fit_slice(0, split)
        high = _fit_slice(split, n)
        rss = max(low["rss"] + high["rss"], 1e-300)
        bic = n * math.log(rss / n) + 5.0 * math.log(n)
        if best is None or bic < best["bic"]:
            best = {
                "split_index": int(split),
                "break_q_low": float(q[split - 1]),
                "break_q_high": float(q[split]),
                "break_q": float(0.5 * (q[split - 1] + q[split])),
                "break_q2": float(0.5 * (x[split - 1] + x[split])),
                "segments": [
                    {"name": "low_q_large", **low},
                    {"name": "high_q_small", **high},
                ],
                "rss": rss,
                "bic": bic,
            }

    delta_bic = bic1 - best["bic"] if best is not None else float("nan")
    bic_supports_two = bool(np.isfinite(delta_bic) and delta_bic > bic_threshold)
    physical_two = False
    if best is not None and bic_supports_two:
        low, high = best["segments"]
        rg_low, rg_high = low["R_g"], high["R_g"]
        physical_two = (
            low["slope"] < 0.0
            and high["slope"] < 0.0
            and np.isfinite(rg_low)
            and np.isfinite(rg_high)
            and rg_low > rg_high
        )
    return {
        "one_segment": {**one, "bic": bic1},
        "two_segment": best,
        "delta_bic": float(delta_bic),
        "bic_supports_two_segments": bic_supports_two,
        "supports_two_segments": bool(physical_two),
        "bic_threshold": float(bic_threshold),
        "n_points": int(n),
    }


def apparent_diffusion_analysis(
    q: Sequence[float], gamma: Sequence[float],
    temperature: float, viscosity: float,
    mu2: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Quantify polydispersity and structure from the q-dependent first cumulant.

    The apparent diffusion coefficient ``D_app(q) = Gamma(q)/q^2`` and apparent
    radius ``R_h,app(q) = kT/(6 pi eta D_app)`` vary with q for non-ideal
    samples.  A linear fit ``D_app(q) = D0 (1 + C q^2)`` gives the q->0
    z-average ``D0`` (hence ``R_h,z``) and a dimensionless coefficient ``C``
    whose sign/magnitude reflects polydispersity and internal/structural modes
    (``C>0``: faster apparent diffusion at high q, the usual polydispersity
    signature; non-monotonic ``R_h,app`` flags a structure-factor peak).  When
    second cumulants ``mu2`` are supplied the mean polydispersity index
    ``PDI = <mu2/Gamma^2>`` is also reported.
    """
    q = np.asarray(q, float)
    g = np.asarray(gamma, float)
    good = np.isfinite(q) & np.isfinite(g) & (g > 0)
    q, g = q[good], g[good]
    q2 = q * q
    D_app = g / q2
    R_app = k_BOLTZMANN * temperature / (6.0 * math.pi * viscosity * D_app)
    # D_app = D0 (1 + C q^2); fit in scaled q^2 for conditioning
    scale = float(np.mean(q2)) or 1.0
    A = np.vstack([np.ones(q2.size), q2 / scale]).T
    sol, *_ = np.linalg.lstsq(A, D_app, rcond=None)
    D0, slope_s = float(sol[0]), float(sol[1])
    C = (slope_s / scale) / D0 if D0 != 0 else float("nan")
    R_h_z = k_BOLTZMANN * temperature / (6.0 * math.pi * viscosity * D0) if D0 > 0 else float("nan")
    # robust polydispersity span: low-q vs high-q decile of R_h,app (not raw
    # extremes, which are sensitive to a single noisy angle).
    order = np.argsort(q)
    R_sorted = R_app[order]
    k = max(1, R_sorted.size // 5)
    lo = float(np.nanmedian(R_sorted[-k:]))     # high q (small apparent R)
    hi = float(np.nanmedian(R_sorted[:k]))      # low q  (large apparent R)
    span = hi / lo if lo > 0 else float("nan")
    D_sorted = D_app[order]
    D_high_q = float(np.nanmedian(D_sorted[-k:]))
    D_low_q = float(np.nanmedian(D_sorted[:k]))
    # structure: count sign changes of a 3-point median-smoothed dR/dq, and the
    # rank correlation of R_h,app with q (strongly negative => polydispersity).
    Rs = R_sorted.copy()
    if Rs.size >= 3:
        Rs[1:-1] = np.median(np.vstack([R_sorted[:-2], R_sorted[1:-1], R_sorted[2:]]), axis=0)
    dR = np.diff(Rs)
    sign_changes = int(np.sum(np.diff(np.sign(dR[dR != 0])) != 0)) if np.any(dR != 0) else 0
    qr = np.argsort(np.argsort(q[order])); rr = np.argsort(np.argsort(R_sorted))
    spearman = float(np.corrcoef(qr, rr)[0, 1]) if q.size > 2 else float("nan")
    out = {
        "q": q, "D_app": D_app, "R_h_app": R_app,
        "D0": D0, "C": C, "R_h_z": R_h_z,
        "polydispersity_span": span,
        "R_h_low_q_median": hi,
        "R_h_high_q_median": lo,
        "D_app_low_q_median": D_low_q,
        "D_app_high_q_median": D_high_q,
        "structure_sign_changes": sign_changes,
        "spearman_q_Rh": spearman,
    }
    if mu2 is not None:
        mu2 = np.asarray(mu2, float)[good]
        pdi = mu2 / (g * g)
        out["PDI_mean"] = float(np.nanmean(pdi[np.isfinite(pdi)]))
    return out


def _shape_from_rho(rho: float) -> str:
    """Morphology label from the structure-sensitive ratio rho = R_g / R_h."""
    if not np.isfinite(rho):
        return "undetermined"
    if rho < 0.70:
        return "compact (denser than uniform sphere)"
    if rho < 0.85:
        return "~uniform sphere (0.775)"
    if rho < 1.15:
        return "hollow shell / vesicle / core-shell (~1.0)"
    if rho < 1.40:
        return "extended / branched"
    if rho < 1.90:
        return "random coil (1.5-1.8)"
    return "rod / stiff / very extended (>1.9)"


def static_dynamic_correlation(
    q: Sequence[float],
    intensity: Sequence[float],
    gamma: Sequence[float],
    temperature: float,
    viscosity: float,
    intensity_err: Optional[Sequence[float]] = None,
    mu2: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Correlate static Guinier diagnostics and dynamic R_h,app(q) sizes.

    Both observables are moments of the *same* q-dependent intensity weight
    ``w(R,q) = n(R) R^6 P(qR)``:

        I(q)        = \\int w(R,q) dR                      (static, 0th moment)
        D_app(q)    = \\int w D dR / \\int w dR  = <D>_w     (dynamic, weighted mean)
        R_h,app(q)  = kT/(6 pi eta D_app) = 1 / <1/R>_w   (harmonic-mean radius)

    Consequences used here:

      * The default static summary is a single Guinier slope.  A two-segment
        static interpretation is used only when both fitted segments have
        physically meaningful negative Guinier slopes and a strong BIC
        improvement.
      * When that stricter two-population condition is met, the low-q and
        high-q static ``R_g`` values can be paired with the low-q and high-q
        dynamic apparent ``R_h`` plateaus.  Their ratio ``rho = R_g/R_h`` is a
        morphology diagnostic (0.775 sphere, ~1 shell, 1.5-1.8 coil, >1.9 rod).
      * The static Guinier break and the dynamic ``R_h,app`` crossover are
        compared only as a consistency check for that accepted two-population
        case; otherwise the static curvature is treated as a diagnostic, not a
        population model.
      * Whether ``R_h,app(q)`` linearizes: a power law ``R_h,app ~ q^{-alpha}``
        (``ln R_h`` vs ``ln q`` linear) is the broad/scale-free-distribution
        signature, with ``alpha -> 0`` monodisperse and ``alpha -> 1`` fully
        scale-free (each q probes size ~1/q).  An exponential ``R_h,app ~
        e^{-k q}`` (``ln R_h`` vs ``q`` linear) is the alternative the data can
        adjudicate by r^2.
    """
    q = np.asarray(q, float)
    sg = segmented_guinier_fit(q, intensity, intensity_err=intensity_err)
    an = apparent_diffusion_analysis(q, gamma, temperature, viscosity, mu2=mu2)

    # --- per-population static R_g and dynamic R_h ------------------------
    if sg["supports_two_segments"]:
        seg_lo, seg_hi = sg["two_segment"]["segments"]
        Rg_large, Rg_small = seg_lo["R_g"], seg_hi["R_g"]
        break_q = sg["two_segment"]["break_q"]
    else:
        Rg_large = Rg_small = (sg["one_segment"] or {}).get("R_g", float("nan"))
        break_q = float("nan")
    Rh_large = an["R_h_low_q_median"]
    Rh_small = an["R_h_high_q_median"]
    rho_large = Rg_large / Rh_large if Rh_large else float("nan")
    rho_small = Rg_small / Rh_small if Rh_small else float("nan")

    # --- dynamic crossover q: where R_h,app crosses the geometric-mean plateau
    qa = an["q"]; Rapp = an["R_h_app"]
    order = np.argsort(qa)
    qa, Rapp = qa[order], Rapp[order]
    target = math.sqrt(max(Rh_large * Rh_small, 0.0))
    cross_q = float("nan")
    sgn = np.sign(Rapp - target)
    idx = np.where(np.diff(sgn) != 0)[0]
    if idx.size:
        i = idx[0]
        # log-linear interpolation in q at the crossing
        x0, x1 = math.log(qa[i]), math.log(qa[i + 1])
        y0, y1 = Rapp[i] - target, Rapp[i + 1] - target
        cross_q = math.exp(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else qa[i]

    # --- linearity tests on R_h,app(q) -----------------------------------
    valid = np.isfinite(qa) & np.isfinite(Rapp) & (Rapp > 0)
    ql, Rl = qa[valid], Rapp[valid]
    def _lin(x, y):
        A = np.vstack([x, np.ones(x.size)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ sol
        ss = np.sum((y - pred) ** 2); st = np.sum((y - y.mean()) ** 2)
        return float(sol[0]), float(sol[1]), float(1 - ss / st if st > 0 else np.nan)
    a_ll, b_ll, r2_loglog = _lin(np.log(ql), np.log(Rl))     # ln Rh vs ln q
    a_sl, b_sl, r2_semilog = _lin(ql, np.log(Rl))            # ln Rh vs q
    alpha = -a_ll                                            # R_h,app ~ q^-alpha

    return {
        "segmented_guinier": sg, "apparent_diffusion": an,
        "R_g_large": Rg_large, "R_g_small": Rg_small,
        "R_h_large": Rh_large, "R_h_small": Rh_small,
        "rho_large": rho_large, "rho_small": rho_small,
        "shape_large": _shape_from_rho(rho_large),
        "shape_small": _shape_from_rho(rho_small),
        "static_break_q": break_q, "dynamic_crossover_q": cross_q,
        "crossover_consistency": (cross_q / break_q
                                  if np.isfinite(break_q) and break_q else float("nan")),
        "powerlaw_alpha": alpha, "r2_loglog": r2_loglog,
        "exp_rate": a_sl, "r2_semilog": r2_semilog,
        "linear_form": ("power-law q^-alpha" if r2_loglog >= r2_semilog
                        else "exponential e^-kq"),
    }


GUINIER_SPHERE = math.sqrt(3.0 / 5.0)   # R_g = sqrt(3/5) R for a uniform sphere


def joint_temperature_fit(
    series: Sequence[Dict[str, object]],
    reference_index: int = 0,
) -> Dict[str, object]:
    """Joint multi-temperature analysis assuming one (T-invariant) sample.

    For a single sample the hydrodynamic size is temperature-independent and the
    only T-dependence of the dynamics is through ``D_z(T) = k T / (6 pi eta(T)
    R_h,z)``.  Each entry of ``series`` is a dict with ``'T'`` [K], ``'q'`` and
    ``'gamma'`` arrays (per-angle first cumulants), and optionally ``'eta'``
    [Pa s] and ``'label'``.  The routine:

      1. fits ``D_z(T)`` from each temperature's Gamma-vs-q^2 slope;
      2. forms ``b(T) = k T / (6 pi D_z(T)) = eta(T) R_h,z``;
      3. if every entry has ``eta``: reports ``R_h(T) = b/eta`` and the shared
         ``R_h,z`` (inverse-variance mean) with its spread -- a *same-sample
         test* (small spread => one sample, just rescaled by T/eta);
      4. solves the viscosity implied by a single shared size,
         ``eta_fit(T) = b(T)/R_h,z``, anchored to the reference temperature's
         ``eta`` -- useful when the recorded viscosity is wrong or missing
         (e.g. a fixed value entered across temperatures).

    Returns D_z(T), b(T), per-T R_h, the shared R_h,z, eta_fit(T) and the
    consistency (coefficient of variation of R_h across T).
    """
    T = np.array([float(s["T"]) for s in series])
    Dz = np.empty(T.size)
    for i, s in enumerate(series):
        fit = gamma_vs_q2(s["q"], s["gamma"])
        Dz[i] = fit["D"]
    b = k_BOLTZMANN * T / (6.0 * math.pi * Dz)             # = eta * R_h,z

    eta_known = np.array([s.get("eta", np.nan) for s in series], dtype=float)
    have_eta = np.all(np.isfinite(eta_known))

    if have_eta:
        R_h = b / eta_known
        R_h_z = float(np.mean(R_h))
        cv = float(np.std(R_h) / R_h_z) if R_h_z > 0 else float("nan")
    else:
        # anchor the shared size to the reference temperature's eta
        eta_ref = float(series[reference_index].get("eta") or np.nan)
        if not np.isfinite(eta_ref):
            raise ValueError("reference entry must provide 'eta'.")
        R_h_z = b[reference_index] / eta_ref
        R_h = np.full(T.size, R_h_z)
        cv = 0.0

    eta_fit = b / R_h_z                                     # viscosity if size shared
    labels = [s.get("label", f"T{i}") for i, s in enumerate(series)]
    return {
        "labels": labels, "T": T, "D_z": Dz, "b": b,
        "R_h": R_h, "R_h_z": R_h_z, "consistency_cv": cv,
        "eta_fit": eta_fit, "eta_known": eta_known,
        "same_sample": bool(have_eta and cv < 0.15),
    }


def gamma_vs_q2(q: Sequence[float], gamma: Sequence[float],
                sigma: Optional[Sequence[float]] = None,
                temperature: Optional[float] = None,
                viscosity: Optional[float] = None) -> Dict[str, float]:
    """Weighted linear fit of decay rate vs q^2 to extract diffusion + size.

    For purely translational diffusion ``Gamma(q) = D q^2`` (zero intercept), so
    the slope is the (z-average) diffusion coefficient and, given T and eta,
    the Stokes-Einstein hydrodynamic radius ``R_h = kT/(6 pi eta D)``.  A
    non-zero intercept or curvature flags non-diffusive or q-dependent dynamics.
    """
    q = np.asarray(q, float)
    g = np.asarray(gamma, float)
    q2 = q * q
    good = np.isfinite(q2) & np.isfinite(g)
    # Scale q^2 to O(1) before the linear fit: q^2 ~ 1e15 alongside a column of
    # ones gives a condition number ~1e29 that swamps the intercept (i.e. the
    # rotational 6 D_r term in depolarized DLS).  Fit in scaled units, unscale.
    scale = float(np.mean(q2[good])) or 1.0
    A = np.vstack([q2[good] / scale, np.ones(np.count_nonzero(good))]).T
    y = g[good]
    if sigma is not None:
        w = 1.0 / np.clip(np.asarray(sigma, float)[good], 1e-30, None)
        sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    else:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    D, intercept = float(sol[0]) / scale, float(sol[1])
    pred = A @ sol
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"D": D, "intercept": intercept, "r2": r2}
    if temperature is not None and viscosity is not None and D > 0:
        out["R_h"] = k_BOLTZMANN * temperature / (6.0 * math.pi * viscosity * D)
    return out


# ---------------------------------------------------------------------------
# Discretization: radius grid, B-spline basis, kernels
# ---------------------------------------------------------------------------
def build_radius_grid(r_min: float, r_max: float, n_R: int) -> np.ndarray:
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("Require 0 < r_min < r_max.")
    return np.geomspace(r_min, r_max, n_R)


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Quadrature weights so that ``w @ f`` approximates ``\\int f dx``."""
    x = np.asarray(x, float)
    w = np.empty_like(x)
    dx = np.diff(x)
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w


def build_bspline_basis(r_grid: np.ndarray, n_basis: int, degree: int = 3
                        ) -> np.ndarray:
    """Cubic B-spline basis evaluated on ``r_grid``.

    Knots are uniform in ``u = log(R)`` (a P-spline construction), which gives
    even resolution across decades of size and makes the second-difference
    coefficient penalty approximate curvature in log(R).

    Returns
    -------
    Phi : (n_R, n_basis) array
        Basis functions (non-negative), columns sum-to-one partition of unity.
    """
    if n_basis < degree + 1:
        raise ValueError("n_basis must be >= degree + 1.")
    u = np.log(r_grid)
    u0, u1 = u[0], u[-1]
    # interior knots uniform in u; clamp ends with multiplicity degree+1
    n_interior = n_basis - degree + 1
    interior = np.linspace(u0, u1, n_interior)
    knots = np.concatenate([np.full(degree, u0), interior, np.full(degree, u1)])
    Phi = np.empty((r_grid.size, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=False)
        Phi[:, j] = np.nan_to_num(spline(u), nan=0.0)
    return Phi


def second_difference_operator(n: int) -> np.ndarray:
    """Second-order difference matrix D2 ((n-2) x n) for the P-spline penalty."""
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def build_kernel(q: float, tau: np.ndarray, R: np.ndarray, w_R: np.ndarray,
                 Phi: np.ndarray, temperature: float, viscosity: float,
                 weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Forward kernel K so that ``g1(tau_i) = (K x)_i``.

    K_ij = \\int weight(R) Phi_j(R) exp(-q^2 D(R) tau_i) dR
         ~ sum_k w_k weight_k Phi_kj exp(-q^2 D_k tau_i).

    ``weight`` lets a multi-angle inversion fold in R^6 P(qR) so that x are the
    coefficients of a *number* distribution; for the standard single-angle
    inversion ``weight`` is None and x are the *intensity* PSD coefficients.
    """
    D = diffusion_coefficient(R, temperature, viscosity)
    Gamma = q * q * D                          # decay rate per radius [1/s]
    E = np.exp(-np.outer(tau, Gamma))          # (n_tau, n_R)
    # Integrate over a normalized measure (weights sum to 1) so the kernel and
    # coefficients stay O(1).  The absolute dR scale cancels because the PSD is
    # area-normalized; this avoids the ~1e8 conditioning blowup of SI metres.
    w_norm = w_R / np.sum(w_R)
    WPhi = (w_norm[:, None] * Phi)             # (n_R, n_basis)
    if weight is not None:
        WPhi = weight[:, None] * WPhi
    return E @ WPhi


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------
def empirical_sigma(g2m_replicates: Sequence[np.ndarray]) -> np.ndarray:
    """Per-lag 1-sigma uncertainty from replicate correlation curves.

    This is the gold standard when several repetitions are available: it makes
    no assumption about the correlator architecture.  Returns the standard
    deviation across replicates (a floor is applied to avoid zero weights).
    """
    M = np.vstack([np.asarray(g, float) for g in g2m_replicates])
    sd = np.std(M, axis=0, ddof=1)
    base = np.nanmax(np.abs(M)) if M.size else 1.0
    floor = max(base * 1e-4, 1e-9)
    return np.where(np.isfinite(sd) & (sd > floor), sd, floor)


def schatzel_sigma(tau: np.ndarray, g1: np.ndarray, beta: float,
                   duration_s: float, sigma_floor: float = 0.0,
                   gaussian_gain: float = 1.0) -> np.ndarray:
    """Parametric Schaetzel-style noise model for ``g2 - 1``.

    Gaussian-field approximation of the variance of the normalized intensity
    correlation estimator (Schaetzel, Quantum Opt. 2, 287 (1990); generalized
    by Biganzoli & Ferri, Opt. Express 26, 29375 (2018)).  The key physics is
    that the number of statistically independent decorrelation windows in a run
    of duration ``T`` falls as ``M(tau) ~ T / tau``, so the variance of the
    estimator *grows* at long lag:

        Var(tau) = sigma_floor^2 + gaussian_gain * (1 + beta g1(tau)^2)^2 / M(tau)

    The ``sigma_floor`` sets the short-lag (shot/quantization) noise level and
    is best taken from replicate scatter; the second term supplies the long-lag
    growth that down-weights the noise floor.  This form is numerically robust
    (no exploding shot term) and reproduces the measured "flat-then-rising"
    sigma(tau).
    """
    tau = np.asarray(tau, float)
    g1 = np.clip(np.asarray(g1, float), 0.0, 1.0)
    M = np.maximum(duration_s / np.maximum(tau, 1e-12), 1.0)
    var = sigma_floor**2 + gaussian_gain * (1.0 + beta * g1**2) ** 2 / M
    return np.sqrt(np.maximum(var, 1e-18))


def inflate_sigma_in_lag_band(
    tau: Sequence[float],
    sigma: Sequence[float],
    lag_band: Tuple[float, float],
    systematic_sigma: Optional[Union[float, Sequence[float]]] = None,
    residual: Optional[Sequence[float]] = None,
    factor: float = 1.0,
    mode: str = "rms",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Add systematic short-lag uncertainty in quadrature.

    ``systematic_sigma`` can be a scalar floor or a per-lag array.  Alternatively,
    pass a residual array and estimate a systematic scale inside ``lag_band``.
    The default residual mode is a log-lag RMS, which treats reproducible
    residual wiggles as model discrepancy rather than PSD evidence.
    """
    tau = np.asarray(tau, float)
    sigma = np.asarray(sigma, float)
    if tau.shape != sigma.shape:
        raise ValueError("tau and sigma must have the same shape.")
    lag_lo, lag_hi = map(float, lag_band)
    if lag_lo <= 0 or lag_hi <= lag_lo:
        raise ValueError("lag_band must be a positive (low, high) interval.")
    keep = (
        (tau >= lag_lo)
        & (tau <= lag_hi)
        & np.isfinite(tau)
        & np.isfinite(sigma)
        & (sigma > 0)
    )
    n_keep = int(np.count_nonzero(keep))
    if n_keep == 0:
        report = {
            "lag_band": (lag_lo, lag_hi),
            "n_inflated": 0,
            "factor": float(factor),
            "mode": mode,
            "systematic_sigma": float("nan"),
            "median_inflation_factor": 1.0,
            "max_inflation_factor": 1.0,
        }
        return sigma.copy(), report

    if systematic_sigma is not None:
        sys_arr = np.asarray(systematic_sigma, float)
        if sys_arr.ndim == 0:
            sys_band = np.full(n_keep, abs(float(sys_arr)))
            sys_report = abs(float(sys_arr))
            mode_used = "scalar"
        elif sys_arr.shape == tau.shape:
            sys_band = np.abs(sys_arr[keep])
            sys_report = float(np.nanmedian(sys_band))
            mode_used = "per_lag"
        else:
            raise ValueError("systematic_sigma must be scalar or match tau shape.")
    elif residual is not None:
        res = np.asarray(residual, float)
        if res.shape != tau.shape:
            raise ValueError("residual must have the same shape as tau.")
        r = res[keep]
        finite = np.isfinite(r)
        if not np.any(finite):
            sys_band = np.zeros(n_keep)
            sys_report = 0.0
        elif mode == "rms":
            t = np.log(tau[keep][finite])
            y2 = r[finite] * r[finite]
            if y2.size == 1 or t[-1] == t[0]:
                sys_report = float(math.sqrt(max(y2[0], 0.0)))
            else:
                sys_report = float(math.sqrt(max(np.trapz(y2, t) / (t[-1] - t[0]), 0.0)))
            sys_band = np.full(n_keep, sys_report)
        elif mode == "mad":
            rv = r[finite]
            sys_report = float(1.4826 * np.nanmedian(np.abs(rv - np.nanmedian(rv))))
            sys_band = np.full(n_keep, sys_report)
        elif mode == "abs":
            sys_band = np.abs(np.where(np.isfinite(r), r, 0.0))
            sys_report = float(np.nanmedian(sys_band))
        else:
            raise ValueError("mode must be 'rms', 'mad', or 'abs'.")
        mode_used = mode
    else:
        raise ValueError("Provide either systematic_sigma or residual.")

    sys_band = float(factor) * np.asarray(sys_band, float)
    sigma_eff = sigma.copy()
    before = sigma_eff[keep].copy()
    sigma_eff[keep] = np.sqrt(before * before + sys_band * sys_band)
    infl = sigma_eff[keep] / np.clip(before, 1e-12, None)
    report = {
        "lag_band": (lag_lo, lag_hi),
        "n_inflated": n_keep,
        "factor": float(factor),
        "mode": mode_used,
        "systematic_sigma": float(abs(float(factor)) * sys_report),
        "median_inflation_factor": float(np.nanmedian(infl)),
        "max_inflation_factor": float(np.nanmax(infl)),
    }
    return sigma_eff, report


# ---------------------------------------------------------------------------
# Core single-angle CORENN inversion
# ---------------------------------------------------------------------------
def _solve_nlwls(K: np.ndarray, g2m: np.ndarray, weights: np.ndarray,
                 beta: float, D2: np.ndarray, lam: float,
                 x0: np.ndarray, fit_baseline: bool = False
                 ) -> Tuple[np.ndarray, float]:
    """Non-linear weighted least squares with x >= 0 and Tikhonov penalty.

    Minimizes  sum_i [w_i (g2m_i - beta (Kx)_i^2 - b)]^2 + lam ||D2 x||^2
    over x >= 0 (and a free baseline b when ``fit_baseline`` is True), using a
    trust-region reflective solver with analytic Jacobian.  The free baseline
    absorbs imperfect long-lag baseline subtraction, a CORENN cornerstone.

    Returns ``(x, baseline)``.
    """
    sqlam = math.sqrt(max(lam, 0.0))
    n = x0.size
    p0 = np.concatenate([x0, [0.0]]) if fit_baseline else x0

    def residual(p):
        x = p[:n]
        b = p[n] if fit_baseline else 0.0
        u = K @ x
        r_data = weights * (g2m - beta * u * u - b)
        r_reg = sqlam * (D2 @ x)
        return np.concatenate([r_data, r_reg])

    def jac(p):
        x = p[:n]
        u = K @ x
        # d/dx_j [ -beta (Kx)_i^2 ] = -2 beta (Kx)_i K_ij
        J_data = (weights[:, None]) * (-2.0 * beta * u[:, None] * K)
        J_reg = sqlam * D2
        if fit_baseline:
            db = (-weights)[:, None]                    # d r_data / d b
            J_data = np.hstack([J_data, db])
            J_reg = np.hstack([J_reg, np.zeros((D2.shape[0], 1))])
        return np.vstack([J_data, J_reg])

    if fit_baseline:
        lo = np.concatenate([np.zeros(n), [-np.inf]])
        hi = np.full(n + 1, np.inf)
    else:
        lo, hi = 0.0, np.inf
    sol = least_squares(
        residual, p0, jac=jac, bounds=(lo, hi),
        method="trf", max_nfev=400, ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    if fit_baseline:
        return sol.x[:n], float(sol.x[n])
    return sol.x, 0.0


def _gcv_score(K: np.ndarray, weights: np.ndarray, beta: float,
               D2: np.ndarray, lam: float, x: np.ndarray,
               g2m: np.ndarray, baseline: float = 0.0) -> float:
    """Generalized cross validation score at the converged solution.

    Uses the Gauss-Newton linearization J = d(model)/dx at x to form the
    effective degrees of freedom dof = tr[J (J^T J + lam D2^T D2)^{-1} J^T].
    """
    u = K @ x
    J = (weights[:, None]) * (2.0 * beta * u[:, None] * K)   # n_data x n_basis
    n = g2m.size
    A = J.T @ J + lam * (D2.T @ D2)
    try:
        M = np.linalg.solve(A, J.T)           # (n_basis x n_data)
    except np.linalg.LinAlgError:
        return np.inf
    dof = float(np.einsum("ij,ji->", J, M))   # tr(J A^{-1} J^T)
    resid = weights * (g2m - beta * u * u - baseline)
    rss = float(resid @ resid)
    denom = (1.0 - dof / n) ** 2
    return rss / n / denom if denom > 1e-12 else np.inf


def _detect_peaks(R: np.ndarray, psd: np.ndarray,
                  min_rel_amount: float = 0.01) -> List[Dict[str, float]]:
    """Find modes of a PSD and report mean radius, relative amount, CV.

    Peaks carrying less than ``min_rel_amount`` of the total area are dropped
    (these are regularization ripples, not physical modes).
    """
    psd = np.clip(psd, 0.0, None)
    total = np.trapz(psd, R)
    if total <= 0:
        return []
    # split at local minima between local maxima
    is_max = np.r_[False, (psd[1:-1] > psd[:-2]) & (psd[1:-1] >= psd[2:]), False]
    maxima = np.where(is_max)[0]
    if maxima.size == 0:
        maxima = np.array([int(np.argmax(psd))])
    # boundaries: midpoints (in log R) between successive maxima at the minima
    bounds = [0]
    for a, b in zip(maxima[:-1], maxima[1:]):
        seg = psd[a:b + 1]
        bounds.append(a + int(np.argmin(seg)))
    bounds.append(len(R) - 1)
    peaks = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sl = slice(lo, hi + 1)
        seg_R, seg = R[sl], psd[sl]
        amount = np.trapz(seg, seg_R)
        if amount <= 0:
            continue
        mean = np.trapz(seg_R * seg, seg_R) / amount
        var = np.trapz((seg_R - mean) ** 2 * seg, seg_R) / amount
        cv = math.sqrt(max(var, 0.0)) / mean if mean > 0 else 0.0
        rel = amount / total
        if rel < min_rel_amount:
            continue
        peaks.append({
            "radius_nm": mean * 1e9,
            "rel_amount": rel,
            "cv": cv,
        })
    peaks.sort(key=lambda p: p["rel_amount"], reverse=True)
    return peaks


def corenn_invert(
    dataset: DLSDataset,
    radius_range: Tuple[float, float] = DEFAULT_CORENN_RADIUS_RANGE,
    n_basis: int = 24,
    n_R: int = 160,
    n_lambda: int = 24,
    lambda_range: Tuple[float, float] = (1e-6, 1e3),
    temperature: Optional[float] = None,
    viscosity: Optional[float] = None,
    fit_beta: bool = False,
    fit_baseline: bool = False,
    lag_min: Optional[float] = None,
    lag_max: Optional[float] = None,
    uncertainty: Optional[str] = None,
    replicates: Optional[Sequence[np.ndarray]] = None,
    sigma_inflation: Optional[Dict[str, object]] = None,
    n_bootstrap: int = 0,
    random_state: int = 0,
) -> CorennResult:
    """Recover an intensity-weighted PSD from one DLS curve via CORENN.

    Parameters
    ----------
    dataset : DLSDataset
        The measured curve (``g2 - 1`` vs tau at one q).
    radius_range, n_basis, n_R : grid/basis controls.
    n_lambda, lambda_range : GCV scan for the Tikhonov weight.
    temperature, viscosity : override the dataset values [K], [Pa s].
    fit_beta : if True the coherence factor is profiled out per iteration;
        otherwise the dataset's ``beta`` is used.
    uncertainty : {None, 'linearized', 'bootstrap'} band estimator.
    replicates : per-lag replicate ``g2-1`` curves, used both for empirical
        sigma (if the dataset has none) and for bootstrap bands.
    sigma_inflation : dict, optional
        Add systematic uncertainty in quadrature over a lag band.  Expected keys
        are ``lag_band`` and either ``systematic_sigma`` or ``residual``; optional
        keys are ``factor`` and ``mode``.  This is intended for reproducible
        residual wiggles that should not become PSD mass.
    n_bootstrap : number of bootstrap inversions for the band.

    Returns
    -------
    CorennResult
    """
    T = temperature if temperature is not None else dataset.temperature
    eta = viscosity if viscosity is not None else dataset.viscosity
    if T is None or eta is None:
        raise ValueError("temperature and viscosity must be provided.")

    tau, g2m, q, beta = dataset.tau, dataset.g2m, dataset.q, dataset.beta

    # --- optional lag-window truncation ----------------------------------
    # Truncating the hardware correlogram to intermediate lags (e.g. < 1 s)
    # is the standard way to exclude slow artifacts from large aggregates;
    # the long-lag dynamics are better recovered from a software correlation
    # of the count traces (see hybrid_correlator.py).
    sigma = dataset.sigma
    if lag_min is not None or lag_max is not None:
        keep = np.ones(tau.size, bool)
        if lag_min is not None:
            keep &= tau >= lag_min
        if lag_max is not None:
            keep &= tau <= lag_max
        tau, g2m = tau[keep], g2m[keep]
        if sigma is not None and np.asarray(sigma).shape == keep.shape:
            sigma = np.asarray(sigma, float)[keep]
        if replicates is not None:
            replicates = [np.asarray(r, float)[keep] for r in replicates
                          if np.asarray(r).shape == keep.shape]

    # --- uncertainty / weights -------------------------------------------
    # The correct DLS noise grows at long lag (fewer independent samples,
    # Var ~ tau/T).  We take the *shape* of sigma(tau) from a Schaetzel-style
    # model and, when replicates exist, calibrate its absolute scale to the
    # measured short-lag scatter.  This down-weights the long-lag noise floor
    # so it is not mistaken for a slow (huge-particle) relaxation.
    g1_proxy = np.sqrt(np.clip(g2m, 0.0, None) / max(beta, 1e-12))
    duration = dataset.duration if dataset.duration else float(tau[-1])
    sigma_emp = (empirical_sigma(replicates)
                 if replicates is not None and len(replicates) >= 2 else None)
    if sigma is None:
        # short-lag floor from replicate scatter (or a small default), plus
        # the Schaetzel 1/M(tau) growth that down-weights long lags.
        if sigma_emp is not None:
            k = max(3, int(0.15 * tau.size))
            floor = float(np.median(sigma_emp[:k]))
        else:
            floor = 0.01 * max(beta, 1e-3)
        sigma = schatzel_sigma(tau, g1_proxy, beta, duration, sigma_floor=floor)
        if sigma_emp is not None:
            sigma = np.maximum(sigma, sigma_emp)  # never under-weight noisy lags
    sigma_inflation_report = None
    if sigma_inflation is not None:
        cfg = dict(sigma_inflation)
        lag_band = cfg.pop("lag_band")
        if "sigma" in cfg and "systematic_sigma" not in cfg:
            cfg["systematic_sigma"] = cfg.pop("sigma")
        sigma, sigma_inflation_report = inflate_sigma_in_lag_band(
            tau, sigma, lag_band, **cfg)
    weights = 1.0 / np.clip(sigma, 1e-12, None)

    # --- discretization ---------------------------------------------------
    R = build_radius_grid(radius_range[0], radius_range[1], n_R)
    w_R = trapezoid_weights(R)
    Phi = build_bspline_basis(R, n_basis)
    D2 = second_difference_operator(n_basis)
    K = build_kernel(q, tau, R, w_R, Phi, T, eta)

    # --- warm start from the (biased) linearized NNLS --------------------
    # sqrt((g2-1)/beta) ~ K x ; only to seed the non-linear solve.
    target = np.sqrt(np.clip(g2m, 0.0, None) / max(beta, 1e-12))
    Aw = np.vstack([weights[:, None] * K, math.sqrt(1e-3) * D2])
    bw = np.concatenate([weights * target, np.zeros(D2.shape[0])])
    x0, _ = nnls(Aw, bw, maxiter=2000)
    if not np.any(x0 > 0):
        x0 = np.full(n_basis, 1e-6)

    # --- profile beta if requested ---------------------------------------
    def maybe_beta(x):
        if not fit_beta:
            return beta
        u = K @ x
        num = float(np.sum((weights**2) * g2m * u * u))
        den = float(np.sum((weights**2) * u**4))
        return num / den if den > 0 else beta

    # --- GCV scan over lambda (full non-linear solve per lambda) ----------
    lam_grid = np.geomspace(lambda_range[0], lambda_range[1], n_lambda)
    gcv = np.full(n_lambda, np.inf)
    best = {"gcv": np.inf, "lam": lam_grid[0], "x": x0, "beta": beta, "base": 0.0}
    x_warm = x0.copy()
    for i, lam in enumerate(lam_grid):
        b_eff = maybe_beta(x_warm)
        x, base = _solve_nlwls(K, g2m, weights, b_eff, D2, lam, x_warm,
                               fit_baseline=fit_baseline)
        b_eff = maybe_beta(x)
        score = _gcv_score(K, weights, b_eff, D2, lam, x, g2m, baseline=base)
        gcv[i] = score
        x_warm = x
        if score < best["gcv"]:
            best = {"gcv": score, "lam": lam, "x": x.copy(),
                    "beta": b_eff, "base": base}

    x_hat, lam_hat, beta_hat, baseline_hat = (
        best["x"], best["lam"], best["beta"], best["base"])

    # --- PSDs and diagnostics --------------------------------------------
    I_R = np.clip(Phi @ x_hat, 0.0, None)
    area = np.trapz(I_R, R)
    if area > 0:
        I_R = I_R / area
    intensity = I_R
    # weighting conversions in the Rayleigh limit: I ~ N R^6 ~ V R^3
    with np.errstate(divide="ignore", invalid="ignore"):
        number = np.where(R > 0, intensity / R**6, 0.0)
        volume = np.where(R > 0, intensity / R**3, 0.0)
    number_area = np.trapz(number, R)
    volume_area = np.trapz(volume, R)
    number = number / number_area if number_area > 0 else number
    volume = volume / volume_area if volume_area > 0 else volume

    u = K @ x_hat
    g2_model = beta_hat * u * u + baseline_hat
    resid = weights * (g2m - g2_model)
    residual_norm = float(np.sqrt(np.mean(resid**2)))
    peaks = _detect_peaks(R, intensity)

    # --- fit diagnostics: effective dof and conditioning ------------------
    # linearized Jacobian at the solution; eff_dof = tr(H), H the influence
    # (smoother) matrix; cond = conditioning of the regularized normal matrix.
    Jl = (weights[:, None]) * (2.0 * beta_hat * u[:, None] * K)
    A_reg = Jl.T @ Jl + lam_hat * (D2.T @ D2)
    try:
        eff_dof = float(np.einsum("ij,ji->", Jl, np.linalg.solve(A_reg, Jl.T)))
        cond = float(np.linalg.cond(A_reg))
    except np.linalg.LinAlgError:
        eff_dof = cond = float("nan")

    # --- uncertainty bands -----------------------------------------------
    band = None
    if uncertainty == "linearized":
        Jl = (weights[:, None]) * (2.0 * beta_hat * u[:, None] * K)
        A = Jl.T @ Jl + lam_hat * (D2.T @ D2)
        try:
            cov = np.linalg.inv(A)
            var_I = np.einsum("ij,jk,ik->i", Phi, cov, Phi)
            sd_I = np.sqrt(np.clip(var_I, 0.0, None))
            if area > 0:
                sd_I = sd_I / area
            band = (np.clip(intensity - 1.96 * sd_I, 0.0, None),
                    intensity + 1.96 * sd_I)
        except np.linalg.LinAlgError:
            band = None
    elif uncertainty == "bootstrap" and replicates is not None and len(replicates) >= 2:
        rng = np.random.default_rng(random_state)
        reps = [np.asarray(r, float) for r in replicates]
        nb = n_bootstrap if n_bootstrap > 0 else 100
        boot = np.empty((nb, R.size))
        for b in range(nb):
            idx = rng.integers(0, len(reps), len(reps))
            g_b = np.mean([reps[k] for k in idx], axis=0)
            xb, _ = _solve_nlwls(K, g_b, weights, beta_hat, D2, lam_hat, x_hat,
                                 fit_baseline=fit_baseline)
            ib = np.clip(Phi @ xb, 0.0, None)
            a = np.trapz(ib, R)
            boot[b] = ib / a if a > 0 else ib
        band = (np.percentile(boot, 2.5, axis=0),
                np.percentile(boot, 97.5, axis=0))

    return CorennResult(
        R=R, intensity=intensity, volume=volume, number=number,
        x=x_hat, beta=beta_hat, lam=lam_hat, lam_grid=lam_grid, gcv=gcv,
        tau=tau, g2m=g2m, g2_model=g2_model, residual_norm=residual_norm,
        peaks=peaks, intensity_band=band,
        extra={"weights": weights, "sigma": sigma, "Phi": Phi, "K": K,
               "baseline": baseline_hat, "eff_dof": eff_dof, "cond": cond,
               "converged": bool(np.isfinite(residual_norm)),
               "n_basis": int(n_basis), "n_data": int(g2m.size),
               "q": float(q), "temperature": float(T), "viscosity": float(eta),
               "radius_range": (float(radius_range[0]), float(radius_range[1])),
               "sigma_inflation": sigma_inflation_report},
    )


def corenn_predict_g2(
    result: CorennResult,
    dataset: Optional[DLSDataset] = None,
    tau: Optional[Sequence[float]] = None,
    q: Optional[float] = None,
    temperature: Optional[float] = None,
    viscosity: Optional[float] = None,
) -> np.ndarray:
    """Evaluate a single-angle CORENN result at arbitrary lag times.

    This is useful when a fit was performed on a restricted lag window and the
    same physical PSD must be extrapolated back to the full correlogram for
    residual diagnostics.
    """
    if tau is None:
        tau_arr = dataset.tau if dataset is not None else result.tau
    else:
        tau_arr = np.asarray(tau, float)

    if q is None:
        q = dataset.q if dataset is not None else result.extra.get("q")
    if temperature is None:
        temperature = (dataset.temperature if dataset is not None
                       else result.extra.get("temperature"))
    if viscosity is None:
        viscosity = (dataset.viscosity if dataset is not None
                     else result.extra.get("viscosity"))
    if q is None or temperature is None or viscosity is None:
        raise ValueError("q, temperature, and viscosity are required for prediction.")

    Phi = result.extra.get("Phi")
    if Phi is None:
        Phi = build_bspline_basis(result.R, int(result.extra.get("n_basis", result.x.size)))
    w_R = trapezoid_weights(result.R)
    K = build_kernel(float(q), tau_arr, result.R, w_R, Phi,
                     float(temperature), float(viscosity))
    g1 = K @ result.x
    baseline = float(result.extra.get("baseline", 0.0))
    return result.beta * g1 * g1 + baseline


def short_lag_residual_diagnostic(
    dataset: DLSDataset,
    replicates: Optional[Sequence[np.ndarray]] = None,
    lag_band: Tuple[float, float] = (8e-7, 1e-4),
    baseline_lag_min: Optional[float] = None,
    baseline_lag_max: Optional[float] = None,
    slow_radius_gamma_tau: float = 2.0,
    detrend_order: int = 2,
    invert_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Quantify microsecond-lag structure relative to a slow CORENN baseline.

    The baseline is a nonlinear CORENN fit outside ``lag_band``. It is then
    extrapolated back into the band, and the residual is summarized with
    replicate-normalized metrics. The fitted radius lower bound is raised to the
    smallest radius that remains dynamically visible at ``baseline_lag_min``;
    otherwise radii that have already fully decayed before the fit window would
    be unconstrained but would dominate the extrapolated intercept.
    """
    lag_lo, lag_hi = map(float, lag_band)
    if lag_lo <= 0 or lag_hi <= lag_lo:
        raise ValueError("lag_band must be a positive (low, high) interval.")
    if baseline_lag_min is None:
        baseline_lag_min = 3.0 * lag_hi
    baseline_lag_min = float(baseline_lag_min)

    T = dataset.temperature
    eta = dataset.viscosity
    if T is None or eta is None:
        raise ValueError("dataset temperature and viscosity are required.")

    gamma_tau = max(float(slow_radius_gamma_tau), 1e-6)
    slow_radius_floor = (
        dataset.q * dataset.q * k_BOLTZMANN * float(T) * baseline_lag_min
        / (6.0 * math.pi * float(eta) * gamma_tau)
    )

    kwargs = dict(invert_kwargs or {})
    rr = kwargs.get("radius_range", (DEFAULT_CORENN_RADIUS_RANGE[0], 5e-5))
    rr = (max(float(rr[0]), float(slow_radius_floor)), float(rr[1]))
    kwargs["radius_range"] = rr
    kwargs.setdefault("fit_beta", True)
    kwargs.setdefault("fit_baseline", True)

    baseline_result = corenn_invert(
        dataset,
        replicates=replicates,
        lag_min=baseline_lag_min,
        lag_max=baseline_lag_max,
        **kwargs,
    )
    g2_model = corenn_predict_g2(baseline_result, dataset=dataset, tau=dataset.tau)
    residual = np.asarray(dataset.g2m, float) - g2_model

    sigma_source = "schatzel"
    reps = [np.asarray(r, float) for r in (replicates or [])]
    if dataset.sigma is not None and np.asarray(dataset.sigma).shape == dataset.tau.shape:
        sigma = np.asarray(dataset.sigma, float)
        sigma_source = "dataset"
    elif len(reps) >= 2 and all(r.shape == dataset.tau.shape for r in reps):
        sigma = empirical_sigma(reps)
        sigma_source = "replicates"
    else:
        beta_ref = max(abs(float(dataset.beta)), abs(float(baseline_result.beta)), 1e-3)
        g1_proxy = np.sqrt(np.clip(g2_model, 0.0, None) / beta_ref)
        duration = dataset.duration if dataset.duration else float(dataset.tau[-1])
        sigma = schatzel_sigma(
            dataset.tau, g1_proxy, beta_ref, duration,
            sigma_floor=0.01 * beta_ref,
        )

    sigma = np.asarray(sigma, float)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    good = (
        (dataset.tau >= lag_lo)
        & (dataset.tau <= lag_hi)
        & np.isfinite(residual)
        & np.isfinite(sigma)
        & (sigma > 0)
    )
    n_band = int(np.count_nonzero(good))
    beta_norm = max(abs(float(dataset.beta)), abs(float(baseline_result.beta)), 1e-12)

    def _log_mean(values: np.ndarray) -> float:
        if n_band == 0:
            return float("nan")
        t = np.log(dataset.tau[good])
        y = np.asarray(values, float)[good]
        if y.size == 1 or t[-1] == t[0]:
            return float(y[0])
        return float(np.trapz(y, t) / (t[-1] - t[0]))

    wiggle_residual = np.full_like(residual, np.nan)
    residual_trend = np.full_like(residual, np.nan)
    if n_band:
        t_band = np.log(dataset.tau[good])
        r_band = residual[good]
        s_band = sigma[good]
        degree = int(min(max(detrend_order, 0), max(n_band - 1, 0)))
        if degree == 0 or n_band < 4:
            trend_band = np.full_like(r_band, _log_mean(residual))
        else:
            try:
                coeff = np.polyfit(t_band, r_band, degree, w=1.0 / s_band)
                trend_band = np.polyval(coeff, t_band)
            except (np.linalg.LinAlgError, ValueError):
                trend_band = np.full_like(r_band, _log_mean(residual))
        residual_trend[good] = trend_band
        wiggle_residual[good] = r_band - trend_band

    z = residual / np.clip(sigma, 1e-12, None)
    wz = wiggle_residual / np.clip(sigma, 1e-12, None)
    rms = math.sqrt(max(_log_mean(residual * residual), 0.0))
    rms_z = math.sqrt(max(_log_mean(z * z), 0.0))
    mean_resid = _log_mean(residual)
    mean_abs_resid = _log_mean(np.abs(residual))
    wiggle_rms = math.sqrt(max(_log_mean(wiggle_residual * wiggle_residual), 0.0))
    wiggle_rms_z = math.sqrt(max(_log_mean(wz * wz), 0.0))

    if n_band:
        r_band = residual[good]
        z_band = z[good]
        w_band = wiggle_residual[good]
        wz_band = wz[good]
        i_abs = int(np.argmax(np.abs(r_band)))
        peak_to_peak = float(np.nanmax(r_band) - np.nanmin(r_band))
        max_abs = float(np.nanmax(np.abs(r_band)))
        max_abs_z = float(np.nanmax(np.abs(z_band)))
        lag_at_max_abs = float(dataset.tau[good][i_abs])
        wiggle_peak_to_peak = float(np.nanmax(w_band) - np.nanmin(w_band))
        wiggle_max_abs = float(np.nanmax(np.abs(w_band)))
        wiggle_max_abs_z = float(np.nanmax(np.abs(wz_band)))
    else:
        peak_to_peak = max_abs = max_abs_z = lag_at_max_abs = float("nan")
        wiggle_peak_to_peak = wiggle_max_abs = wiggle_max_abs_z = float("nan")

    return {
        "baseline_result": baseline_result,
        "lag_band": (lag_lo, lag_hi),
        "baseline_lag_min": baseline_lag_min,
        "baseline_lag_max": baseline_lag_max,
        "slow_radius_floor_m": float(slow_radius_floor),
        "slow_radius_gamma_tau": float(gamma_tau),
        "tau": dataset.tau,
        "g2m": dataset.g2m,
        "g2_model": g2_model,
        "residual": residual,
        "residual_trend": residual_trend,
        "wiggle_residual": wiggle_residual,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "band_mask": good,
        "n_band": n_band,
        "detrend_order": int(detrend_order),
        "mean_residual_log": mean_resid,
        "mean_abs_residual_log": mean_abs_resid,
        "rms_residual_log": rms,
        "rms_residual_over_beta": rms / beta_norm,
        "rms_z_log": rms_z,
        "peak_to_peak": peak_to_peak,
        "peak_to_peak_over_beta": peak_to_peak / beta_norm,
        "max_abs_residual": max_abs,
        "max_abs_residual_over_beta": max_abs / beta_norm,
        "max_abs_z": max_abs_z,
        "wiggle_rms_log": wiggle_rms,
        "wiggle_rms_over_beta": wiggle_rms / beta_norm,
        "wiggle_rms_z_log": wiggle_rms_z,
        "wiggle_peak_to_peak": wiggle_peak_to_peak,
        "wiggle_peak_to_peak_over_beta": wiggle_peak_to_peak / beta_norm,
        "wiggle_max_abs_residual": wiggle_max_abs,
        "wiggle_max_abs_residual_over_beta": wiggle_max_abs / beta_norm,
        "wiggle_max_abs_z": wiggle_max_abs_z,
        "lag_at_max_abs_s": lag_at_max_abs,
        "q": float(dataset.q),
        "angle_deg": dataset.metadata.get("angle_deg"),
        "polarization": dataset.metadata.get("polarization"),
        "measurement": dataset.metadata.get("measurement"),
    }


def sigma_inflation_from_diagnostic(
    diagnostic: Dict[str, object],
    component: str = "wiggle",
    factor: float = 1.0,
    mode: str = "rms",
) -> Dict[str, object]:
    """Build a ``corenn_invert(..., sigma_inflation=...)`` config.

    ``component='wiggle'`` uses the detrended residual, which is the conservative
    default for suppressing non-CORENN oscillatory structure.  Use
    ``component='residual'`` only when the entire short-lag mismatch should be
    treated as systematic model discrepancy.
    """
    if component == "wiggle":
        residual_key = "wiggle_residual"
    elif component == "residual":
        residual_key = "residual"
    else:
        raise ValueError("component must be 'wiggle' or 'residual'.")
    if residual_key not in diagnostic:
        raise ValueError(f"diagnostic does not contain {residual_key!r}.")
    return {
        "lag_band": diagnostic["lag_band"],
        "residual": diagnostic[residual_key],
        "factor": float(factor),
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Multi-angle joint inversion for a q-independent NUMBER distribution
# ---------------------------------------------------------------------------
def corenn_multi_angle(
    datasets: Sequence[DLSDataset],
    radius_range: Tuple[float, float] = DEFAULT_CORENN_RADIUS_RANGE,
    n_basis: int = 24,
    n_R: int = 160,
    n_lambda: int = 20,
    lambda_range: Tuple[float, float] = (1e-6, 1e3),
    temperature: Optional[float] = None,
    viscosity: Optional[float] = None,
    use_form_factor: bool = True,
    lag_min: Optional[float] = None,
    lag_max: Optional[float] = None,
) -> CorennResult:
    """Joint inversion of several angles for a single q-consistent PSD.

    The unknown is the intensity PSD extrapolated to q->0, ``I0(R) ~ N(R) R^6``
    (q-independent).  At angle q the intensity weight of each size is scaled by
    the sphere form factor ``P(qR) <= 1`` (the q-independent ``R^6`` folds into
    I0), and the per-angle amplitude is the *measured* coherence factor
    ``beta_q`` -- not a free parameter.  Concretely the model at angle q is

        g2_q(tau) - 1 = beta_q * [ (K_q x)(tau) / (c_q . x) ]^2 ,

    where ``c_q,j = \\int Phi_j(R) P(qR) dR`` normalizes ``g1_q(0)=1``.  Fixing
    beta_q couples all angles through the single I0(R), which is exactly the
    cross-angle constraint that makes multi-angle informative (a free per-angle
    amplitude would decouple the angles and collapse the fit).
    """
    if not datasets:
        raise ValueError("datasets must be non-empty.")
    T = temperature if temperature is not None else datasets[0].temperature
    eta = viscosity if viscosity is not None else datasets[0].viscosity
    if T is None or eta is None:
        raise ValueError("temperature and viscosity must be provided.")

    R = build_radius_grid(radius_range[0], radius_range[1], n_R)
    w_R = trapezoid_weights(R)
    w_norm = w_R / np.sum(w_R)
    Phi = build_bspline_basis(R, n_basis)
    D2 = second_difference_operator(n_basis)

    Ks, cs, g_list, w_list, betas = [], [], [], [], []
    for ds in datasets:
        tau_d = np.asarray(ds.tau, float)
        g2_d = np.asarray(ds.g2m, float)
        sig_d = None if ds.sigma is None else np.asarray(ds.sigma, float)
        # optional lag window: focus the inversion on the main relaxation and
        # exclude the slow aggregate tail that the hybrid long-lag data exposes
        # (mirrors corenn_invert; consistent with the Gamma-vs-q^2 cumulant).
        if lag_min is not None or lag_max is not None:
            keep = np.ones(tau_d.size, bool)
            if lag_min is not None:
                keep &= tau_d >= lag_min
            if lag_max is not None:
                keep &= tau_d <= lag_max
            tau_d, g2_d = tau_d[keep], g2_d[keep]
            if sig_d is not None and sig_d.shape == keep.shape:
                sig_d = sig_d[keep]
        weight = sphere_form_factor(ds.q, R) if use_form_factor else np.ones_like(R)
        K = build_kernel(ds.q, tau_d, R, w_R, Phi, T, eta, weight=weight)
        c = (w_norm * weight) @ Phi          # normalization vector c_q (n_basis,)
        Ks.append(K); cs.append(c)
        g_list.append(g2_d)
        betas.append(ds.beta if ds.beta not in (None, 0) else 1.0)
        if sig_d is not None and sig_d.shape == g2_d.shape:
            w_list.append(1.0 / np.clip(sig_d, 1e-12, None))
        else:
            n = tau_d.size
            rel = (np.arange(n) + 1.0) / np.maximum(n - np.arange(n), 1.0)
            s = np.maximum(np.abs(g2_d) * rel, np.max(np.abs(g2_d)) * 1e-3)
            w_list.append(1.0 / s)

    def model_and_jac(x):
        """Return stacked (residual_data, J_data) for the fixed-beta model."""
        res, Jrows = [], []
        for K, c, g, w, b in zip(Ks, cs, g_list, w_list, betas):
            u = K @ x
            z = float(c @ x)
            z = z if abs(z) > 1e-30 else 1e-30
            g1 = u / z
            model = b * g1 * g1
            res.append(w * (g - model))
            # d model_i/dx_j = 2 b g1_i (K_ij/z - u_i c_j / z^2)
            dg1 = K / z - np.outer(u, c) / (z * z)
            Jrows.append((w[:, None]) * (-2.0 * b * g1[:, None] * dg1))
        return np.concatenate(res), np.vstack(Jrows)

    def residual_full(x, lam):
        res, _ = model_and_jac(x)
        return np.concatenate([res, math.sqrt(max(lam, 0.0)) * (D2 @ x)])

    def jac_full(x, lam):
        _, J = model_and_jac(x)
        return np.vstack([J, math.sqrt(max(lam, 0.0)) * D2])

    # warm start: NNLS on a mid-q curve (best size sensitivity, mild form factor)
    ds0 = sorted(datasets, key=lambda d: d.q)[len(datasets) // 2]
    weight0 = sphere_form_factor(ds0.q, R) if use_form_factor else None
    K0 = build_kernel(ds0.q, ds0.tau, R, w_R, Phi, T, eta, weight=weight0)
    target0 = np.sqrt(np.clip(ds0.g2m, 0.0, None) / max(ds0.beta, 1e-12))
    A0 = np.vstack([K0, math.sqrt(1e-3) * D2])
    b0 = np.concatenate([target0, np.zeros(D2.shape[0])])
    x0, _ = nnls(A0, b0, maxiter=2000)
    if not np.any(x0 > 0):
        x0 = np.full(n_basis, 1e-6)

    lam_grid = np.geomspace(lambda_range[0], lambda_range[1], n_lambda)
    n_data = sum(g.size for g in g_list)
    gcv = np.full(n_lambda, np.inf)
    best = {"gcv": np.inf, "lam": lam_grid[0], "x": x0}
    x_warm = x0.copy()
    for i, lam in enumerate(lam_grid):
        sol = least_squares(
            residual_full, x_warm, jac=jac_full, args=(lam,),
            bounds=(0.0, np.inf), method="trf", max_nfev=400,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        x = sol.x
        res, J = model_and_jac(x)
        rss = float(res @ res)
        Amat = J.T @ J + lam * (D2.T @ D2)
        dof = float("nan")
        try:
            Msolve = np.linalg.solve(Amat, J.T)
            dof = float(np.einsum("ij,ji->", J, Msolve))
            denom = (1.0 - dof / n_data) ** 2
            score = rss / n_data / denom if denom > 1e-12 else np.inf
        except np.linalg.LinAlgError:
            score = np.inf
        gcv[i] = score
        x_warm = x
        if score < best["gcv"]:
            best = {"gcv": score, "lam": lam, "x": x.copy(), "eff_dof": dof,
                    "cond": float(np.linalg.cond(Amat)),
                    "converged": bool(sol.success)}

    x_hat, lam_hat = best["x"], best["lam"]
    # The recovered coefficients describe the q->0 intensity PSD I0(R).
    intensity = np.clip(Phi @ x_hat, 0.0, None)
    a = np.trapz(intensity, R)
    intensity = intensity / a if a > 0 else intensity
    with np.errstate(divide="ignore", invalid="ignore"):
        number = np.where(R > 0, intensity / R**6, 0.0)
        volume = np.where(R > 0, intensity / R**3, 0.0)
    number_area = np.trapz(number, R)
    volume_area = np.trapz(volume, R)
    number = number / number_area if number_area > 0 else number
    volume = volume / volume_area if volume_area > 0 else volume

    # model curve for the first dataset for diagnostics (fixed-beta model)
    z0 = float(cs[0] @ x_hat) or 1e-30
    g2_model0 = betas[0] * (Ks[0] @ x_hat / z0) ** 2
    peaks = _detect_peaks(R, intensity)
    return CorennResult(
        R=R, intensity=intensity, volume=volume, number=number,
        x=x_hat, beta=float(betas[0]), lam=lam_hat, lam_grid=lam_grid, gcv=gcv,
        tau=datasets[0].tau, g2m=datasets[0].g2m, g2_model=g2_model0,
        residual_norm=float(np.sqrt(np.mean((g_list[0] - g2_model0) ** 2))),
        peaks=peaks,
        extra={"betas": betas, "Ks": Ks, "cs": cs, "Phi": Phi,
               "q": [d.q for d in datasets],
               "eff_dof": best.get("eff_dof", float("nan")),
               "cond": best.get("cond", float("nan")),
               "converged": best.get("converged"),
               "n_basis": int(n_basis), "n_data": int(n_data)},
    )


# ---------------------------------------------------------------------------
# LS Instruments file loaders
# ---------------------------------------------------------------------------
def _read_two_column(path: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path, skiprows=2)
    df = df.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    df = df.dropna()
    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)


def load_correlation_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read an LS Instruments 'Correlation Function.csv' -> (tau[s], g2-1)."""
    return _read_two_column(path)


def load_summary(path: str) -> Dict[str, float]:
    """Read a measurement/repetition 'Summary.csv' into a flat dict.

    ``index_col=False`` guards against a trailing comma in the data row (present
    in some exports, e.g. SISV) that would otherwise make pandas promote the
    first column to the index and shift every value by one.
    """
    import pandas as pd
    df = pd.read_csv(path, index_col=False)
    row = df.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def load_count_trace(path: str) -> Dict[str, np.ndarray]:
    """Read 'Count Trace.csv' -> {'time','cr_a','cr_b'} (count rates in Hz)."""
    import pandas as pd
    df = pd.read_csv(path, skiprows=2)
    df = df.apply(lambda c: pd.to_numeric(c, errors="coerce")).dropna()
    cols = list(df.columns)
    out = {"time": df[cols[0]].to_numpy(float)}
    if len(cols) > 1:
        out["cr_a"] = df[cols[1]].to_numpy(float) * 1e3   # kHz -> Hz
    if len(cols) > 2:
        out["cr_b"] = df[cols[2]].to_numpy(float) * 1e3
    return out


def static_intensity_proxy(dataset: DLSDataset) -> Optional[float]:
    """Return the LS Instruments static intensity proxy for SLS/Guinier fits.

    The exported ``Scaled count rate [kHz/mW]`` is already the laser-power
    normalized static count-rate proxy used here.  Do not apply an additional
    ``sin(theta)`` factor in the Guinier path.
    """
    value = dataset.metadata.get("scaled_count_rate")
    if value is None:
        value = dataset.metadata.get("count_rate_khz")
    if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
        return float(value)
    return None


def build_dataset_from_repetition(
    rep_dir: str,
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
    correlation_file: str = "Correlation Function.csv",
    summary_file: str = "Summary.csv",
) -> DLSDataset:
    """Construct a :class:`DLSDataset` from one repetition directory."""
    tau, g2m = load_correlation_csv(os.path.join(rep_dir, correlation_file))
    meta: Dict[str, object] = {"path": rep_dir}
    beta, T, eta, n_sol, q = 1.0, None, None, 1.33, None
    duration, crate = None, None
    spath = os.path.join(rep_dir, summary_file)
    if os.path.exists(spath):
        s = load_summary(spath)
        beta = float(s.get("Intercept", 1.0) or 1.0)
        T = s.get("Temperature [K]")
        visc = s.get("Solvent viscosity [mPas]")
        eta = float(visc) * 1e-3 if isinstance(visc, (int, float)) else None
        n_sol = float(s.get("Solvent refractive index", 1.33) or 1.33)
        angle = s.get("Scattering angle [deg]")
        if isinstance(angle, (int, float)):
            q = q_from_angle(float(angle), n_sol, wavelength_nm)
            meta["angle_deg"] = float(angle)
        else:
            # Angle missing: fall back to the vendor scattering-vector column, which
            # is MISLABELLED "[1/nm]" but actually stores values in 1/m.  Standardize
            # to 1/m (SI) by magnitude — a genuine 1/nm light-scattering q is <1,
            # whereas the mislabelled 1/m value is ~1e6–1e8.
            sv = s.get("Scattering vector [1/nm]")
            if isinstance(sv, (int, float)) and sv > 0:
                q = float(sv) * (1e9 if float(sv) < 1.0 else 1.0)
        duration = _parse_duration(s.get("Duration [s]"))
        cr = s.get("Count rate [kHz]")
        if not isinstance(cr, (int, float)):
            cr = s.get("Average count rate A [kHz]")
        crate = float(cr) * 1e3 if isinstance(cr, (int, float)) else None
        # static-intensity proxies for Guinier / SLS (laser-power corrected)
        scr = s.get("Scaled count rate [kHz/mW]")
        if isinstance(scr, (int, float)):
            meta["scaled_count_rate"] = float(scr)
        if isinstance(cr, (int, float)):
            meta["count_rate_khz"] = float(cr)
        tk = s.get("Temperature [K]")
        if isinstance(tk, (int, float)):
            meta["temperature_K"] = float(tk)
    if q is None:
        raise ValueError(f"Could not determine q for {rep_dir}.")
    return DLSDataset(tau=tau, g2m=g2m, q=q, beta=beta, temperature=T,
                      viscosity=eta, n_solvent=n_sol, duration=duration,
                      count_rate=crate, metadata=meta)


def _parse_duration(value) -> Optional[float]:
    """Parse a duration that may be seconds or an 'HH:MM:SS' string."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    if isinstance(value, str) and ":" in value:
        parts = [float(p) for p in value.split(":")]
        secs = 0.0
        for p in parts:
            secs = secs * 60.0 + p
        return secs
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_hardware_replicates(
    measurement_dir: str,
    correlation_file: str = "Correlation Function.csv",
) -> Tuple[DLSDataset, List[np.ndarray]]:
    """Hardware-only: replicate-averaged dataset + per-rep ``g2-1`` curves.

    This is the low-level loader (no software/hybrid correlation) used as the
    base for the hybrid merge and as the fallback when no count traces exist.
    """
    rep_dirs = sorted(glob.glob(os.path.join(measurement_dir, "Repetition *")))
    if not rep_dirs:
        raise ValueError(f"No repetitions found under {measurement_dir}.")
    base = build_dataset_from_repetition(rep_dirs[0])
    tau_ref = base.tau
    reps = []
    for rd in rep_dirs:
        try:
            tau, g = load_correlation_csv(os.path.join(rd, correlation_file))
        except Exception:
            continue
        if tau.shape == tau_ref.shape and np.allclose(tau, tau_ref, rtol=1e-3):
            reps.append(g)
    if len(reps) >= 2:
        # Use the replicate mean as the curve but leave sigma unset: the
        # inverter calibrates a Schaetzel-shaped sigma from these replicates,
        # which captures the long-lag noise growth that a flat empirical
        # std would miss.
        base.g2m = np.mean(reps, axis=0)
        base.sigma = None
    return base, reps


def load_measurement_replicates(
    measurement_dir: str,
    correlation_file: str = "Correlation Function.csv",
    hybrid: bool = False,
    stitched: bool = False,
) -> Tuple[DLSDataset, List[np.ndarray]]:
    """Load a measurement's replicate-averaged dataset and per-rep curves.

    By default returns the hardware replicate-mean dataset (with an unset sigma
    so the inverter calibrates a Schaetzel-shaped weighting).  Opt in with
    ``hybrid=True`` to merge the hardware correlogram with a per-repetition
    *pooled* software correlation of the count-rate traces, or ``stitched=True``
    for the **gap-aware stitched** software correlation -- the whole run
    correlated as one long record (see
    :func:`hybrid_correlator.build_stitched_dataset`).  The stitched merge gives
    clean long-lag statistics, and because :func:`hybrid_correlator.merge_correlations`
    rescales each measurement's software curve to *its own* hardware overlap, it
    is well behaved across angles -- i.e. suitable for the cross-angle coupled
    inversion :func:`corenn_multi_angle` (the older per-rep merge was not), where
    it yields a better-determined main mode by replacing the noise-dominated
    hardware long-lag.  Note the multi-angle *intensity* PSD upper radius remains
    set by the q-resolution guardrail (~pi/q_min): the sphere form factor P(qR)
    is meaningless for R >> pi/q, so pushing the grid larger only edge-piles the
    R^6-weighted intensity.  For genuine large-aggregate (slow-mode) sizing use
    the single-angle :func:`corenn_invert` on a stitched dataset (see
    ``demo_stitched_correlation.py``).
    When hybrid/stitched is used the hardware mean/lag grid are stashed in
    ``metadata['hardware_g2m'/'hardware_tau']``.  The second return value is
    always the list of hardware per-rep ``g2-1`` curves.
    """
    base, reps = _load_hardware_replicates(measurement_dir, correlation_file)
    if (hybrid or stitched) and len(reps) >= 2:
        has_traces = any(
            os.path.exists(os.path.join(rd, "Count Trace.csv"))
            for rd in glob.glob(os.path.join(measurement_dir, "Repetition *"))
        )
        if has_traces:
            try:
                import hybrid_correlator as _hc
                builder = (_hc.build_stitched_dataset if stitched
                           else _hc.build_hybrid_dataset)
                ds, _ = builder(measurement_dir, correlation_file=correlation_file)
                ds.metadata.setdefault("hardware_tau", base.tau)
                ds.metadata.setdefault("hardware_g2m", base.g2m)
                ds.metadata["hybrid"] = True
                return ds, reps
            except Exception:
                pass
    return base, reps


# ---------------------------------------------------------------------------
# Polarized / multi-angle MAPCS experiment loader
# ---------------------------------------------------------------------------
def load_mapcs_experiment(
    temperature_dir: str,
    manifest_path: Optional[str] = None,
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
    hybrid: bool = False,
    stitched: bool = False,
) -> Dict[str, List[DLSDataset]]:
    """Load an LS Instruments MAPCS experiment with VH/VV/3D measurements.

    The experiment manifest (``*MAPCS*.csv``) lists, per measurement, the
    scattering angle, the correlation type ("Pseudo Cross Correlation" for the
    2D polarized/depolarized channels, "Modulated 3D Cross" for the 3D
    cross-correlation), and the scaled count rate.  Each angle has three
    measurements; the two Pseudo-Cross ones are the depolarized (VH, low
    intensity) and polarized (VV, high intensity) channels, the third is 3D.

    Returns a dict ``{'VH': [...], 'VV': [...], '3D': [...]}`` of
    :class:`DLSDataset`, each sorted by scattering angle, with ``polarization``,
    ``angle_deg`` and ``scaled_count_rate`` in ``metadata``.
    """
    import pandas as pd

    if manifest_path is None:
        cands = (glob.glob(os.path.join(temperature_dir, "*MAPCS*.csv"))
                 or [p for p in glob.glob(os.path.join(temperature_dir, "*.csv"))])
        if not cands:
            raise ValueError(f"No manifest CSV found in {temperature_dir}.")
        manifest_path = cands[0]

    # the DLS-measurements table header is a few rows into the manifest
    man = None
    for skip in range(0, 8):
        try:
            tmp = pd.read_csv(manifest_path, skiprows=skip, engine="python",
                              on_bad_lines="skip")
        except Exception:
            continue
        if "Name" in tmp.columns and "Correlation type" in tmp.columns:
            man = tmp
            break
    if man is None:
        raise ValueError(f"Could not parse manifest {manifest_path}.")

    man = man[man["Name"].astype(str).str.startswith("Measurement")].copy()
    man["m"] = man["Name"].str.extract(r"(\d+)").astype(int)
    man["ang"] = np.round(pd.to_numeric(man["Scattering angle [deg]"],
                                        errors="coerce"), 0)
    man["scr"] = pd.to_numeric(man["Scaled count rate [kHz/mW]"], errors="coerce")

    out: Dict[str, List[DLSDataset]] = {"VH": [], "VV": [], "3D": []}
    for ang, grp in man.groupby("ang"):
        ctype = grp["Correlation type"].astype(str)
        pseudo = grp[ctype.str.contains("Pseudo", na=False)].sort_values("scr")
        mod3d = grp[ctype.str.contains("3D", na=False)]
        assign = []
        if len(pseudo) >= 2:                       # low scr = VH, high scr = VV
            assign += [("VH", pseudo.iloc[0]), ("VV", pseudo.iloc[-1])]
        elif len(pseudo) == 1:
            assign += [("VV", pseudo.iloc[0])]
        if len(mod3d):
            assign += [("3D", mod3d.iloc[0])]
        for pol, row in assign:
            meas_dir = os.path.join(temperature_dir, f"Measurement {int(row['m'])}")
            reps: List[np.ndarray] = []
            try:
                # multi-rep measurements are loaded with the hybrid merge by
                # default; single-rep ones fall back to the hardware curve.
                ds, reps = load_measurement_replicates(meas_dir, hybrid=hybrid,
                                                       stitched=stitched)
            except Exception:
                rep0 = os.path.join(meas_dir, "Repetition 0")
                try:
                    ds = build_dataset_from_repetition(rep0, wavelength_nm=wavelength_nm)
                except Exception:
                    continue
            # expose the per-repetition g2-1 curves for replicate-level analysis
            # (e.g. a true replicate bootstrap).  Only keep them when they align
            # with this dataset's lag grid -- under the hybrid merge ds.g2m lives
            # on a different (stitched) grid than the hardware replicates.
            aligned = [np.asarray(r, float) for r in reps
                       if np.asarray(r).shape == np.asarray(ds.g2m).shape]
            ds.metadata.update({"polarization": pol,
                                "angle_deg": float(row["ang"]),
                                "scaled_count_rate": float(row["scr"]),
                                "measurement": int(row["m"]),
                                "replicates": aligned})
            out[pol].append(ds)
    for pol in out:
        out[pol].sort(key=lambda d: d.metadata.get("angle_deg", 0.0))
    return out


def load_temperature_series(
    sample_dir: str,
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
    hybrid: bool = False,
    stitched: bool = False,
) -> Dict[float, Dict[str, List[DLSDataset]]]:
    """Load every temperature subdirectory of a MAPCS sample.

    Temperature folders are named like ``"22C"`` or ``"11.5C"``.  Returns an
    ordered dict ``{T_celsius: {'VH': [...], 'VV': [...], '3D': [...]}}`` (each
    value from :func:`load_mapcs_experiment`), sorted by temperature.
    """
    temps = {}
    for entry in sorted(os.listdir(sample_dir)):
        full = os.path.join(sample_dir, entry)
        if not os.path.isdir(full):
            continue
        m = re.match(r"^(-?\d+(?:\.\d+)?)\s*C$", entry.strip(), re.IGNORECASE)
        if not m:
            continue
        try:
            temps[float(m.group(1))] = load_mapcs_experiment(
                full, wavelength_nm=wavelength_nm, hybrid=hybrid, stitched=stitched)
        except Exception:
            continue
    return dict(sorted(temps.items()))


def depolarized_diffusion(
    vv: Sequence[DLSDataset],
    vh: Sequence[DLSDataset],
    temperature: Optional[float] = None,
    viscosity: Optional[float] = None,
    tau_max: Optional[float] = None,
    cumulant_order: int = 2,
) -> Dict[str, object]:
    """Separate translational and rotational diffusion from VV/VH curves.

    Polarized (VV) scattering decays with the translational rate
    ``Gamma_VV(q) = q^2 D_t`` while depolarized (VH) scattering decays with the
    rotation-enhanced rate ``Gamma_VH(q) = q^2 D_t + 6 D_r``.  Fitting each
    channel's per-angle cumulant decay rate against ``q^2`` gives the
    translational diffusion coefficient from the slopes and ``6 D_r`` from the
    VH intercept (relative to VV).  For large particles ``D_r`` (~1/R^3) is
    often below resolution, in which case the returned ``D_r`` clips to ~0 and
    ``R_rot`` is None -- the framework still validates that both channels share
    one ``D_t`` and reports an upper bound on rotation.
    """
    T = temperature if temperature is not None else (vv[0].temperature if vv else None)
    eta = viscosity if viscosity is not None else (vv[0].viscosity if vv else None)
    if T is None or eta is None:
        raise ValueError("temperature and viscosity must be provided.")

    def rates(dsets):
        q, g = [], []
        for ds in dsets:
            cum = cumulant_fit(ds.tau, ds.g2m, beta=ds.beta,
                               order=cumulant_order, tau_max=tau_max)
            if np.isfinite(cum["Gamma"]) and cum["Gamma"] > 0:
                q.append(ds.q); g.append(cum["Gamma"])
        return np.array(q), np.array(g)

    q_vv, g_vv = rates(vv)
    q_vh, g_vh = rates(vh)
    fit_vv = gamma_vs_q2(q_vv, g_vv, temperature=T, viscosity=eta)
    fit_vh = gamma_vs_q2(q_vh, g_vh, temperature=T, viscosity=eta)

    D_t = fit_vv["D"]
    six_Dr = fit_vh["intercept"] - fit_vv["intercept"]
    D_r = max(six_Dr / 6.0, 0.0)
    R_h = k_BOLTZMANN * T / (6.0 * math.pi * eta * D_t) if D_t > 0 else float("nan")
    R_rot = (k_BOLTZMANN * T / (8.0 * math.pi * eta * D_r)) ** (1.0 / 3.0) \
        if D_r > 0 else None

    # depolarization ratio from scaled count rates (intensity proxy)
    def scr_by_angle(dsets):
        return {round(d.metadata.get("angle_deg", 0.0)):
                d.metadata.get("scaled_count_rate") for d in dsets}
    scr_vv, scr_vh = scr_by_angle(vv), scr_by_angle(vh)
    depol = {a: scr_vh[a] / scr_vv[a] for a in scr_vv
             if a in scr_vh and scr_vv[a]}

    return {
        "D_t": D_t, "D_r": D_r, "six_Dr": six_Dr, "R_h": R_h, "R_rot": R_rot,
        "fit_vv": fit_vv, "fit_vh": fit_vh,
        "q_vv": q_vv, "gamma_vv": g_vv, "q_vh": q_vh, "gamma_vh": g_vh,
        "depolarization_ratio": depol,
        "depol_mean": float(np.mean(list(depol.values()))) if depol else float("nan"),
        "rotation_resolved": bool(D_r > 0 and six_Dr > 0.05 * abs(fit_vv["intercept"] + 1e-12)),
    }
