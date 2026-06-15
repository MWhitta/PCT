# Analysis Guidance

This repo is moving in the right direction: the nonlinear CORENN benchmark and
joint SLS/DLS forward model are defensible foundations. The current demos,
however, still mix physical inference with heuristic diagnostics. Keep those
diagnostics, but do not let them become the final model.

## What Is Currently Heuristic

- `cumulant_fit()` uses explicit fit-window choices such as `g1_min=0.60` and
  demo-level `tau_max` values. These prevent late-tail artifacts from corrupting
  the initial decay rate, but they are still window heuristics.
- Single-slope `Gamma` vs `q^2` fits assume one effective diffusion coefficient.
  The LiTFSI data show `D_app(q)=Gamma/q^2` increasing with `q`, so a single
  slope is only an operational summary.
- `segmented_guinier_fit()` is single-slope first. A two-population Guinier
  interpretation is accepted only when BIC improves and both segment slopes are
  physical negative Guinier slopes; otherwise it is only a curvature diagnostic.
- Low/high-q `R_h,app` medians, spans, and Spearman trends are robust descriptive
  summaries. They are not generative population estimates.
- The joint-temperature viscosity solve is a cross-check. It assumes the same
  dynamic population across temperature and should be interpreted only after
  q-dependent population weighting has been modeled.

## Physical Grounding

Make the forward model primary. The data should be explained by one shared
population model:

```text
I(q) = A * integral N(R) R^6 P(q,R) S(q,R,...) dR + background

g1(q,t) = integral N(R) R^6 P(q,R) S(q,R,...) exp[-q^2 D(R)t] dR
          / integral N(R) R^6 P(q,R) S(q,R,...) dR

g2(q,t)-1 = beta_q * |g1(q,t)|^2 + offset_q
```

Use `Gamma(q)`, apparent `R_h(q)`, and Guinier slopes/curvature as diagnostic
projections of this model, not as the model itself.

## Conditional Two-Population Treatment

Do not default to two Guinier slopes. The LiTFSI static data pass the physical
two-slope checks, but SISV and JM_F0 do not. When an accepted two-slope Guinier
and the dynamic upward curvature in `Gamma` vs `q^2` are both present, they are
consistent with at least two optically weighted length scales:

- low q: large, slow population dominates;
- high q: large population rolls off through form factor / Guinier weighting,
  so smaller, faster population contributes more strongly.

A reusable two-population model should fit something like:

```text
N(R) = a1 LogNormal(R; mu1, sigma1) + a2 LogNormal(R; mu2, sigma2)
```

or use a nonparametric PSD with posterior peak detection. The same population
model should explain:

- segmented static intensity,
- q-dependent apparent `R_h`,
- full DLS curves,
- temperature rescaling.

## Microsecond-Lag Wiggles And Fast Modes

Treat short-lag wiggles as a model-check signal before assigning them to a
population. The defensible diagnostic is:

1. Fit the slower population with nonlinear CORENN outside the microsecond band.
2. Extrapolate that same forward model back into the excluded band.
3. Quantify the residual with replicate uncertainties: log-lag RMS, signed
   log-lag mean, peak-to-peak amplitude, max absolute z-score, and matched
   residual-template amplitude.
4. Plot those metrics versus angle, q, q^2, and polarization.

This separates three cases:

- fixed lag shape and fixed amplitude across angle/polarization: likely
  correlator or detector artifact;
- residual amplitude/rate changing with q and surviving cross-correlation:
  sample-related fast dynamics become plausible;
- residual strongest in VH relative to VV: rotational or anisotropic optical
  dynamics may contribute.

A smaller translational population should not merely make a wiggle at a fixed
lag. Its decay rate should scale approximately as `Gamma_fast = q^2 D_fast`.
For VH, test whether an additional q-independent intercept is needed:
`Gamma_VH = q^2 D_t + 6 D_r`. Report a fast-population radius only after this
q-scaling check or after the joint SLS/DLS forward model improves predictive
residuals without creating an unconstrained small-radius tail.

For CORENN, start by preserving the affected lags but inflating their
uncertainty:

```python
diag = short_lag_residual_diagnostic(dataset, replicates=reps)
sigma_inflation = sigma_inflation_from_diagnostic(diag, component="wiggle")
result = corenn_invert(dataset, replicates=reps, sigma_inflation=sigma_inflation)
```

This uses the detrended microsecond residual as a systematic error term,
`sigma_eff^2 = sigma^2 + sigma_wiggle^2`, so the optimizer does not convert
non-Laplace wiggle structure into spurious PSD mass.

## Recommended Architecture

Separate reusable physical inference from exploratory demos:

- `data.py`: LSII loaders, units, q calibration, replicate handling.
- `models.py`: sphere form factor, optional structure factor, diffusion, SLS/DLS
  forward models.
- `inference.py`: nonlinear CORENN, Bayesian MAP/posterior fits, mixture-model
  inference.
- `diagnostics.py`: cumulants, segmented Guinier, apparent `R_h(q)`, model
  checks.
- `demo_*.py`: thin scripts that call the reusable APIs and produce figures.

## Configuration And Sensitivity

Do not bury analysis choices in demos. Promote these to explicit fit configs:

- `g1_min`;
- `tau_max`;
- q-range inclusion/exclusion;
- baseline and offset treatment;
- beta treatment;
- segmented-Guinier BIC threshold;
- viscosity assumptions;
- form-factor and structure-factor choices.

Each demo should print or save these settings. Important claims should include
a small sensitivity check over the relevant choices.

## Model Checks

Every physical fit should produce posterior or predictive checks:

- static `I(q)` overlay and residuals;
- all measured `g2(q,t)` curves with fitted curves;
- derived `Gamma(q)` from the fitted model vs measured cumulants;
- apparent `R_h(q)` from fitted model vs diagnostic estimates;
- inferred low/high-q population contributions;
- uncertainty intervals on PSD peaks, population fractions, and temperature
  viscosity estimates.

## Practical Rule

Use diagnostics to decide what model class is needed. Use the joint physical
forward model to report physical quantities.

For this repo, that means:

1. Keep nonlinear PCT-CORENN as the benchmark.
2. Build the Bayesian joint SLS/DLS model on the same forward model and data
   loaders.
3. Treat segmented Guinier, `Gamma/q^2`, apparent `R_h(q)`, and temperature
   viscosity solves as model checks unless they are explicitly embedded in the
   joint forward model.
