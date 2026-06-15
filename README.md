# MAPCS
Multi-angle photon correlation spectroscopy reduction, analysis and reconstruction

## Modules

- **`pct.py`** — the main data loader (`Dataloader`) and `Analyzer` for LS
  Instruments DLS/SLS exports (count traces, correlation functions, summaries),
  with wavelet, cumulant, S(q), and CORENN-style analysis.
- **`corenn.py`** — a standalone, tested re-implementation of the CORENN
  particle-size-distribution inversion, plus extensions:
  - non-linear weighted least squares on the un-linearized Siegert model
    (does *not* take `sqrt((g2-1)/beta)`, so the noise is not distorted),
  - cubic B-spline / P-spline discretization on a log-radius grid,
    Tikhonov curvature penalty, GCV-selected regularization, non-negativity,
    and a *free baseline*;
  - replicate-calibrated Schätzel noise weighting, bootstrap/linearized
    uncertainty bands, lag-window control, and a multi-angle joint inversion
    that folds the angular dependence through the sphere form factor `P(qR)`.
  - multi-angle tools: `cumulant_fit` (per-curve decay rate / PDI over the
    initial decay) and `gamma_vs_q2` (Γ-vs-q² → z-average D and R_h); the joint
    `corenn_multi_angle` fixes each angle's intercept β_q and lets the
    form-factor normalization follow from a single I₀(R), so all angles are
    coupled (validated to recover a synthetic number-PSD center).
  - static / structure tools: `guinier_fit` (single-slope R_g),
    `segmented_guinier_fit` (single-slope first; a two-population Guinier
    interpretation is accepted only when BIC improves and both segment slopes
    are physical negative Guinier slopes),
    `apparent_diffusion_analysis` (D_app(q)=Γ/q² → z-average R_h,
    polydispersity coefficient C, low/high-q span, structure diagnostics), and
    `static_dynamic_correlation` (compares an accepted static Guinier split, if
    present, with the R_h,app(q) crossover; otherwise it reports static
    curvature and dynamic q-dependence separately; `demo_static_dynamic.py`).
  - polarized tools: `load_mapcs_experiment` (loads a MAPCS experiment and
    labels each measurement VH / VV / 3D-mod-cross from the manifest),
    `load_temperature_series` (all temperature folders of a sample), and
    `depolarized_diffusion` (separates translational D_t from rotational D_r via
    Γ_VV = q²D_t and Γ_VH = q²D_t + 6D_r; validated to recover a synthetic
    rotational coefficient to <0.1%).
  - multi-temperature: `joint_temperature_fit` assumes one (T-invariant) sample
    and either validates that R_h is constant across T or, anchored to a
    reference T, *solves* the unknown viscosity η(T) from the requirement of a
    shared size (useful when the recorded viscosity is a fixed placeholder).
  - Note on Γ(q): the first cumulant is a diagnostic *projection*, not a
    generative model — it cannot resolve the number of populations or a
    continuous distribution. The actual size distribution lives in the full
    decay `g2(q,τ)` and is recovered by `corenn_invert` (per angle) or
    `corenn_multi_angle` (joint), which the reports treat as the primary result.
  - Entry points: `corenn_invert(dataset, ...)`, `corenn_multi_angle([...])`,
    `cumulant_fit(...)`, `gamma_vs_q2(...)`,
    `static_dynamic_correlation(...)`, `load_mapcs_experiment(dir)`,
    `depolarized_diffusion(vv, vh, ...)`,
    `build_dataset_from_repetition(dir)`, `load_measurement_replicates(dir)`.
- **`hybrid_correlator.py`** — extends the usable lag range by
  software-correlating the photon count-rate trace (symmetric-normalization
  A×B cross-correlation) across replicates and **gracefully merging** it with
  the hardware correlogram via inverse-variance weighting, so the hardware
  dominates the main decay and the software supplies the slow dynamics.
  It also provides a gap-aware stitched-trace diagnostic
  (`correlate_stitched_traces`) that uses repetition start timestamps and NaN
  gaps to test whether sequential repeats can be treated as one longer
  elapsed-time experiment.
  Opt in per call with `load_measurement_replicates(dir, hybrid=True)` (the
  default is the hardware curve).  Best for slow-mode / single-curve analysis;
  not recommended for the cross-angle `corenn_multi_angle` (the per-angle merge
  adds angle-dependent count-trace bias).
  Entry point: `build_hybrid_dataset(measurement_dir)` → a `corenn.DLSDataset`
  spanning the full lag range, ready for `corenn.corenn_invert`.

## Reports

See **[`reports/`](reports/README.md)** for reproducible PDF reports: a
synthetic capabilities validation (`reports/CAPABILITIES.pdf`) and a
walk-through report for each dataset (`reports/LiTFSI.pdf`, `SISV.pdf`,
`JM_F0.pdf`) running every available analysis on the real data.

## Demo

- `demo_corenn_hybrid.py` / `corenn_hybrid_demo.ipynb` — synthetic ground-truth
  validation, the hybrid merge on real data, and recovered PSDs
  (hardware-only vs hybrid). Run: `python demo_corenn_hybrid.py [measurement_dir]`.
- `demo_multiangle.py` — angle-resolved analysis on the 28-angle LiTFSI data:
  Γ-vs-q² diffusion fit, apparent-R_h-vs-q aggregate diagnostic, and joint
  multi-angle vs single-angle PSD. Run: `python demo_multiangle.py [temperature_dir]`.
- `demo_polarized.py` — depolarized (VV/VH) analysis on the JM_F0 MAPCS data:
  validates rotational-diffusion extraction on synthetic anisotropic particles,
  then shows the real translational-only result and the depolarization ratio.
  Run: `python demo_polarized.py [JM_F0_temperature_dir]`.
- `demo_guinier.py` — LiTFSI 22C & 50C: accepted two-population Guinier static
  diagnostic and initial-cumulant apparent R_h-vs-q polydispersity/structure
  analysis.
- `demo_joint_temperature.py` — LiTFSI 22C & 50C joint same-sample dynamic
  cross-check; uses initial-cumulant Gamma values and solves the implied 50C
  viscosity when anchoring the shared dynamic population to 22C.
- `demo_kinetic_sisv.py` — SISV_1% across 7 temperatures: size/PDI/intensity and
  VV/VH polarization vs T, with a same-sample viscosity cross-check.

Note: the `data/` directory is git-ignored.

## Theory

- `docs/theory/dls_theory.tex` — field/intensity correlation and Siegert basics.
- `docs/theory/extended_fitting_theory.tex` — generalizes and relaxes CORENN's
  assumptions: multi-angle form-factor weighting, polarized (VV/VH) inversion
  with rotational diffusion, the hybrid correlation merge, and the noise model.
