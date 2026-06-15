# Dataset Report Structure Template

This template defines the required structure for a complete LSII / MAPCS
analysis report. It covers every DLS/SLS dataset class the LSII instrument
produces: single- and multi-angle, single- and multi-temperature, polarized
(VV/VH/3D), concentration series, and time-resolved/kinetic runs. The report
must be readable at two levels:

1. A high-level summary for decision-making: executive summary text, a compact
   table, and consolidating figures.
2. A complete diagnostic record: all relevant raw data, preprocessing choices,
   fitted models, residuals, propagated errors, and analysis assumptions needed
   to reproduce or audit the result.

The default report must not hide assumptions in code. Every numerical result
that depends on a fit window, interpolation grid, regularization, calibration,
viscosity, refractive index, or model choice must state that dependency.

Static intensities are **quantitative**: the LSII export is toluene-referenced
by default, so the scaled count rate is an absolute Rayleigh ratio R(q), not a
relative count rate. Static analyses (Guinier/Porod regime, R_g, structure
factor, and — for a concentration series — M_w and A_2) are therefore reported
on an absolute scale. Any report that cannot confirm the toluene reference must
downgrade the static scale to "relative" and flag it.

Every requirement below must map to a pipeline function that emits it, or be
marked `NA — not implemented` / `manual`. A "required" section that is silently
skipped is a defect: it reads as covered when it was not.

## Report Metadata

Required fields:

| field | required content |
|---|---|
| Report title | sample name, experiment type, date or run ID |
| Data source | root data path, export format, instrument/software version if known |
| Report generator | script name, git commit or working-tree note, run timestamp |
| Analyst / owner | person or pipeline that produced the report |
| Dataset class | single-angle, depolarized, multi-angle, multi-temperature, concentration series, time-resolved/kinetic, or combination |
| Channels present | 3D modulated cross, VV, VH, other |
| Temperatures | set points and measured temperatures |
| Angles | scattering angles and computed q range |
| Replicates | count per angle/channel/temperature |
| Solvent constants | viscosity, refractive index, wavelength, source of each value |
| Sample constants | concentration, dn/dc, material assumptions, if used |
| Static scale | absolute (toluene-referenced Rayleigh ratio) by default; "relative" if reference unconfirmed |
| Particle model | assumed form-factor/shape model (sphere, coil, rod, vesicle, ...) and whether it is assumed or inferred from rho = R_g/R_h |
| Primary model | cumulant, CORENN, joint SLS/DLS, depolarized model, temperature model |
| Report limitations | known missing data, invalid metadata, suspected artifacts |

## Executive Summary

This section must be short and decision-oriented. It should answer:

- What was measured?
- What is the primary physical conclusion?
- What result is most reliable?
- What result is conditional or assumption-sensitive?
- What data-quality issue, if any, could change the conclusion?
- What follow-up measurement or reanalysis is recommended?

Required executive summary paragraphs:

1. Dataset overview: sample, channels, angle/temperature coverage, replicates.
2. Main result: size, distribution shape, aggregation, anisotropy, or transition,
   **including the shape factor rho = R_g/R_h and the particle model it implies**
   (e.g. compact sphere rho ~ 0.78, random coil rho ~ 1.5), with an explicit flag
   when the data contradict the assumed model.
3. Confidence statement: what supports the result and what limits it, including
   whether the primary R_h/PSD is bootstrap-validated and whether the fit passes
   the residual-structure and hold-out cross-validation checks.
4. Assumption-sensitive quantities: viscosity, refractive index, q calibration,
   fit windows, static intensity calibration, regularization, assumed form factor.
5. Recommended next action.

## Summary Table

Provide one compact table with one row per dataset condition. A condition is the
smallest natural grouping: for example one temperature in a multi-temperature
experiment, one angle in a single-angle temperature sweep, or one channel when
VV/VH/3D are analyzed separately.

Required columns when available:

| column | description |
|---|---|
| Condition | temperature, angle, channel, or combined label |
| N curves | number of raw correlograms and number kept |
| q or q range | q in 1/m; include angle range for multi-angle data |
| Temperature | measured K and set point C |
| Viscosity | value used, uncertainty, source |
| Refractive index | value used, uncertainty, source |
| Primary R_h | estimate as median + 68%/95% CI from replicate bootstrap, plus method |
| PDI / width | cumulant PDI or PSD width with uncertainty |
| Static result | regime-dependent: Guinier R_g (if q.R_g < 1.3 accepted) or Porod exponent p (if q.R >= pi); absolute R(q) scale |
| Shape factor rho | R_g/R_h with uncertainty, and the implied particle model |
| Particle model | assumed form factor and consistency flag (rho vs 0.78, depolarization) |
| M_w, A_2 | weight-average mass and second virial coefficient (concentration series only) |
| CORENN peaks | peak radii, intervals, and intensity/volume/number basis |
| Depolarization | I_VH/I_VV and uncertainty, if present |
| Rotational term | 6D_r with uncertainty and resolved/unresolved flag |
| Fit quality | reduced chi^2, r2, residual-structure test (runs/lag-1 autocorr), hold-out CV score |
| Data quality | rejected angles, invalid curves, saturation, intercept issues |
| Assumption flags | viscosity placeholder, q recalculated, baseline fitted, form factor assumed, etc. |

Rules:

- Report uncertainties in the same units as the values.
- Distinguish measurement uncertainty, fit uncertainty, bootstrap uncertainty,
  and systematic uncertainty where possible.
- If a value is not physically resolved, write "not resolved" or "not
  identifiable", not "nan".
- If a diagnostic is descriptive rather than generative, label it as diagnostic.

## Consolidating Figures

The report should contain a small set of high-value figures before the detailed
diagnostics. These figures should summarize the dataset without hiding raw
evidence.

Required summary figures by dataset class:

| dataset class | required consolidating figures |
|---|---|
| Single-angle, single-temperature | raw correlograms with fit overlay; residuals; PSD or cumulant result |
| Single-angle, multi-temperature | R_h(T), Gamma(T), intensity(T), viscosity assumption check |
| Multi-angle | Gamma vs q^2; apparent R_h(q); static I(q) regime diagnostic (Guinier/Porod, anticipated from PSD); PSD; rho = R_g/R_h |
| Depolarized VV/VH | Gamma_VV and Gamma_VH vs q^2; VH-VV intercept; depolarization ratio vs angle |
| Multi-temperature multi-angle | per-temperature Gamma vs q^2; R_h(T); apparent R_h(q,T); implied viscosity or shared-size check |
| 3D plus 2D channels | 2D vs 3D intercept beta; channel intensity comparison; multiple-scattering indicators |
| Concentration series | Kc/R(q,c) Zimm/Debye/Berry plot; M_w and A_2 with CI; D(c) -> D_0 and k_D; R_h(c) |
| Time-resolved / kinetic | observable vs time (R_h(t), I(t), PSD(t)); fitted rate/growth law with residuals; stationarity check |

Every consolidating figure must include:

- Axis labels with units.
- Error bars or uncertainty bands where available.
- Fit/model overlays when a fit is reported.
- Residual or goodness-of-fit panel for primary fits.
- Legend entries that state fitted slopes or parameters when fitted.
- Caption text stating what is raw, what is derived, and what assumptions enter.

## Required Analysis Assumptions

This section is mandatory. It must list every assumption built into the pipeline,
even if it seems standard.

### Instrument And Calibration

Required statements:

- Wavelength in vacuum and source.
- Refractive index used for q calculation and source.
- q calculation formula and whether exported q was trusted or recalculated.
- Scattering angle source and any rounding/grouping.
- Temperature source: set point, measured value, or fallback.
- Viscosity source and whether it is measured, literature, inferred, or a
  placeholder.
- Static intensity scale: confirm the toluene reference is present so the scaled
  count rate is an absolute Rayleigh ratio R(q); if it cannot be confirmed,
  downgrade to "relative" and flag it. State any further angular/volume
  correction, or state explicitly that none is applied.
- Whether laser-power normalization is trusted from the export.
- Whether background count rate or solvent blank was subtracted.
- Whether multiple-scattering corrections are applied.
- Whether 2D pseudo-cross and 3D modulated cross data are analyzed separately.

### Correlogram Preprocessing

Required statements:

- Raw correlation column used: g2-1, g2, normalized g2, or reconstructed.
- Replicate handling: individual fits, average curve, median curve, or pooled
  likelihood.
- Count trace usage: hardware curve only, per-repeat pooled software
  correlation, gap-aware stitched software correlation, or hybrid merge.
- If stitched count traces are used: timestamp source, gap distribution,
  whether negative gaps were clipped, valid pair counts versus lag, and
  comparison to per-repeat pooled correlation.
- Baseline treatment: fixed, fitted offset, tail estimate, or free parameter.
- Intercept beta treatment: fixed from export, fitted per curve, fitted per
  angle/channel, or marginalized.
- Lag range included in every fit: tau_min, tau_max, g1_min, maximum points.
- Early-time exclusion and afterpulsing handling.
- Late-time exclusion and slow-tail handling.
- Smoothing, interpolation, binning, or resampling applied to correlograms.
- Outlier rejection criteria for replicates or lag points.

### Cumulant Fits

Required statements:

- Cumulant order.
- Fit variable: g1, log(g1), g2-1, or other.
- Siegert transform details and beta used.
- Bounds: tau_min, tau_max, g1_min, minimum number of points.
- Weighting: unweighted, intensity weighted, Schaetzel, replicate variance, or
  other.
- Error estimate: covariance, bootstrap over replicates, residual bootstrap, or
  none.
- Criteria for accepting/rejecting Gamma, mu2, and PDI.
- Whether Gamma is used as a diagnostic only or as an input to a physical model.

### CORENN / PSD Inversion

Required statements:

- Radius grid range and units.
- Number of grid points or spline interpolation/control points.
- Basis type: direct radius grid, B-spline, P-spline, log-normal mixture, etc.
- Basis normalization.
- Regularization type and strength selection method.
- Non-negativity constraints.
- Smoothness penalty definition and order.
- Baseline/offset treatment.
- Noise model and weights.
- Fit domain: full g2, g1, linearized field correlation, or transformed data.
- Form factor included or omitted.
- Static intensity coupling included or omitted.
- Number, volume, or intensity basis for reported PSD.
- Conversion assumptions between intensity, volume, and number distributions.
- Peak detection method and minimum separation/prominence.
- Uncertainty method: bootstrap, Hessian/covariance, posterior samples, or
  sensitivity sweep.

### Static SLS / Scattering-Regime Diagnostics

The static analysis is **regime-aware and PSD-anticipated**, not a blind Guinier
fit. The CORENN multi-angle PSD gives the intensity-weighted characteristic size,
which sets the radius of gyration R_g = sqrt(3/5) R and therefore which regime
each measured q probes: Guinier (q.R_g < ~1.3, ln I linear in q^2), form factor
(q.R ~ 1-pi), or Porod (q.R >= ~pi, I ~ q^-4 for sharp interfaces). The
regime-appropriate fit is then run on the qualifying q-points.

Required statements:

- Static intensity source and scale: confirm toluene-referenced absolute Rayleigh
  ratio R(q); otherwise flag as relative.
- Whether any angular, volume, background, detector, or polarization correction
  is applied; whether a solvent blank/baseline is fitted/subtracted.
- The regime classification from the PSD, with the measured q.R_g span.
- Guinier branch (when q.R_g < 1.3 points exist): fit window, R_g and uncertainty,
  acceptance criteria (negative slope, finite R_g, enough points, q.R_g in window).
- Porod branch (when q.R >= pi points exist): power-law exponent p (I ~ q^-p) with
  uncertainty and interpretation (p ~ 4 compact/sharp interface; 1 < p < 4
  fractal/diffuse; p < ~1 near-flat/noisy, no clean slope).
- If the low-q intensity rises (no positive R_g) report it as a structure factor
  S(q) < 1 / interparticle correlation, not a particle R_g.
- If curvature is present but the physical checks fail, report it as static
  curvature, not as particle-size slopes.
- Concentration series only: Kc/R(q,c) Zimm/Debye/Berry analysis on the absolute
  R(q); report M_w, A_2 (second virial), and the static R_g from the q-dependence,
  each with uncertainty. State dn/dc, concentration, and the optical constant K.

### Particle Model And Shape (rho = R_g/R_h)

The shape factor rho = R_g/R_h is a **first-class deliverable**: it combines the
static R_g and the dynamic R_h into a model-independent shape diagnostic and is
the primary check on the form factor assumed by the PSD inversion.

Required statements:

- The particle model assumed by the PSD form factor (sphere by default) and
  whether it is assumed or inferred.
- rho = R_g/R_h with uncertainty (propagated from both R_g and R_h), and the
  R_g and R_h definitions used (which static regime supplied R_g; z-average R_h).
- Interpretation against the canonical map: compact sphere rho ~ 0.775, hollow
  sphere/vesicle rho ~ 1, random coil (theta) rho ~ 1.5, rod rho >~ 2,
  star/microgel rho < 0.775.
- Consistency flags that the assumed model is wrong: rho departs from the assumed
  model's value, measurable depolarization (optical anisotropy => non-spherical),
  or a Porod exponent inconsistent with the assumed shape. When flagged, state
  that the sphere-form-factor PSD radii are model-biased and how.

### Depolarized DLS

Required statements when VV/VH are present:

- Channel assignment rule: manifest labels, count-rate ordering, or explicit
  metadata.
- Geometry: 2D pseudo-cross or 3D modulated cross.
- Translational model: Gamma_VV = q^2 D_t.
- Depolarized model: Gamma_VH = q^2 D_t + 6 D_r, if used.
- Whether D_t is shared between VV and VH.
- Whether the rotational intercept is constrained nonnegative.
- Resolution criterion for declaring D_r resolved.
- Depolarization ratio definition and uncertainty.
- Angle matching/tolerance between VV and VH.
- Optical anisotropy interpretation limits.

### Multi-Angle Analysis

Required statements when multiple q values are present:

- Angle inclusion/exclusion criteria.
- q recalculation and uncertainty.
- Gamma vs q^2 fit model: through origin or with intercept.
- Weighting of angles.
- Error bars on Gamma and propagation to D and R_h.
- Apparent R_h(q) definition and uncertainty.
- Criteria for rejecting unresolvably slow apparent radii.
- Static/dynamic consistency checks, including rho = R_g/R_h.
- Whether multi-angle PSD is fit jointly or by aggregating single-angle fits.
- Whether beta, baseline, and static intensity scale are fitted per angle.
- Hold-out cross-validation: predict a withheld angle's correlogram (or Gamma)
  from the joint PSD / Gamma-vs-q^2 fit and report the predicted-vs-observed
  residual, so a good fit is shown to be a good prediction.
- Residual-structure test on Gamma-vs-q^2 and the PSD g2 fits (runs test or
  lag-1 autocorrelation), not r^2 alone: flag systematically structured residuals
  as model inadequacy even when r^2 is high.

### Multi-Temperature Analysis

Required statements when temperatures vary:

- Temperature source and uncertainty.
- Viscosity source at each temperature.
- Whether size is assumed temperature-invariant.
- Whether viscosity is fixed, inferred, or jointly fit.
- Whether all temperatures share one PSD or each has its own PSD.
- Whether concentration, solvent composition, or sample history changes with T.
- Heating/cooling order and equilibration time, if known.
- Hysteresis or irreversible aggregation checks.
- Error propagation from T and viscosity into D and R_h.

## Error And Uncertainty Propagation

Uncertainty must be displayed throughout the report, not only in the final table.

Required uncertainty sources:

- Replicate-to-replicate variance.
- Correlation noise as a function of lag time.
- Fit parameter covariance or bootstrap uncertainty.
- q uncertainty from angle and refractive index.
- Temperature uncertainty.
- Viscosity uncertainty.
- Refractive index uncertainty.
- Static intensity uncertainty.
- Baseline/beta uncertainty.
- Regularization or model-selection sensitivity.
- Systematic uncertainty from calibration or missing blanks.

Required propagation targets:

- Gamma and mu2.
- D = Gamma / q^2.
- R_h = kT / (6 pi eta D).
- R_g from Guinier slope.
- PSD peak positions and amplitudes.
- Population fractions.
- Depolarization ratio.
- Rotational diffusion D_r.
- Implied viscosity in same-sample multi-temperature fits.

Required standard for the primary quantities (headline R_h, the PSD/peak radii,
and rho = R_g/R_h):

```text
Replicate bootstrap is the default, not an upgrade. Resample replicates, rerun
the full fit inside each sample, recompute every derived quantity (q, D, R_h,
R_g, rho, PSD peaks, population ratios), and report median + 68% + 95% interval.
A bare point estimate is not acceptable for a headline number.
```

Minimum standard for secondary/diagnostic quantities:

```text
Report value +/- statistical uncertainty, then list systematic sensitivity.
```

For derived quantities:

- Do not propagate errors by differentiating only the final scalar if the input
  is produced by a nonlinear fit; bootstrap or sample the full fit where
  possible.
- Keep covariance between parameters when reporting ratios such as R_g/R_h.
- For upper/lower bounds, state the bound explicitly instead of quoting a
  symmetric uncertainty.

## Fit Quality And Model Adequacy

r^2 and reduced chi^2 alone do not establish that a model is adequate: a fit can
have r^2 = 0.99 with systematically structured residuals (a sign the model class
is wrong). This section makes adequacy and predictivity explicit for every
primary fit (cumulant, Gamma-vs-q^2, CORENN g2, static regime, joint-temperature).

### Residual Structure

Required:

- A residual-structure test, not just r^2/chi^2: runs test, Durbin-Watson, or
  lag-1 autocorrelation of the standardized residuals.
- Pass/flag verdict: structured (correlated) residuals are flagged as model
  inadequacy even at high r^2, with the suspected cause (wrong form factor,
  unresolved second mode, baseline drift).
- The residual plot itself for every primary fit (already required per-curve;
  here summarized across the fit).

### Predictive Cross-Validation

Required where there is more than one independent unit to hold out:

- Multi-angle: withhold one angle, fit the rest, predict the held-out angle's
  Gamma / correlogram, report the predicted-vs-observed residual; repeat
  leave-one-angle-out where practical.
- Single curve: withhold a lag-decade, fit the rest, predict the held-out decade.
- Report the cross-validated score alongside the in-sample fit quality so that
  "fits well" is demonstrated to be "predicts well" — the test that matters for a
  forward-model-primary pipeline.

### Convergence And Identifiability

Required for every nonlinear fit (CORENN inversion, cumulant, joint-temperature):

- Optimizer convergence flag and final cost / gradient norm.
- Regularization selection well-posedness: that the GCV (or L-curve) objective
  has a clear interior minimum, not a boundary/flat selection; show the curve.
- Kernel / design-matrix condition number, as a measure of how ill-posed the
  inversion is for this q/lag coverage.
- Effective degrees of freedom (trace of the smoother / influence matrix) vs the
  number of basis parameters, and the number of statistically supportable modes —
  so the PSD is not over-interpreted beyond what the data identify.

## Detailed Raw Data Sections

After the executive summary and consolidating figures, include raw and
intermediate data sections sufficient to diagnose failures.

### Data Inventory

Required tables:

- File inventory: path, channel, angle, temperature, replicate, measurement ID.
- Metadata inventory: beta, count rate, scaled count rate, duration,
  temperature, viscosity, refractive index, angle, q.
- Missing/invalid metadata table.
- Replicate count table by condition.

### Raw Correlograms

Required figures:

- All raw correlograms, grouped by condition.
- Replicate overlay for each condition.
- Mean/median curve with replicate spread.
- Early-lag zoom.
- Late-lag zoom.
- Intercept beta distribution.
- Baseline/tail behavior.

Required diagnostics:

- Curves rejected and reason.
- Curves with nonpositive values, bad intercepts, saturation, or nondecay.
- Residuals for every fitted correlogram.

### Count Rates And Static Intensities

Required figures:

- Count rate vs angle, channel, and temperature.
- Scaled count rate vs angle, channel, and temperature.
- Static I(q) and log I(q) vs q^2.
- Static residuals for accepted SLS/Guinier/structure fits.
- Blank/background curves if available.

Required diagnostics:

- Whether static intensity is usable as SLS.
- Whether angular corrections were applied.
- Whether static curvature is physical, unresolved, or an artifact candidate.

### Per-Curve Fit Results

Required tables:

- Fit bounds per curve.
- Gamma, uncertainty, mu2, PDI, beta, baseline, residual score.
- Fit acceptance/rejection reason.
- Derived D and R_h with uncertainty.

Required figures:

- Histogram of Gamma residuals.
- Gamma uncertainty vs angle/temperature.
- PDI vs angle/temperature.
- Apparent R_h vs q or T.

### Multi-Angle Diagnostics

Required when applicable:

- Gamma vs q^2 with error bars, fit, residuals.
- D_app(q) with uncertainty.
- R_h,app(q) with uncertainty.
- Static I(q) with accepted fit or curvature annotation.
- CORENN multi-angle fit overlays for representative angles.
- Per-angle beta/baseline/static-scale diagnostics.
- Contribution of each radius region to each q, if PSD is broad.

### Depolarized Diagnostics

Required when applicable:

- VV and VH correlogram overlays at matched angles.
- Gamma_VV and Gamma_VH vs q^2.
- Difference Gamma_VH - Gamma_VV vs angle.
- Intercept fit for 6D_r with uncertainty.
- Depolarization ratio vs angle with error bars.
- Channel assignment audit.
- 2D versus 3D comparison if both are available.

### Multi-Temperature Diagnostics

Required when applicable:

- Per-temperature raw correlogram panels.
- Gamma vs q^2 at each temperature.
- R_h(T) under recorded viscosity.
- R_h(T) under corrected or inferred viscosity.
- Implied viscosity(T) if same-size constraint is used.
- Static intensity vs T.
- PSD or peak radius vs T.
- Heating/cooling order, if available.
- Residuals grouped by temperature.

## Model-Specific Sections

Include only the sections relevant to the dataset, but every included section
must follow the same reporting rules.

### Single-Angle, Single-Temperature 3D DLS

Required:

1. Raw correlogram and replicate overlay.
2. Cumulant or primary decay fit with bounds and residuals.
3. CORENN or PSD inversion if used.
4. R_h and uncertainty.
5. Statement of what cannot be inferred from one angle:
   - no q-dependence;
   - limited polydispersity identifiability;
   - no static/dynamic consistency check;
   - no rotational diffusion unless depolarized data exist.

### Single-Angle, Multi-Temperature

Required:

1. All single-angle requirements at each temperature.
2. Gamma(T), D(T), R_h(T) with propagated T and viscosity uncertainty.
3. Recorded-viscosity and corrected-viscosity comparison.
4. Same-sample shared-size check if appropriate.
5. Hysteresis or irreversible-change note if measurement order is known.

### Depolarized Single-Angle

Required:

1. VV and VH raw correlograms.
2. Channel-specific Gamma fits.
3. Difference/intercept estimate for 6D_r, if identifiable.
4. Depolarization ratio and uncertainty.
5. Resolved/unresolved rotational diffusion statement.

### Multi-Angle Single-Temperature

Required:

1. Per-angle raw correlograms and fit overlays.
2. Gamma vs q^2 fit and residuals.
3. Apparent R_h(q).
4. Static I(q) / Guinier / structure diagnostic.
5. Joint multi-angle PSD or explicit reason not used.
6. Angle rejection and q-resolution limits.

### Multi-Angle Multi-Temperature

Required:

1. Multi-angle diagnostics at each temperature.
2. Consolidated Gamma vs q^2 panel across temperatures.
3. R_h(T), apparent R_h(q,T), and PSD(T).
4. Joint temperature model assumptions.
5. Shared-size / changing-size model comparison.
6. Viscosity sensitivity and inferred-viscosity diagnostics.

### Multi-Angle Depolarized Multi-Temperature

Required:

1. All multi-angle multi-temperature requirements.
2. VV/VH channel audit at each temperature.
3. D_t(T) and D_r(T), with resolved/unresolved flags.
4. Depolarization ratio vs angle and T.
5. 2D/3D multiple-scattering comparison if 3D exists.

### Concentration Series

Required when the same sample is measured at multiple concentrations:

1. All applicable single-/multi-angle requirements at each concentration.
2. Static SLS on the absolute (toluene-referenced) R(q): Kc/R(q,c) Zimm/Debye/
   Berry plot, with the optical constant K and dn/dc stated.
3. M_w and A_2 (second virial) with uncertainty from the c -> 0 and q -> 0
   extrapolations; static R_g from the q-dependence.
4. Dynamic concentration dependence: D(c) and the c -> 0 extrapolation D_0, the
   interaction parameter k_D, and R_h from D_0.
5. Dilute-limit statement: whether R_h is the self-diffusion (dilute) value or a
   collective/interdiffusion value affected by S(q) and hydrodynamic interaction.
6. Concentration source/uncertainty and whether dilution series is by mass or
   volume.

### Time-Resolved / Kinetic

Required when a condition is sampled repeatedly over time (aggregation, gelation,
dissolution, ripening):

1. The tracked observable(s) vs time: R_h(t), static I(t)/R(t), PSD(t), PDI(t).
2. Stationarity check: whether each correlogram is quasi-static over its
   acquisition (so a single g2 fit is valid) or the sample evolves within a run.
3. The fitted kinetic law (rate constant, growth exponent, induction time) with
   residuals, or an explicit "descriptive only" statement if no model is fit.
4. Acquisition order, time stamps, and total elapsed time.
5. Reversibility / hysteresis note where a return condition is measured.

## Assumption Register

Every report must include an assumption register table. This is separate from
the methods prose and should be machine-readable if possible.

| assumption | value | source | affects | sensitivity checked? | comment |
|---|---|---|---|---|---|
| wavelength | 632.8 nm | instrument default | q, D, R_h | no | example |
| refractive index | 1.33 | summary export | q | yes | replace with actual |
| viscosity | 0.89 mPa s | solvent table | R_h | yes | replace with actual |
| q source | recalculated from angle | pipeline | all q-dependent fits | no | example |
| static scale | absolute (toluene-referenced R(q)) | LSII export | SLS, R_g, M_w, A_2 | yes | confirm reference present |
| form factor | sphere (Rayleigh-Gans-Debye) | pipeline config | PSD radii, rho check | yes | flag if rho != 0.78 or depol |
| cumulant g1_min | 0.60 | pipeline config | Gamma, PDI | yes | example |
| tau_max | 1.0 s | pipeline config | Gamma, PSD | yes | example |
| PSD radius grid | 1 nm to 1 um | pipeline config | PSD peaks | yes | example |
| spline basis | 24 control points | pipeline config | PSD smoothness | yes | example |
| regularization | GCV-selected | pipeline config | PSD width | yes | example |
| baseline | fitted per curve | pipeline config | Gamma, PSD | yes | example |

## Quality Gates

A report should explicitly pass, fail, or waive each gate.

Required gates:

- Raw files and metadata parsed without silent column shifts.
- q values are physically plausible and reproducible from angle and refractive
  index.
- Temperature and viscosity are not placeholders, or placeholder status is
  flagged.
- Fit bounds are stated for every correlogram fit.
- Rejected curves/angles are listed with reasons.
- Primary fit residuals are shown, and a residual-structure test (runs/lag-1
  autocorrelation) is reported, not r^2 alone.
- The headline R_h / PSD / rho carry replicate-bootstrap median + 68%/95% CI, not
  a bare point estimate.
- Hold-out cross-validation is reported where more than one angle (or lag-decade)
  exists.
- Every nonlinear fit reports a convergence flag, and the CORENN inversion shows
  a well-posed regularization selection (interior GCV/L-curve minimum) and its
  kernel condition number.
- Static intensity scale is confirmed absolute (toluene-referenced) or flagged
  relative.
- The static result uses the PSD-anticipated regime (Guinier or Porod); a rising
  low-q intensity is reported as a structure factor, not a particle R_g.
- rho = R_g/R_h is reported and the assumed particle model is stated, with a
  consistency flag when rho or depolarization contradict it.
- CORENN/PSD result states intensity/volume/number basis and the number of
  statistically supportable modes.
- Depolarized rotational diffusion is labeled unresolved when the intercept is
  not significant.
- Multi-temperature size/viscosity ambiguity is explicitly addressed; for a
  concentration series the dilute-limit (self vs collective diffusion) is stated.
- Every figure caption states raw vs derived data.

## Reproducibility Appendix

Required:

- Command used to generate the report.
- Python environment or dependency snapshot.
- Git commit or working-tree status.
- Random seeds.
- Full pipeline configuration.
- Full list of generated files.
- Checksums or timestamps for input data if practical.

## Minimal Report Skeleton

Use this skeleton for every dataset and omit only sections that are explicitly
not applicable.

```text
# Dataset Name - Analysis Report

## Metadata
## Executive Summary
## Summary Table
## Consolidating Figures
## Key Assumptions And Quality Flags
## Data Inventory
## Raw Correlograms
## Count Rates And Static Intensities (absolute, toluene-referenced)
## Per-Curve Fit Results
## Primary Model Results
## Static Regime And Shape (Guinier/Porod, rho = R_g/R_h, particle model)
## Error Propagation And Sensitivity
## Fit Quality And Model Adequacy (residual structure, cross-validation, convergence)
## Mode-Specific Diagnostics
  - Single-angle DLS
  - Depolarized VV/VH
  - Multi-angle
  - Multi-temperature
  - Concentration series
  - Time-resolved / kinetic
## Rejected Data And Failure Diagnostics
## Assumption Register
## Reproducibility Appendix
```
