# Analysis reports

Reproducible, self-contained reports. Each generator script (re)computes every
quantity from the raw data, regenerates its figure under `figures/`, and writes
both a `.md` and a `.pdf`. Run from the repo root with the `pct` environment.

Use **[REPORT_TEMPLATE.md](REPORT_TEMPLATE.md)** as the required structure for
new dataset reports. It defines the executive summary, summary tables,
consolidating figures, raw-data appendices, error propagation, and assumption
register that each report must contain.

## Capabilities (synthetic ground-truth validation)

- **[CAPABILITIES.pdf](CAPABILITIES.pdf)** — every analysis function validated
  on synthetic data with known truth, in one montage figure + table.
  Generator: `python reports/capabilities_report.py`.

## Per-dataset reports (all available analyses run on the real data)

- **[LiTFSI.pdf](LiTFSI.pdf)** — 20 molal electrolyte, 22 °C & 50 °C, 28 angles
  (3D cross-correlation). Cumulants, Γ-vs-q², apparent R_h(q), a regime-aware
  static I(q) diagnostic (Guinier/Porod, anticipated from the PSD),
  static↔dynamic correlation, CORENN multi-angle PSD, and the same-sample
  viscosity solve.
  `python reports/litfsi_report.py`.
- **[SISV.pdf](SISV.pdf)** — polymer in THF/DOX, 7 temperatures, VH/VV/3D. Full
  static+dynamic suite on the VV channel at every temperature, plus the
  temperature series (size/PDI/intensity), depolarized VV/VH analysis, and the
  same-sample viscosity check. `python reports/sisv_report.py`.
- **[JM_F0.pdf](JM_F0.pdf)** — 25 °C, VH/VV/3D, 28 angles. Full static+dynamic
  suite on VV, depolarized rotational analysis, depolarization ratio, and the
  2D-vs-3D coherence comparison. `python reports/jm_f0_report.py`.
- **[SWy3.pdf](SWy3.pdf)** — SWy-3 Wyoming montmorillonite clay (1 %) in water,
  22 °C, VH/VV/3D, 14 angles. Full static+dynamic suite on VV; the valid Porod
  slope (q^−2.3) identifies a mass-fractal / branched clay floc (so the PSD
  sphere form factor is flagged as an effective approximation), plus the
  depolarized VV/VH and 2D-vs-3D analyses. `python reports/swy3_report.py`.

## Principle (see `../guidance.md`)

The **forward model is primary**: the size distribution is the CORENN inversion
of the full `g2(q,τ)`. The cumulant `Γ(q)`, apparent `R_h(q)`, and the static
`I(q)` regime (Guinier vs Porod, anticipated from the PSD) are **diagnostic
projections** of that model — useful to decide what model class is needed, not
the model itself.
