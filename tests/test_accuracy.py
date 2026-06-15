import unittest
import inspect
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reports"))
import _common as report_common
import corenn
import hybrid_correlator
import pct


TEMPERATURE_K = 298.15
VISCOSITY_PA_S = 0.00089
SOLVENT_N = 1.33
BETA = 0.78


def q_from_angle_nm(angle_deg, n_solvent=SOLVENT_N, wavelength_nm=632.8):
    theta = np.radians(angle_deg)
    return 4.0 * np.pi * n_solvent * np.sin(theta / 2.0) / wavelength_nm


def sphere_form_factor(q_nm, radii_m):
    x = q_nm * 1e9 * radii_m
    out = np.ones_like(x)
    mask = np.abs(x) > 1e-8
    xm = x[mask]
    out[mask] = (3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm**3)) ** 2
    return out


def intensity_lognormal(radii_m, median_m, sigma_ln):
    density = np.exp(-0.5 * (np.log(radii_m / median_m) / sigma_ln) ** 2)
    density = density / (radii_m * sigma_ln * np.sqrt(2.0 * np.pi))
    return density / np.trapz(density, radii_m)


def synthetic_dls_dataset(
    radii_m,
    intensity,
    q_nm,
    tau,
    beta=BETA,
    sigma=0.002,
    use_form_factor=False,
):
    diffusion = pct.k_BOLTZMANN * TEMPERATURE_K / (
        6.0 * np.pi * VISCOSITY_PA_S * radii_m
    )
    optical = intensity.copy()
    if use_form_factor:
        optical = optical * sphere_form_factor(q_nm, radii_m)
    denom = np.trapz(optical, radii_m)
    g1 = np.array(
        [
            np.trapz(
                optical * np.exp(-((q_nm * 1e9) ** 2) * diffusion * lag),
                radii_m,
            )
            / denom
            for lag in tau
        ]
    )
    return pct.CorennDataSet(
        tau=tau,
        g2=beta * g1 * g1,
        q=q_nm,
        beta=beta,
        sigma=np.full_like(tau, sigma),
    )


@unittest.skipIf(
    pct.least_squares is None or pct.nnls is None,
    "scipy.optimize is required for CORENN accuracy tests",
)
class LightScatteringAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = pct.Dataloader.Analyzer()

    def test_q_from_angle_uses_refractive_index_and_overrides_reported_q(self):
        summary = pd.DataFrame(
            {
                "Repetition": [0],
                "Scattering vector [1/nm]": [999.0],
                "Scattering angle [deg]": [90.0],
                "Solvent refractive index": [1.40],
            }
        )

        resolved = self.analyzer._resolve_scattering_summary(
            summary, repetition_label="Repetition 0"
        )

        expected = q_from_angle_nm(90.0, n_solvent=1.40)
        self.assertEqual(resolved["q_source"], "angle")
        self.assertAlmostEqual(resolved["refractive_index"], 1.40)
        self.assertAlmostEqual(resolved["q"], expected, places=12)
        self.assertNotAlmostEqual(resolved["q"], 999.0)

    def test_corenn_recovers_mono_and_lognormal_synthetic_distributions(self):
        tau = np.geomspace(8e-7, 3e-2, 90)

        mono_radius = 50e-9
        mono_q = q_from_angle_nm(90.0)
        mono_radii = np.geomspace(mono_radius / 1.001, mono_radius * 1.001, 5)
        mono_intensity = np.ones_like(mono_radii)
        mono_intensity = mono_intensity / np.trapz(mono_intensity, mono_radii)
        mono_ds = synthetic_dls_dataset(
            mono_radii,
            mono_intensity,
            mono_q,
            tau,
            beta=0.75,
            sigma=0.002,
        )

        mono_result = self.analyzer.corenn_multi_q(
            datasets=[mono_ds],
            temperature=TEMPERATURE_K,
            viscosity=VISCOSITY_PA_S,
            radius_range=(5e-9, 300e-9),
            n_basis=18,
            n_R=220,
            n_lam=20,
            lam_min=1e-9,
            lam_max=1e1,
            max_nfev=1200,
        )
        mono_mean = np.trapz(
            mono_result["R_grid"] * mono_result["I_R"], mono_result["R_grid"]
        )
        self.assertLess(abs(mono_mean / mono_radius - 1.0), 0.12)

        radii = np.geomspace(8e-9, 300e-9, 800)
        target = intensity_lognormal(radii, median_m=75e-9, sigma_ln=0.28)
        target_mean = np.trapz(radii * target, radii)
        lognormal_ds = synthetic_dls_dataset(
            radii,
            target,
            q_from_angle_nm(70.0),
            tau,
            beta=0.80,
            sigma=0.0015,
        )

        lognormal_result = self.analyzer.corenn_multi_q(
            datasets=[lognormal_ds],
            temperature=TEMPERATURE_K,
            viscosity=VISCOSITY_PA_S,
            radius_range=(8e-9, 300e-9),
            n_basis=22,
            n_R=260,
            n_lam=24,
            lam_min=1e-9,
            lam_max=1e1,
            max_nfev=1400,
        )
        recovered_mean = np.trapz(
            lognormal_result["R_grid"] * lognormal_result["I_R"],
            lognormal_result["R_grid"],
        )
        self.assertLess(abs(recovered_mean / target_mean - 1.0), 0.05)

    def test_bayesian_joint_sls_dls_map_recovers_synthetic_intensity_psd(self):
        radii = np.geomspace(10e-9, 250e-9, 600)
        target = intensity_lognormal(radii, median_m=65e-9, sigma_ln=0.22)
        target_mean = np.trapz(radii * target, radii)
        tau = np.geomspace(8e-7, 2e-2, 80)

        dls_datasets = [
            synthetic_dls_dataset(
                radii,
                target,
                q_from_angle_nm(angle),
                tau,
                beta=BETA,
                sigma=0.002,
                use_form_factor=True,
            )
            for angle in (50.0, 90.0)
        ]

        sls_q = np.array([q_from_angle_nm(a) for a in (35.0, 55.0, 75.0, 95.0, 115.0)])
        sls_intensity = np.array(
            [
                2.5 * np.trapz(target * sphere_form_factor(q_val, radii), radii) + 0.05
                for q_val in sls_q
            ]
        )

        result = self.analyzer.bayesian_joint_sls_dls_map(
            dls_datasets=dls_datasets,
            sls_q=sls_q,
            sls_intensity=sls_intensity,
            sls_sigma=np.full_like(sls_q, 0.005),
            temperature=TEMPERATURE_K,
            viscosity=VISCOSITY_PA_S,
            radius_range=(10e-9, 250e-9),
            n_basis=18,
            n_R=220,
            smoothness=1e-4,
            regularization=1e-9,
            base_distribution="intensity",
            use_form_factor=True,
            fit_sls_background=True,
            max_nfev=1200,
        )

        recovered_mean = np.trapz(
            result["R_grid"] * result["I_R"], result["R_grid"]
        )
        self.assertTrue(result["optimizer"]["success"])
        self.assertLess(abs(recovered_mean / target_mean - 1.0), 0.05)
        self.assertLess(np.max(np.abs(result["sls"]["residual"])), 5e-4)


class CorennDiagnosticTests(unittest.TestCase):
    def test_corenn_default_lower_radius_is_one_angstrom(self):
        invert_default = inspect.signature(corenn.corenn_invert).parameters[
            "radius_range"
        ].default
        multi_default = inspect.signature(corenn.corenn_multi_angle).parameters[
            "radius_range"
        ].default

        self.assertEqual(invert_default[0], 1e-10)
        self.assertEqual(multi_default[0], 1e-10)
        self.assertEqual(corenn.DEFAULT_CORENN_RADIUS_RANGE[0], 1e-10)

    def test_segmented_guinier_identifies_two_static_length_scales(self):
        q = np.linspace(3.0e6, 2.6e7, 28)
        split = 10
        rg_large = 220e-9
        rg_small = 100e-9
        slope_large = -(rg_large**2) / 3.0
        slope_small = -(rg_small**2) / 3.0
        q2 = q * q

        ln_intensity = np.empty_like(q)
        ln_intensity[:split] = np.log(12.0) + slope_large * q2[:split]
        continuity = ln_intensity[split - 1] - slope_small * q2[split - 1]
        ln_intensity[split:] = continuity + slope_small * q2[split:]

        result = corenn.segmented_guinier_fit(q, np.exp(ln_intensity), n_min=5)
        segments = result["two_segment"]["segments"]

        self.assertTrue(result["supports_two_segments"])
        self.assertGreater(result["delta_bic"], 10.0)
        self.assertLess(abs(segments[0]["R_g"] / rg_large - 1.0), 0.05)
        self.assertLess(abs(segments[1]["R_g"] / rg_small - 1.0), 0.05)

    def test_segmented_guinier_rejects_nonnegative_slope_curvature(self):
        q = np.linspace(3.0e6, 2.6e7, 28)
        q2 = q * q
        ln_intensity = np.empty_like(q)
        ln_intensity[:6] = np.log(150.0) + 1.7e-15 * q2[:6]
        ln_intensity[6:] = np.log(170.0) + 4.0e-17 * q2[6:]

        result = corenn.segmented_guinier_fit(q, np.exp(ln_intensity), n_min=5)
        segments = result["two_segment"]["segments"]

        self.assertTrue(result["bic_supports_two_segments"])
        self.assertFalse(result["supports_two_segments"])
        self.assertFalse(np.isfinite(segments[0]["R_g"]))
        self.assertFalse(np.isfinite(segments[1]["R_g"]))

    def test_default_cumulant_fit_rejects_slow_tail_artifact(self):
        tau = np.geomspace(1e-6, 200.0, 220)
        beta = 0.7
        g1 = 0.6 * np.exp(-20.0 * tau) + 0.4 * np.exp(-0.01 * tau)
        g2 = beta * g1 * g1

        default_fit = corenn.cumulant_fit(tau, g2, beta=beta, order=2)
        loose_fit = corenn.cumulant_fit(tau, g2, beta=beta, order=2, g1_min=0.30)

        self.assertGreater(default_fit["Gamma"], 10.0)
        self.assertLess(loose_fit["Gamma"], 1.0)

    def test_short_lag_residual_diagnostic_detects_fast_component(self):
        tau = np.geomspace(8e-7, 3e-2, 90)
        q = corenn.q_from_angle(90.0, SOLVENT_N)
        beta = 0.78
        d_slow = corenn.diffusion_coefficient(
            np.array([250e-9]), TEMPERATURE_K, VISCOSITY_PA_S
        )[0]
        d_fast = corenn.diffusion_coefficient(
            np.array([8e-9]), TEMPERATURE_K, VISCOSITY_PA_S
        )[0]
        sigma = np.full_like(tau, 0.0015)
        kwargs = {
            "n_basis": 12,
            "n_R": 80,
            "n_lambda": 6,
            "lambda_range": (1e-7, 1e2),
        }

        def make_dataset(fast_fraction):
            g1 = (
                (1.0 - fast_fraction) * np.exp(-q * q * d_slow * tau)
                + fast_fraction * np.exp(-q * q * d_fast * tau)
            )
            return corenn.DLSDataset(
                tau=tau,
                g2m=beta * g1 * g1,
                q=q,
                beta=beta,
                sigma=sigma,
                temperature=TEMPERATURE_K,
                viscosity=VISCOSITY_PA_S,
                n_solvent=SOLVENT_N,
                duration=30.0,
            )

        slow_only = corenn.short_lag_residual_diagnostic(
            make_dataset(0.0),
            lag_band=(8e-7, 4e-5),
            baseline_lag_min=3e-4,
            invert_kwargs=kwargs,
        )
        fast_mode = corenn.short_lag_residual_diagnostic(
            make_dataset(0.10),
            lag_band=(8e-7, 4e-5),
            baseline_lag_min=3e-4,
            invert_kwargs=kwargs,
        )

        predicted = corenn.corenn_predict_g2(
            fast_mode["baseline_result"],
            dataset=make_dataset(0.10),
            tau=fast_mode["baseline_result"].tau,
        )
        self.assertTrue(np.allclose(predicted, fast_mode["baseline_result"].g2_model))
        self.assertGreater(
            fast_mode["rms_residual_over_beta"],
            3.5 * slow_only["rms_residual_over_beta"],
        )
        self.assertGreater(fast_mode["rms_z_log"], 20.0)

    def test_corenn_sigma_inflation_downweights_wiggle_band(self):
        tau = np.geomspace(1e-7, 2e-2, 70)
        q = corenn.q_from_angle(90.0, SOLVENT_N)
        beta = 0.78
        radius = 55e-9
        diffusion = corenn.diffusion_coefficient(
            np.array([radius]), TEMPERATURE_K, VISCOSITY_PA_S
        )[0]
        g1 = np.exp(-q * q * diffusion * tau)
        sigma = np.full_like(tau, 0.0015)
        ds = corenn.DLSDataset(
            tau=tau,
            g2m=beta * g1 * g1,
            q=q,
            beta=beta,
            sigma=sigma,
            temperature=TEMPERATURE_K,
            viscosity=VISCOSITY_PA_S,
            n_solvent=SOLVENT_N,
            duration=30.0,
        )
        lag_band = (8e-7, 4e-5)
        systematic_sigma = 0.02
        result = corenn.corenn_invert(
            ds,
            radius_range=(1e-9, 300e-9),
            n_basis=10,
            n_R=70,
            n_lambda=4,
            lambda_range=(1e-7, 1e-2),
            sigma_inflation={
                "lag_band": lag_band,
                "systematic_sigma": systematic_sigma,
            },
        )

        mask = (result.tau >= lag_band[0]) & (result.tau <= lag_band[1])
        expected = np.sqrt(sigma[mask] ** 2 + systematic_sigma ** 2)
        self.assertTrue(np.allclose(result.extra["sigma"][mask], expected))
        self.assertLess(np.max(result.extra["weights"][mask]), 1.0 / 0.02)
        self.assertEqual(
            result.extra["sigma_inflation"]["n_inflated"],
            int(np.count_nonzero(mask)),
        )


class ReportStaticPowerLawTests(unittest.TestCase):
    def test_powerlaw_selector_excludes_high_q_flattening_upturn(self):
        q = np.geomspace(4.6e6, 2.55e7, 14)
        p_true = 2.45
        intensity = 600.0 * (q / q[0]) ** (-p_true)
        intensity[-3:] = intensity[-4] * np.array([0.95, 0.97, 1.35])
        sigma = 0.015 * intensity

        fit = report_common.select_static_powerlaw_window(
            q,
            intensity,
            intensity_err=sigma,
            n_min=5,
            r2_min=0.998,
            chi2_red_max=10.0,
            max_abs_z=4.0,
        )

        self.assertTrue(fit["accepted"])
        self.assertEqual(fit["n_points"], 11)
        self.assertLess(abs(fit["exponent"] / p_true - 1.0), 0.02)
        self.assertEqual(fit["high_q"]["n_points"], 3)
        self.assertIn("upturn", fit["high_q"]["behavior"])


class ReportCorennBoundsTests(unittest.TestCase):
    def test_psd_upper_bound_uses_dynamic_observability_when_limiting(self):
        tau = np.geomspace(1e-6, 1e-2, 20)
        datasets = [
            corenn.DLSDataset(tau=tau, g2m=np.exp(-tau), q=3.0e6),
            corenn.DLSDataset(tau=tau, g2m=np.exp(-tau), q=5.0e6),
        ]
        z_min = -np.log(0.80)

        bounds = report_common.corenn_psd_radius_bounds(
            datasets,
            TEMPERATURE_K,
            1.0e-3,
            lag_max=1.0e-2,
            min_field_decay=z_min,
            static_qr_factor=np.pi,
        )

        expected_dynamic = (
            (3.0e6) ** 2
            * corenn.k_BOLTZMANN
            * TEMPERATURE_K
            * 1.0e-2
            / (6.0 * np.pi * 1.0e-3 * z_min)
        )
        self.assertEqual(bounds["limiting"], "dynamic")
        self.assertAlmostEqual(bounds["R_max"], expected_dynamic)
        self.assertLess(bounds["dynamic_upper"], bounds["static_upper"])


class HybridCorrelationTests(unittest.TestCase):
    def test_stitched_count_traces_preserve_gaps(self):
        t = np.arange(4.0)
        start = datetime(2025, 1, 1, 12, 0, 0)
        traces = [
            {
                "time": t,
                "cr_a": np.full(t.size, 100.0),
                "cr_b": np.full(t.size, 101.0),
                "start_datetime": start,
                "duration_s": 4.0,
            },
            {
                "time": t,
                "cr_a": np.full(t.size, 100.0),
                "cr_b": np.full(t.size, 101.0),
                "start_datetime": start + timedelta(seconds=6),
                "duration_s": 4.0,
            },
        ]

        stitched = hybrid_correlator.stitch_count_traces(traces)
        self.assertEqual(stitched["segments"][1]["gap_s"], 2.0)
        self.assertTrue(np.all(np.isnan(stitched["cr_a"][4:6])))

        tau, g2m, n_pairs, _ = hybrid_correlator.correlate_stitched_traces(
            traces, max_frac=0.8, min_pairs=1
        )
        self.assertGreater(tau.size, 0)
        self.assertTrue(np.all(n_pairs > 0))
        self.assertTrue(np.allclose(g2m, 0.0))

        tau_cut, _, _, _ = hybrid_correlator.correlate_stitched_traces(
            traces, max_frac=0.8, min_pairs=1, lag_min=2.0
        )
        self.assertTrue(np.all(tau_cut >= 2.0))


if __name__ == "__main__":
    unittest.main()
