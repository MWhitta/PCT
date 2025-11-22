import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

try:
    from scipy.interpolate import BSpline
    from scipy.special import gammaln
except ImportError:  # pragma: no cover
    BSpline = None
    gammaln = None

k_BOLTZMANN = 1.380649e-23  # J/K


@dataclass
class CorennDataSet:
    """Container describing a single DLS dataset for CORENN inversion."""

    tau: np.ndarray
    g2: np.ndarray
    q: float
    beta: float = 1.0
    sigma: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, object]] = None


def _build_radius_grid(r_min: float, r_max: float, n_R: int) -> np.ndarray:
    if r_min <= 0 or r_max <= 0:
        raise ValueError("Radius bounds must be positive.")
    return np.geomspace(r_min, r_max, num=n_R)


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Trapezoid weights require a 1D grid with at least two points.")
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    if x.size > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w


def _build_bspline_basis(
    r_grid: np.ndarray,
    n_basis: int,
    degree: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if BSpline is None:
        raise ImportError("scipy is required for B-spline basis construction.")

    r_grid = np.asarray(r_grid, dtype=float)
    if n_basis < degree + 1:
        raise ValueError("n_basis must be at least degree + 1.")

    knot_min, knot_max = r_grid[0], r_grid[-1]
    interior = np.linspace(knot_min, knot_max, n_basis - degree + 1)
    knots = np.concatenate(
        [np.full(degree, knot_min), interior, np.full(degree, knot_max)]
    )

    Phi = []
    Phi_dd = []
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spline = BSpline(knots, coeffs, degree, extrapolate=False)
        Phi.append(np.nan_to_num(spline(r_grid), nan=0.0))
        Phi_dd.append(np.nan_to_num(spline(r_grid, nu=2), nan=0.0))

    return np.column_stack(Phi), np.column_stack(Phi_dd), knots


def _build_kernel_for_gamma(
    tau: np.ndarray,
    gamma_values: np.ndarray,
    weighted_basis: np.ndarray,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    rates = np.exp(-np.outer(tau, gamma_values))
    return rates @ weighted_basis


class Data:
    """Object for holding data, metadata, and analysis."""

    def __init__(self):
        """Initialise containers for paths, file extension, and sample data."""
        self.path = None
        self.ext = None
        # dictionary keyed by sample name. Each sample contains
        #   - experiments: dict keyed by experiment name
        #       - info: dataframe of experiment information
        #       - measurements: dict keyed by measurement number
        self.samples = {}



class Dataloader():
    """Load PCT data arranged under sample/experiment/measurement/repetition."""

    REPETITION_FILE_LOADERS = {
        "Count Trace.csv": {"skiprows": 2},
        "Correlation Function.csv": {"skiprows": 2},
        "CORENN Gamma Results.csv": {"skiprows": 5},
    }
    SCATTERING_VECTOR_COLUMN = "Scattering vector [1/nm]"
    SCATTERING_VECTOR_CONVERSION_ENABLED = True
    SCATTERING_VECTOR_SOURCE_UNITS = "1/m"

    def __init__(self, path, auto_clean=True):
        """Initialise loader state and optionally clean a single CSV file.

        Parameters
        ----------
        path : str
            Root directory of the dataset to traverse.
        auto_clean : bool, optional
            If ``True`` and ``path`` points directly to a CSV file, the file is
            cleaned immediately (default ``True``).
        """
        self.path = path
        self.directory_tree = None

        if auto_clean and os.path.isfile(path):
            self.remove_trailing_comma(path)


    def remove_trailing_comma(self, filepath):
        """Remove a dangling comma from the last line of a CSV file in-place.

        The acquisition software sometimes appends a trailing comma to the final
        record which causes pandas to introduce an extra empty column.
        Stripping the dangling comma keeps the file well-formed without having
        to load the entire file into memory.

        Parameters
        ----------
        filepath : str
            Path to the CSV file that should be normalised.

        Returns
        -------
        bool
            True if the file was modified, False otherwise.

        Raises
        ------
        FileNotFoundError
            If ``filepath`` does not point to a file on disk.
        ValueError
            If ``filepath`` is empty.
        """
        if not filepath:
            raise ValueError("A valid file path is required.")

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        with open(filepath, "rb+") as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            if file_size == 0:
                return False

            pointer = file_size - 1
            # Skip newline characters at the end of the file.
            while pointer >= 0:
                file.seek(pointer)
                char = file.read(1)
                if char not in (b"\n", b"\r"):
                    break
                pointer -= 1

            if pointer < 0:
                return False

            last_char_index = pointer
            pointer -= 1
            # Locate the start of the final line.
            while pointer >= 0:
                file.seek(pointer)
                if file.read(1) == b"\n":
                    pointer += 1
                    break
                pointer -= 1
            else:
                pointer = 0

            line_start = pointer
            file.seek(line_start)
            raw_line = file.read(last_char_index - line_start + 1).decode(
                "utf-8", errors="replace"
            )

            if not raw_line.endswith(","):
                return False

            cleaned_line = raw_line[:-1] + "\n"
            file.seek(line_start)
            file.write(cleaned_line.encode("utf-8"))
            file.truncate(line_start + len(cleaned_line))

            return True


    def map_directory_tree(self, metadata_ext=".csv"):
        """Build and cache a nested representation of the dataset file tree.

        The dataset is expected to be organised as:
            dataset/
                sample/
                    experiment/
                        <experiment metadata>.csv
                        Measurement */
                            Repetition */
                                *.csv

        The generated mapping is cached on ``self.directory_tree`` for reuse.

        Parameters
        ----------
        metadata_ext : str, optional
            Extension used to identify experiment-level metadata files.

        Returns
        -------
        dict
            Nested dictionaries describing the available samples, experiments,
            measurements (with any direct CSV files) and repetitions, including
            the concrete file paths.

        Raises
        ------
        NotADirectoryError
            If ``self.path`` does not exist or is not a directory.
        ValueError
            If a measurement folder does not contain repetition folders, or a
            repetition folder does not contain CSV files.
        """
        if not os.path.isdir(self.path):
            raise NotADirectoryError(
                f"Cannot map data directory because '{self.path}' is not valid."
            )

        metadata_ext = metadata_ext.lower()
        tree = {}

        for sample in sorted(os.listdir(self.path)):
            if sample.startswith("."):
                continue

            sample_path = os.path.join(self.path, sample)
            if not os.path.isdir(sample_path):
                continue

            experiments = {}
            for experiment in sorted(os.listdir(sample_path)):
                if experiment.startswith("."):
                    continue

                exp_path = os.path.join(sample_path, experiment)
                if not os.path.isdir(exp_path):
                    continue

                info_file = None
                for entry in sorted(os.listdir(exp_path)):
                    entry_path = os.path.join(exp_path, entry)
                    if os.path.isfile(entry_path) and entry.lower().endswith(
                        metadata_ext
                    ):
                        info_file = entry_path
                        break

                measurements = {}
                for measurement in sorted(os.listdir(exp_path)):
                    if measurement.startswith("."):
                        continue

                    meas_path = os.path.join(exp_path, measurement)
                    if not os.path.isdir(meas_path):
                        continue

                    measurement_files = {}
                    repetitions = {}
                    for entry in sorted(os.listdir(meas_path)):
                        if entry.startswith("."):
                            continue

                        entry_path = os.path.join(meas_path, entry)
                        if os.path.isfile(entry_path):
                            if entry.lower().endswith(".csv"):
                                measurement_files[entry] = entry_path
                            continue

                        rep_path = entry_path
                        if not entry.lower().startswith("repetition"):
                            raise ValueError(
                                f"Measurement folders may only contain 'Repetition*' "
                                f"directories (found {rep_path})."
                            )

                        files = {}
                        for filename in sorted(os.listdir(rep_path)):
                            if filename.startswith("."):
                                continue

                            file_path = os.path.join(rep_path, filename)
                            if os.path.isdir(file_path):
                                raise ValueError(
                                    f"Unexpected directory '{file_path}' found in repetition "
                                    "folder; only CSV files are supported."
                                )

                            if filename.lower().endswith(".csv"):
                                files[filename] = file_path

                        if not files:
                            raise ValueError(
                                f"No CSV files found inside repetition folder {rep_path}."
                            )

                        repetitions[entry] = files

                    if not repetitions:
                        raise ValueError(
                            f"No 'Repetition' folders found inside measurement {meas_path}."
                        )

                    measurements[measurement] = {
                        "files": measurement_files,
                        "repetitions": repetitions,
                    }

                experiments[experiment] = {
                    "info_file": info_file,
                    "measurements": measurements,
                }

            tree[sample] = {"experiments": experiments}

        self.directory_tree = tree
        return tree


    def _load_experiment_info(self, info_file):
        """Return the experiment-level metadata DataFrame if available.

        Parameters
        ----------
        info_file : str or None
            Path to the experiment metadata CSV file.

        Returns
        -------
        pandas.DataFrame or None
            Normalised metadata if parsing succeeded, otherwise ``None``.
        """
        if not info_file:
            return None

        try:
            return pd.read_csv(
                info_file, skiprows=3, usecols=["Name", "Correlation Type"]
            )
        except Exception:
            return None


    def _load_csv_file(self, filepath, skiprows=0, delimiter=","):
        """Load a CSV file from disk after applying trailing-comma cleanup.

        Parameters
        ----------
        filepath : str
            Location of the CSV file on disk.
        skiprows : int, optional
            Number of initial rows to skip before parsing (default ``0``).
        delimiter : str, optional
            Field separator passed to ``pandas.read_csv`` (default ``,``).

        Returns
        -------
        pandas.DataFrame or None
            Parsed dataframe if successful, otherwise ``None``.
        """
        try:
            self.remove_trailing_comma(filepath)
        except Exception:
            pass

        try:
            df = pd.read_csv(filepath, skiprows=skiprows, delimiter=delimiter)
            if (
                self.SCATTERING_VECTOR_CONVERSION_ENABLED
                and self.SCATTERING_VECTOR_COLUMN in df.columns
            ):
                col = pd.to_numeric(
                    df[self.SCATTERING_VECTOR_COLUMN], errors="coerce"
                )
                mask = col.notna()
                df.loc[mask, self.SCATTERING_VECTOR_COLUMN] = col[mask] / 1e9
            return df
        except Exception:
            return None


    def _load_repetition_file(self, filename, filepath, delimiter=","):
        """Load a single CSV file inside a repetition folder.

        Parameters
        ----------
        filename : str
            Name of the CSV file relative to the repetition directory.
        filepath : str
            Absolute path to the CSV file.
        delimiter : str, optional
            Delimiter passed to ``pandas.read_csv`` (default ``,``).

        Returns
        -------
        pandas.DataFrame or None
            Loaded data frame, or ``None`` when parsing fails.
        """
        skiprows = self.REPETITION_FILE_LOADERS.get(filename, {}).get("skiprows", 0)
        return self._load_csv_file(filepath, skiprows=skiprows, delimiter=delimiter)


    def load(self, ext='.csv', delimiter=','):
        """Load a dataset organised in a sample/experiment/measurement/repetition tree.

        Reuses a cached directory tree when available to skip redundant walks.

        Parameters
        ----------
        ext : str, optional
            File extension used for experiment level meta data (default '.csv').

        Returns
        -------
        Data
            Structured data object containing experiment information,
            measurement summaries and repetition data.
        """

        data = Data()
        data.path = self.path
        data.ext = ext

        if self.directory_tree is None:
            directory_tree = self.map_directory_tree(metadata_ext=ext)
        else:
            directory_tree = self.directory_tree

        for sample, sample_node in directory_tree.items():
            sample_dict = {"experiments": {}}

            for experiment, experiment_node in sample_node["experiments"].items():
                experiment_dict = {
                    "info": self._load_experiment_info(experiment_node["info_file"]),
                    "measurements": {},
                }

                for measurement, measurement_node in experiment_node["measurements"].items():
                    match = re.search(r"(\d+)", measurement)
                    meas_num = int(match.group(1)) if match else measurement
                    measurement_dict = {"summary": {}, "repetitions": {}}

                    measurement_files = measurement_node.get("files", {})
                    repetitions = measurement_node.get("repetitions", {})

                    for filename, filepath in measurement_files.items():
                        df = self._load_csv_file(filepath, delimiter=delimiter)
                        if df is not None:
                            measurement_dict["summary"][filename] = df

                    for repetition, files in repetitions.items():
                        rep_data = {}

                        for filename, filepath in files.items():
                            df = self._load_repetition_file(
                                filename, filepath, delimiter=delimiter
                            )
                            if df is None:
                                continue
                            rep_data[filename] = df

                        if rep_data:
                            measurement_dict["repetitions"][repetition] = rep_data

                    experiment_dict["measurements"][meas_num] = measurement_dict

                sample_dict["experiments"][experiment] = experiment_dict

            data.samples[sample] = sample_dict

        return data
    

    class Analyzer():
        """Placeholder for analysis routines operating on ``Data`` objects."""

        CONCATENATED_REPETITION_NAME = "Concatenated"
        
        def __init__(self):
            """Initialise internal state for derived analysis helpers."""

        @staticmethod
        def _coerce_numeric(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            try:
                return float(value)
            except Exception:
                try:
                    coerced = pd.to_numeric(value, errors="coerce")
                    if hasattr(coerced, "item"):
                        coerced = coerced.item()
                    return None if pd.isna(coerced) else float(coerced)
                except Exception:
                    return None
            
        
        def sq(
            self,
            data,
            sample,
            experiment,
            count_trace_file="Count Trace.csv",
            summary_file="Summary.csv",
            intensity_col="Laser intensity [mW]",
            q_col="Scattering vector [1/nm]",
            scattering_angle_col="Scattering angle [deg]",
            x0=None,
        ):
            """
            Compute scattering intensity vs scattering vector (S(q)) for an experiment.

            Steps:
                1) Load all count traces for an experiment.
                2) Compute average counts per repetition.
                3) Scale count traces by laser intensity from ``Summary.csv``.
                4) Record the scattering vector for each measurement.
                5) Average repetition means per measurement with propagated error.
                6) Return scaled intensity means, their errors, and q-values.

            Parameters
            ----------
            data : Data
                Loaded dataset structure from ``Dataloader.load``.
            sample : str
                Sample key under ``data.samples``.
            experiment : str
                Experiment key under ``data.samples[sample]['experiments']``.
            count_trace_file : str, optional
                Filename of the count trace CSV within each repetition (default
                ``"Count Trace.csv"``).
            summary_file : str, optional
                Measurement-level summary CSV used for laser intensity and q
                (default ``"Summary.csv"``).
            intensity_col : str, optional
                Column name containing laser intensity values (default
                ``"Laser intensity [mW]"``).
            q_col : str, optional
                Column name containing scattering vectors (default
                ``"Scattering vector [1/nm]"``).
            scattering_angle_col : str, optional
                Column containing scattering angles in degrees. Used to compute
                the scattering vector when available.
            x0 : float or None, optional
                Baseline shift applied before fitting the Gamma distribution.
                When ``None`` the shift is optimised (default ``None``).

            Returns
            -------
            tuple of numpy.ndarray
                (scaled_intensity, scaled_intensity_err, scattering_vector, nb_scales, nb_scale_errs, cv_means, cv_stds)
                where each entry has length equal to the number of measurements
                successfully processed.
            """

            def _coerce_scalar_numeric(value):
                """Return numeric scalar if convertible, otherwise None."""
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return None
                try:
                    return float(value)
                except Exception:
                    try:
                        coerced = pd.to_numeric(value, errors="coerce")
                        if hasattr(coerced, "item"):
                            coerced = coerced.item()
                        return None if pd.isna(coerced) else float(coerced)
                    except Exception:
                        return None

            def _match_repetition_row(summary_df, repetition_label):
                """Return matching row for a given repetition label, or None."""
                if summary_df is None or summary_df.empty:
                    return None

                if "Repetition" in summary_df.columns:
                    rep_token = repetition_label
                    match = re.search(r"(\d+)", repetition_label)
                    if match:
                        rep_token = match.group(1)

                    def _matches(value):
                        if pd.isna(value):
                            return False
                        value_str = str(value)
                        return value_str == repetition_label or value_str == rep_token

                    rows = summary_df[summary_df["Repetition"].apply(_matches)]
                    if not rows.empty:
                        return rows.iloc[0]

                return None

            def _extract_count_values(count_df):
                """Return flattened count values excluding obvious time columns."""
                if count_df is None or count_df.empty:
                    return None

                numeric_cols = [
                    c
                    for c in count_df.columns
                    if pd.api.types.is_numeric_dtype(count_df[c])
                ]
                if not numeric_cols:
                    return None

                count_cols = [
                    c for c in numeric_cols if "time" not in c.lower()
                ] or numeric_cols

                values = count_df[count_cols].to_numpy(dtype=float).ravel()
                values = values[np.isfinite(values)]
                return values if values.size else None

            sample_node = data.samples.get(sample, {})
            experiment_node = sample_node.get("experiments", {}).get(experiment, {})
            measurements = experiment_node.get("measurements", {})

            scaled_intensity = []
            scaled_intensity_err = []
            scattering_vector = []
            nb_scales = []
            nb_scale_errs = []
            cv_means = []
            cv_stds = []

            for meas_key in sorted(measurements):
                measurement = measurements[meas_key]
                repetitions = measurement.get("repetitions", {})
                summary_map = measurement.get("summary", {})
                summary_df = summary_map.get(summary_file)

                rep_scaled_means = []
                rep_q_values = []
                rep_nb_scales = []
                rep_cvs = []

                for repetition, files in repetitions.items():
                    count_df = files.get(count_trace_file)
                    count_values = _extract_count_values(count_df)
                    if count_values is None:
                        continue

                    try:
                        location = float(np.min(count_values))
                        shifted_counts = count_values - location
                        shifted_counts = shifted_counts[shifted_counts >= 0]
                        nb_fit = self._fit_negative_binomial_from_values(
                            shifted_counts
                        )
                        rep_nb_scale = nb_fit.get("size")
                    except ValueError:
                        rep_nb_scale = None

                    raw_mean = float(np.mean(count_values))
                    raw_var = float(np.var(count_values, ddof=1)) if count_values.size > 1 else 0.0
                    if not np.isfinite(raw_mean) or raw_mean <= 0:
                        continue
                    cv = np.sqrt(raw_var) / raw_mean if raw_var > 0 else 0.0
                    rep_cvs.append(cv)

                    rep_summary_df = files.get(summary_file)
                    summary_source = rep_summary_df if rep_summary_df is not None else summary_df
                    summary_row = _match_repetition_row(summary_source, repetition)
                    laser_intensity = None
                    q_value = None
                    scattering_angle = None

                    if summary_row is not None:
                        if intensity_col in summary_row:
                            laser_intensity = _coerce_scalar_numeric(
                                summary_row.get(intensity_col)
                            )
                        if q_col in summary_row:
                            q_value = _coerce_scalar_numeric(summary_row.get(q_col))
                        if scattering_angle_col in summary_row:
                            scattering_angle = _coerce_scalar_numeric(
                                summary_row.get(scattering_angle_col)
                            )

                    if laser_intensity is None and summary_source is not None:
                        laser_series = summary_source.get(intensity_col)
                        if hasattr(laser_series, "dropna"):
                            laser_series = pd.to_numeric(
                                laser_series, errors="coerce"
                            ).dropna()
                            laser_intensity = (
                                float(laser_series.iloc[0])
                                if not laser_series.empty
                                else None
                            )

                    if scattering_angle is not None:
                        theta = np.radians(scattering_angle)
                        q_value = float(
                            (4 * np.pi / self._default_wavelength())
                            * np.sin(theta / 2)
                        )
                    elif q_value is None and summary_source is not None:
                        if q_col in summary_source.columns:
                            q_series = pd.to_numeric(
                                summary_source[q_col], errors="coerce"
                            ).dropna()
                            if not q_series.empty:
                                q_value = float(q_series.iloc[0])
                        else:
                            # Soft fallback: search for a column containing "scattering vector" or "q"
                            lower_cols = {c.lower(): c for c in summary_source.columns}
                            fallback_col = None
                            for key, orig in lower_cols.items():
                                if "scattering vector" in key or key == "q":
                                    fallback_col = orig
                                    break
                            if fallback_col:
                                q_series = pd.to_numeric(
                                    summary_source[fallback_col], errors="coerce"
                                ).dropna()
                                if not q_series.empty:
                                    q_value = float(q_series.iloc[0])

                    if laser_intensity in (None, 0):
                        laser_intensity = 1.0

                    scaled_mean = raw_mean / laser_intensity
                    rep_scaled_means.append(scaled_mean)
                    rep_q_values.append(q_value)
                    if rep_nb_scale is not None:
                        rep_nb_scales.append(rep_nb_scale)

                if not rep_scaled_means:
                    continue

                rep_means_arr = np.array(rep_scaled_means, dtype=float)
                mean_intensity = float(np.mean(rep_means_arr))
                err_intensity = float(
                    np.std(rep_means_arr, ddof=1) / np.sqrt(len(rep_means_arr))
                    if len(rep_means_arr) > 1
                    else 0.0
                )

                q_value = None
                for candidate in rep_q_values:
                    numeric_candidate = _coerce_scalar_numeric(candidate)
                    if numeric_candidate is not None:
                        q_value = numeric_candidate
                        break

                scaled_intensity.append(mean_intensity)
                scaled_intensity_err.append(err_intensity)
                scattering_vector.append(q_value)
                if rep_nb_scales:
                    nb_scales.append(float(np.mean(rep_nb_scales)))
                    nb_scale_errs.append(
                        float(np.std(rep_nb_scales, ddof=1) / np.sqrt(len(rep_nb_scales)))
                        if len(rep_nb_scales) > 1
                        else 0.0
                    )
                else:
                    nb_scales.append(np.nan)
                    nb_scale_errs.append(np.nan)

                if rep_cvs:
                    cv_means.append(float(np.mean(rep_cvs)))
                    cv_stds.append(
                        float(np.std(rep_cvs, ddof=1))
                        if len(rep_cvs) > 1
                        else 0.0
                    )
                else:
                    cv_means.append(np.nan)
                    cv_stds.append(np.nan)

            return (
                np.array(scaled_intensity, dtype=float),
                np.array(scaled_intensity_err, dtype=float),
                np.array(scattering_vector, dtype=float),
                np.array(nb_scales, dtype=float),
                np.array(nb_scale_errs, dtype=float),
                np.array(cv_means, dtype=float),
                np.array(cv_stds, dtype=float),
            )
            
        
        def trace(
            self,
            data,
            sample,
            experiment,
            measurement=None,
            count_trace_file="Count Trace.csv",
            wavelet="morl",
            scales=None,
            time_column=None,
            signal_column=None,
        ):
            """
            Perform wavelet analysis on each repetition's count trace using PyWavelets.

            Parameters
            ----------
            data : Data
                Loaded dataset object from ``Dataloader.load``.
            sample : str
                Sample identifier within ``data.samples``.
            experiment : str
                Experiment identifier under the chosen sample.
            measurement : int or str or None, optional
                Specific measurement identifier to analyze. When ``None``, all
                available measurements are processed.
            count_trace_file : str, optional
                Repetition-level CSV that stores the count trace signal
                (default ``"Count Trace.csv"``).
            wavelet : str or pywt.Wavelet, optional
                Wavelet used for the continuous transform (default ``"morl"``).
            scales : array-like or None, optional
                Custom scales passed to ``pywt.cwt``. When ``None``, a geometric
                progression is generated based on the signal length.
            time_column : str or None, optional
                Column representing time; used to estimate the sampling period.
                If ``None``, the first column containing "time" (case-insensitive)
                is used when available.
            signal_column : str or list[str] or None, optional
                Column(s) to treat as signals. When ``None``, columns
                containing "count", "cr ch", or "intensity" are considered and
                all matches are processed.

            Returns
            -------
            dict
                Nested dictionary keyed by measurement number then repetition name
                containing the input signal, computed ``pywt.cwt`` coefficients,
                frequencies, scales used, and sampling period.
            """

            def _select_signal_columns(df):
                if df is None or df.empty:
                    return []

                if signal_column:
                    requested = (
                        signal_column
                        if isinstance(signal_column, (list, tuple, set))
                        else [signal_column]
                    )
                    cols = [c for c in requested if c in df.columns]
                    if cols:
                        return cols

                numeric_cols = [
                    c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                ]
                preferred = [
                    c
                    for c in numeric_cols
                    if any(keyword in c.lower() for keyword in ["count", "cr ch", "intensity"])
                ]

                if preferred:
                    return preferred

                return numeric_cols[:1]

            def _sampling_period(df):
                col_name = None
                if time_column and time_column in df.columns:
                    col_name = time_column
                else:
                    candidates = [
                        c for c in df.columns if "time" in c.lower()
                    ]
                    col_name = candidates[0] if candidates else None

                if not col_name:
                    return 1.0

                time_values = pd.to_numeric(df[col_name], errors="coerce").dropna()
                if len(time_values) < 2:
                    return 1.0

                diffs = np.diff(time_values)
                diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
                return float(np.median(diffs)) if diffs.size else 1.0

            def _default_scales(length):
                if scales is not None:
                    return np.asarray(scales, dtype=float)

                max_scale = max(2, min(length // 2, 256))
                return np.linspace(1, max_scale, num=max_scale, endpoint=True)

            sample_node = data.samples.get(sample, {})
            experiment_node = sample_node.get("experiments", {}).get(experiment, {})
            measurements = experiment_node.get("measurements", {})

            results = {}

            if measurement is not None:
                lookup_key = measurement
                available_keys = set(measurements.keys())
                if isinstance(measurement, str) and measurement not in available_keys:
                    try:
                        lookup_key = int(re.search(r"(\d+)", measurement).group(1))
                    except Exception:
                        pass
                measurements_iterable = [
                    key
                    for key in available_keys
                    if key == lookup_key or str(key) == str(lookup_key)
                ]
                measurements_iterable = measurements_iterable or [
                    key for key in available_keys if str(key) == str(measurement)
                ]
                if not measurements_iterable:
                    return {}
            else:
                measurements_iterable = sorted(measurements)

            for meas_key in measurements_iterable:
                measurement = measurements[meas_key]
                repetitions = measurement.get("repetitions", {})
                measurement_results = {}

                for repetition, files in repetitions.items():
                    trace_df = files.get(count_trace_file)
                    if trace_df is None or trace_df.empty:
                        continue

                    columns = _select_signal_columns(trace_df)
                    if not columns:
                        continue

                    channel_results = {}
                    for column_name in columns:
                        signal_series = pd.to_numeric(
                            trace_df[column_name], errors="coerce"
                        ).dropna()
                        if signal_series.empty:
                            continue

                        signal = signal_series.to_numpy(dtype=float)
                        if signal.size < 4:
                            continue

                        # Count traces are already binned, so use the recorded sampling period directly.
                        sampling_period = _sampling_period(trace_df)
                        scale_arr = _default_scales(signal.size)

                        coefficients, frequencies = pywt.cwt(
                            signal, scale_arr, wavelet, sampling_period=sampling_period
                        )

                        channel_results[column_name] = {
                            "signal_column": column_name,
                            "signal": signal,
                            "sampling_period": sampling_period,
                            "scales": scale_arr,
                            "wavelet": wavelet,
                            "coefficients": coefficients,
                            "frequencies": frequencies,
                        }

                    if channel_results:
                        measurement_results[repetition] = channel_results

                if measurement_results:
                    results[meas_key] = measurement_results

            return results
        

        def trace_lags(
            self,
            wavelet_results,
            measurement=None,
            repetition=None,
            signal_column=None,
            cone_width=3.0,
            noise_factor=3.0,
            min_peak_distance=6,
            concatenate=False,
        ):
            """
            Estimate characteristic lag times using wavelet power spectra.

            Parameters
            ----------
            wavelet_results : dict
                Output from ``trace`` containing CWT coefficients per signal.
            measurement : int or str or None, optional
                Restrict analysis to a specific measurement identifier.
            repetition : str or list[str] or None, optional
                Limit the analysis to selected repetition folders.
            signal_column : str or list[str] or None, optional
                Restrict analysis to the provided signal column(s).
            cone_width : float, optional
                Multiple of scale used to mask the cone of influence near the
                signal edges (default ``3.0``).
            noise_factor : float, optional
                Multiplicative factor applied to the noise floor when selecting
                peaks in the averaged wavelet power (default ``3.0``).
            min_peak_distance : int, optional
                Minimum index spacing between peak candidates in scale space
                (default ``6``).
            concatenate : bool, optional
                When ``True``, concatenate all repetitions for a measurement and
                recompute the wavelet transform prior to lag estimation.

            Returns
            -------
            dict
                Nested dictionary mirroring ``wavelet_results`` with additional
                lag estimates per signal channel. Each lag array is reported in
                seconds (``lags_seconds``) alongside metadata such as peak scales.
            """

            def _resolve_measurement_keys():
                if measurement is None:
                    return list(wavelet_results.keys())
                try:
                    return [
                        self._resolve_measurement_key(
                            wavelet_results, measurement
                        )
                    ]
                except KeyError:
                    return []

            def _repetition_allowed(rep):
                if repetition is None:
                    return True
                if isinstance(repetition, (list, tuple, set)):
                    return any(_rep_equals(rep, r) for r in repetition)
                return _rep_equals(rep, repetition)

            def _rep_equals(actual, requested):
                if actual == requested or str(actual) == str(requested):
                    return True
                try:
                    actual_num = int(re.search(r"(\d+)", str(actual)).group(1))
                    req_num = int(re.search(r"(\d+)", str(requested)).group(1))
                    return actual_num == req_num
                except Exception:
                    return False

            def _signal_allowed(column):
                if signal_column is None:
                    return True
                if isinstance(signal_column, (list, tuple, set)):
                    return column in signal_column
                return column == signal_column

            def _detect_peaks(series, min_height, min_distance):
                peaks = []
                last_idx = -min_distance
                for idx in range(1, len(series) - 1):
                    if series[idx] < min_height:
                        continue
                    if series[idx] <= series[idx - 1] or series[idx] < series[idx + 1]:
                        continue
                    if idx - last_idx < min_distance:
                        continue
                    peaks.append(idx)
                    last_idx = idx
                return np.array(peaks, dtype=int)

            measurement_keys = _resolve_measurement_keys()
            if not measurement_keys:
                return {}

            lag_results = {}

            for meas_key in measurement_keys:
                measurement_data = wavelet_results.get(meas_key)
                if not measurement_data:
                    continue

                if concatenate:
                    measurement_data = self._concatenate_measurement_wavelets(
                        measurement_data
                    )
                    if not measurement_data:
                        continue

                measurement_output = {}

                for rep_name, channels in measurement_data.items():
                    if not _repetition_allowed(rep_name):
                        continue

                    channel_output = {}
                    for column_name, entry in channels.items():
                        if not _signal_allowed(column_name):
                            continue

                        coefficients = entry["coefficients"]
                        scales = np.asarray(entry["scales"], dtype=float)
                        sampling_period = float(entry.get("sampling_period", 1.0))
                        if coefficients.size == 0 or scales.size == 0:
                            continue

                        power = np.abs(coefficients) ** 2
                        n_time = power.shape[1]
                        if cone_width > 0:
                            time_idx = np.arange(n_time)
                            margins = np.minimum(
                                np.ceil(cone_width * scales).astype(int), n_time // 2
                            )
                            start = margins
                            end = n_time - margins
                            valid_mask = (time_idx >= start[:, None]) & (
                                time_idx < end[:, None]
                            )
                        else:
                            valid_mask = np.ones_like(power, dtype=bool)

                        masked_power = np.where(valid_mask, power, 0.0)
                        counts = valid_mask.sum(axis=1)
                        avg_power = np.divide(
                            masked_power.sum(axis=1),
                            counts,
                            out=np.zeros_like(scales, dtype=float),
                            where=counts > 0,
                        )

                        if not np.any(np.isfinite(avg_power)):
                            continue

                        window = max(1, min(10, len(avg_power) // 4))
                        floor_candidates = avg_power[:window]
                        floor_candidates = floor_candidates[floor_candidates > 0]
                        if floor_candidates.size == 0:
                            floor_candidates = avg_power[avg_power > 0]
                        noise_floor = (
                            np.median(floor_candidates)
                            if floor_candidates.size
                            else np.max(avg_power)
                        )
                        threshold = noise_factor * noise_floor if noise_floor > 0 else np.max(avg_power)

                        peaks = _detect_peaks(avg_power, threshold, max(1, min_peak_distance))
                        if peaks.size == 0:
                            channel_output[column_name] = {
                                "lags_seconds": np.array([]),
                                "peak_scales": np.array([]),
                                "average_power": avg_power,
                                "scales": scales,
                                "sampling_period": sampling_period,
                                "wavelet": entry.get("wavelet", "morl"),
                            }
                            continue

                        freq_arr = entry.get("frequencies")
                        if freq_arr is None or len(freq_arr) != len(scales):
                            freq_arr = pywt.scale2frequency(
                                entry["wavelet"], scales
                            ) / sampling_period
                            entry["frequencies"] = freq_arr
                        pseudo_freq = np.asarray(freq_arr)[peaks]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            lags = np.where(
                                pseudo_freq > 0, 1.0 / pseudo_freq, np.nan
                            )

                        channel_output[column_name] = {
                            "lags_seconds": lags,
                            "lag_units": "s",
                            "peak_scales": scales[peaks],
                            "average_power": avg_power,
                            "scales": scales,
                            "sampling_period": sampling_period,
                            "wavelet": entry.get("wavelet", "morl"),
                        }

                    if channel_output:
                        measurement_output[rep_name] = channel_output

                if measurement_output:
                    lag_results[meas_key] = measurement_output

            return lag_results

        def wavelet_lag_constraints(
            self,
            lag_results,
            measurement,
            signal_column,
            repetition=None,
            min_ratio=1.0,
            drop_ratio=0.5,
        ):
            """
            Filter wavelet-derived lag times and estimate uncertainties.

            Parameters
            ----------
            lag_results : dict
                Output from ``trace_lags``.
            measurement : int or str
                Measurement identifier to inspect.
            signal_column : str
                Channel name (e.g., ``"CR CHA [kHz]"``).
            repetition : str or int, optional
                Repetition identifier. Required if multiple repetitions exist.
            min_ratio : float, optional
                Require lag > ``min_ratio * sampling_period`` (default ``1.0``).
            drop_ratio : float, optional
                Power ratio used for half-width estimation (default ``0.5``).

            Returns
            -------
            list of dict
                Each dict has keys ``lag``, ``uncertainty``, ``scale``, and
                ``sampling_period``.
            """

            meas_key = self._resolve_measurement_key(lag_results, measurement)
            measurement_data = lag_results.get(meas_key, {})
            if not measurement_data:
                raise KeyError(f"No lag data for measurement '{measurement}'.")

            if repetition is None:
                rep_key = next(iter(measurement_data.keys()))
            else:
                rep_key = self._resolve_repetition_name(measurement_data, repetition)

            channels = measurement_data.get(rep_key, {})
            if signal_column not in channels:
                raise KeyError(
                    f"Signal '{signal_column}' not found for measurement '{measurement}', repetition '{rep_key}'."
                )

            entry = channels[signal_column]
            lags = np.asarray(entry.get("lags_seconds", []), dtype=float)
            peak_scales = np.asarray(entry.get("peak_scales", []), dtype=float)
            sampling_period = float(entry.get("sampling_period", 1.0))

            constraints = []
            for idx, lag in enumerate(lags):
                if not np.isfinite(lag):
                    continue
                if lag <= min_ratio * sampling_period:
                    continue

                peak_scale = peak_scales[idx] if idx < len(peak_scales) else None
                uncertainty = self._estimate_lag_uncertainty(
                    entry, peak_scale, drop_ratio
                )
                constraints.append(
                    {
                        "lag": float(lag),
                        "uncertainty": uncertainty,
                        "scale": float(peak_scale) if peak_scale is not None else np.nan,
                        "sampling_period": sampling_period,
                    }
                )

            return constraints

        def _default_wavelength(self):
            """Return the default probe wavelength in nm (customise as needed)."""
            return 632.8  # nm (e.g., HeNe laser)

        def _real_space_length(self, q):
            q = np.asarray(q, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                length = 2 * np.pi / q
            return length

        def _inverse_real_space_length(self, length):
            length = np.asarray(length, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = 2 * np.pi / length
            return q
        
        def fit_negbinomial_trace(
            self,
            data=None,
            sample=None,
            experiment=None,
            measurement=None,
            repetition=None,
            count_trace_file="Count Trace.csv",
            signal_column="CR CHA [kHz]",
            count_bins=50,
            max_count=None,
            values=None,
            wavelets=None,
        ):
            """
            Fit a negative binomial distribution to a count trace.

            Parameters
            ----------
            data : Data, optional
                Loaded dataset structure from ``Dataloader.load``. Required unless
                ``values`` or ``wavelets`` are provided.
            sample : str, optional
                Sample key under ``data.samples``.
            experiment : str, optional
                Experiment key under ``data.samples[sample]['experiments']``.
            measurement : int or str, optional
                Measurement identifier. Required unless ``values`` is provided.
            repetition : str or int, optional
                Repetition identifier (e.g., ``"Repetition 0"``).
            count_bins : int, optional
                Number of bins used for the histogram overlay (default ``50``).
            max_count : int or None, optional
                Optional cap on counts when computing the histogram.
            values : array-like or None, optional
                Direct count values to fit. If provided, ``data`` and ``wavelets`` are ignored.
            wavelets : dict or None, optional
                Output from ``trace`` containing repetition-level signals.

            Returns
            -------
            dict
                Dictionary containing fitted negative-binomial parameters, the raw samples,
                an optional histogram, and the evaluated PMF.
            """
            if (
                wavelets is None
                and data is not None
                and not hasattr(data, "samples")
                and isinstance(data, dict)
            ):
                # Backwards compatibility: user passed ``wavelets`` as first argument.
                wavelets = data
                data = None

            if values is not None:
                samples = np.asarray(values, dtype=float)
            elif wavelets is not None:
                meas_key = self._resolve_measurement_key(wavelets, measurement)
                measurement_data = wavelets[meas_key]
                rep_key = self._resolve_repetition_name(measurement_data, repetition)
                repetition_data = measurement_data.get(rep_key)
                if not repetition_data or signal_column not in repetition_data:
                    raise KeyError(
                        f"Signal '{signal_column}' not found for measurement '{measurement}', "
                        f"repetition '{rep_key}'."
                    )
                signal = repetition_data[signal_column].get("signal")
                if signal is None or len(signal) == 0:
                    raise ValueError("No signal data found for the specified trace.")
                samples = np.asarray(signal, dtype=float)
            elif data is not None:
                sample_node = data.samples.get(sample, {})
                experiment_node = sample_node.get("experiments", {}).get(experiment, {})
                measurements = experiment_node.get("measurements", {})
                measurement_node = measurements.get(measurement)
                if measurement_node is None:
                    raise KeyError(f"Measurement '{measurement}' not found.")
                repetitions = measurement_node.get("repetitions", {})
                rep_key = self._resolve_repetition_name(repetitions, repetition)
                rep_files = repetitions.get(rep_key, {})
                count_df = rep_files.get(count_trace_file)
                if count_df is None or count_df.empty:
                    raise ValueError("Count trace file not found for the specified repetition.")

                values_arr = []
                for column in count_df.columns:
                    if "time" in column.lower():
                        continue
                    if pd.api.types.is_numeric_dtype(count_df[column]):
                        values_arr.append(count_df[column].to_numpy(dtype=float))
                if not values_arr:
                    raise ValueError("No numeric count columns available for NB fitting.")
                samples = np.concatenate(values_arr)
            else:
                raise ValueError("Provide either 'values', 'wavelets', or 'data'.")

            counts = np.asarray(samples, dtype=float)
            location = float(np.min(counts))
            shifted = np.round(counts - location).astype(int)
            shifted = shifted[shifted >= 0]
            params = self._fit_negative_binomial_from_values(shifted)

            hist_data = None
            if count_bins and count_bins > 1:
                count_bins = max(int(count_bins), 2)
                vmin, vmax = counts.min(), counts.max()
                if np.isclose(vmin, vmax):
                    vmin = max(0.0, vmin - 0.5)
                    vmax = vmax + 0.5
                edges = np.linspace(vmin, vmax, count_bins + 1)
                density, edges = np.histogram(counts, bins=edges, density=True)
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                hist_data = {
                    "bin_centers": bin_centers,
                    "density": density,
                    "edges": edges,
                }

            k_min, k_max = int(shifted.min()), int(shifted.max())
            k_values = np.arange(k_min, k_max + 1)
            pmf = self._negative_binomial_pmf(
                k_values, params["size"], params["prob"], params["mean"]
            )

            return {
                "mean": params["mean"],
                "variance": params["variance"],
                "size": params["size"],
                "prob": params["prob"],
                "counts": counts,
                "histogram": hist_data,
                "location": location,
                "nb": {
                    "k_values": k_values + location,
                    "pmf": pmf,
                },
            }

        def plot_negbinomial_fit(
            self,
            wavelets,
            measurement,
            repetition,
            signal_column="CR CHA [kHz]",
            fit_results=None,
            bins=50,
            ax=None,
        ):
            """
            Plot the negative binomial fit alongside the observed count distribution.

            Parameters
            ----------
            wavelets : dict
                Output from ``trace`` containing repetition-level signals.
            measurement : int or str
                Measurement identifier under the provided wavelet data.
            repetition : str or int
                Repetition identifier (e.g., ``"Repetition 0"``).
            signal_column : str, optional
                Column name to analyse (default ``"CR CHA [kHz]"``).
        fit_results : dict or None, optional
            Pre-computed results from ``fit_negbinomial_trace``. When ``None``,
            the method calls ``fit_negbinomial_trace`` internally.
            bins : int, optional
                Number of bins used for the histogram overlay when refitting
                (default ``50``).
            ax : matplotlib.axes.Axes, optional
                Axes to draw on. Creates a new figure when ``None``.

            Returns
            -------
            matplotlib.axes.Axes
                Axes containing the histogram and negative-binomial PMF overlay.
            """

            meas_key = self._resolve_measurement_key(wavelets, measurement)
            measurement_data = wavelets[meas_key]
            rep_key = self._resolve_repetition_name(measurement_data, repetition)

            if fit_results is None:
                fit_results = self.fit_negbinomial_trace(
                    wavelets,
                    measurement=measurement,
                    repetition=rep_key,
                    signal_column=signal_column,
                    count_bins=bins,
                )

            counts = fit_results.get("counts", [])
            if counts is None or len(counts) == 0:
                raise ValueError("Fit results do not contain count data.")

            hist = fit_results.get("histogram")
            if hist is None:
                # Generate histogram for plotting if not already available.
                vmin, vmax = np.min(counts), np.max(counts)
                if np.isclose(vmin, vmax):
                    vmin = max(0.0, vmin - 0.5)
                    vmax = vmax + 0.5
                edges = np.linspace(vmin, vmax, max(int(bins), 2) + 1)
                density, edges = np.histogram(counts, bins=edges, density=True)
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
            else:
                density = hist["density"]
                edges = hist["edges"]
                bin_centers = hist["bin_centers"]

            nb_data = fit_results.get("nb")
            if nb_data is None:
                raise ValueError("Fit results do not contain negative binomial PMF data.")

            if ax is None:
                _, ax = plt.subplots()

            width = edges[1] - edges[0] if len(edges) > 1 else 1.0
            ax.bar(
                bin_centers,
                density,
                width=0.9 * width,
                alpha=0.4,
                label="Observed density",
            )
            ax.plot(
                nb_data["k_values"],
                nb_data["pmf"],
                color="C1",
                linewidth=2,
                label="Negative binomial fit",
            )

            ax.set_xlabel(signal_column)
            ax.set_ylabel("Density")
            ax.set_title(
                f"Negative binomial fit for {signal_column} - {rep_key} (Measurement {measurement})"
            )
            ax.legend()

            return ax
       
        def _estimate_correlation_uncertainty(
            self,
            measurement_node,
            target_repetition,
            tau,
            g2,
            delay_col,
            value_col,
            file_name="Correlation Function.csv",
        ):
            """Estimate correlation-function uncertainty emphasizing sampling noise."""

            if (
                measurement_node is None
                or tau is None
                or g2 is None
                or len(tau) == 0
                or len(g2) == 0
            ):
                return None

            tau = np.asarray(tau, dtype=float)
            g2 = np.asarray(g2, dtype=float)
            if tau.size == 0 or g2.size == 0:
                return None

            n_points = tau.size
            lag_indices = np.arange(n_points, dtype=float)
            effective_samples = np.maximum(n_points - lag_indices, 1.0)
            relative = (lag_indices + 1.0) / effective_samples
            sampling_sigma = np.abs(g2) * relative
            sampling_sigma = np.where(np.isfinite(sampling_sigma), sampling_sigma, 0.0)

            combined = sampling_sigma
            combined = np.where(np.isfinite(combined), combined, 0.0)
            finite_abs = np.abs(g2[np.isfinite(g2)])
            base = float(np.max(finite_abs)) if finite_abs.size else 1.0
            floor = max(base * 1e-6, 1e-12)
            combined = np.where(combined > 0, combined, floor)
            return combined

        def _estimate_gamma_from_trace(self, tau, g2, beta, tau_max=None):
            tau = np.asarray(tau, dtype=float)
            g2 = np.asarray(g2, dtype=float)
            mask = np.isfinite(tau) & np.isfinite(g2)
            if tau_max is not None:
                mask &= tau <= float(tau_max)
            signal = np.asarray(g2, dtype=float)
            mask &= signal > 0
            if np.count_nonzero(mask) < 2:
                return None
            tau_sel = tau[mask]
            signal_sel = signal[mask]
            beta_val = beta if beta not in (None, 0) else 1.0
            norm = np.clip(signal_sel / max(beta_val, 1e-12), 1e-12, None)
            log_term = np.log(norm)
            slope, _ = np.polyfit(tau_sel, log_term, 1)
            gamma = -0.5 * slope
            return gamma if np.isfinite(gamma) and gamma > 0 else None

        def _prepare_corenn_datasets(
            self,
            data,
            sample,
            experiment,
            measurements,
            correlation_file,
            summary_file,
            beta_col,
            q_col,
            scattering_angle_col,
        ):
            sample_node = data.samples.get(sample, {})
            experiment_node = sample_node.get("experiments", {}).get(experiment, {})
            measurement_nodes = experiment_node.get("measurements", {})
            if not measurement_nodes:
                raise KeyError(f"No measurements found for experiment '{experiment}'.")

            if measurements is None:
                measurement_keys = sorted(measurement_nodes)
            else:
                measurement_keys = []
                for requested in measurements:
                    key = self._resolve_measurement_key(measurement_nodes, requested)
                    if key not in measurement_keys:
                        measurement_keys.append(key)

            datasets = []
            temperature_values = []
            viscosity_values = []

            def _extract_q(summary_df):
                if summary_df is None:
                    return None
                if q_col in summary_df.columns:
                    series = pd.to_numeric(summary_df[q_col], errors="coerce").dropna()
                    if not series.empty:
                        return float(series.iloc[0])
                if scattering_angle_col in summary_df.columns:
                    angle_series = pd.to_numeric(
                        summary_df[scattering_angle_col], errors="coerce"
                    ).dropna()
                    if not angle_series.empty:
                        theta = angle_series.iloc[0]
                        theta_rad = np.radians(theta)
                        return float(
                            (4 * np.pi / self._default_wavelength())
                            * np.sin(theta_rad / 2.0)
                        )
                for col in summary_df.columns:
                    lower = col.lower()
                    if "scattering vector" in lower or lower.strip() == "q":
                        series = pd.to_numeric(summary_df[col], errors="coerce").dropna()
                        if not series.empty:
                            return float(series.iloc[0])
                return None

            for meas_key in measurement_keys:
                measurement = measurement_nodes.get(meas_key, {})
                summary_map = measurement.get("summary", {})
                measurement_summary = summary_map.get(summary_file)
                measurement_q = _extract_q(measurement_summary)
                measurement_beta = None
                if measurement_summary is not None and beta_col in measurement_summary.columns:
                    beta_series = pd.to_numeric(
                        measurement_summary[beta_col], errors="coerce"
                    ).dropna()
                    if not beta_series.empty:
                        measurement_beta = float(beta_series.iloc[0])

                if measurement_summary is not None:
                    for temp_col in ["Temperature [K]", "Temperature"]:
                        if temp_col in measurement_summary.columns:
                            value = pd.to_numeric(
                                measurement_summary[temp_col], errors="coerce"
                            ).dropna()
                            if not value.empty:
                                temperature_values.append(float(value.iloc[0]))
                                break
                    for visc_col in ["Solvent viscosity [mPas]", "Solvent viscosity [mPa*s]", "Viscosity [mPas]"]:
                        if visc_col in measurement_summary.columns:
                            value = pd.to_numeric(
                                measurement_summary[visc_col], errors="coerce"
                            ).dropna()
                            if not value.empty:
                                viscosity_values.append(float(value.iloc[0]) * 1e-3)
                                break

                repetitions = measurement.get("repetitions", {})
                for repetition, files in repetitions.items():
                    corr_df = files.get(correlation_file)
                    if corr_df is None or corr_df.empty:
                        continue

                    delay_col = next(
                        (
                            col
                            for col in corr_df.columns
                            if "delay" in col.lower() or "time" in col.lower()
                        ),
                        corr_df.columns[0],
                    )
                    value_col = next(
                        (
                            col
                            for col in corr_df.columns
                            if "g2" in col.lower() or "correlation" in col.lower()
                        ),
                        corr_df.columns[1] if len(corr_df.columns) > 1 else corr_df.columns[0],
                    )

                    tau = pd.to_numeric(corr_df[delay_col], errors="coerce").to_numpy(dtype=float)
                    g2 = pd.to_numeric(corr_df[value_col], errors="coerce").to_numpy(dtype=float)
                    valid_mask = np.isfinite(tau) & np.isfinite(g2)
                    if not np.any(valid_mask):
                        continue
                    tau = tau[valid_mask]
                    g2 = g2[valid_mask]
                    order = np.argsort(tau)
                    tau = tau[order]
                    g2 = g2[order]

                    rep_summary_df = files.get(summary_file)
                    q_current = _extract_q(rep_summary_df) or measurement_q
                    if q_current is None:
                        continue

                    beta_value = measurement_beta
                    if beta_value is None and rep_summary_df is not None and beta_col in rep_summary_df.columns:
                        beta_series = pd.to_numeric(
                            rep_summary_df[beta_col], errors="coerce"
                        ).dropna()
                        if not beta_series.empty:
                            beta_value = float(beta_series.iloc[0])
                    beta_value = beta_value if beta_value is not None else 1.0

                    sigma = self._estimate_correlation_uncertainty(
                        measurement,
                        repetition,
                        tau,
                        g2,
                        delay_col,
                        value_col,
                        file_name=correlation_file,
                    )

                    datasets.append(
                        CorennDataSet(
                            tau=tau,
                            g2=g2,
                            q=q_current,
                            beta=beta_value,
                            sigma=sigma,
                            metadata={"measurement": meas_key, "repetition": repetition},
                        )
                    )

            temp_value = float(np.mean(temperature_values)) if temperature_values else None
            visc_value = float(np.mean(viscosity_values)) if viscosity_values else None
            return datasets, temp_value, visc_value

        def fit_correlation_function(
            self,
            data,
            sample,
            experiment,
            measurement,
            repetition,
            file_name="Correlation Function.csv",
            ax=None,
            logx=True,
            plot=True,
            lag_results=None,
            lag_signal_column="CR CHA [kHz]",
            lag_min_ratio=1.0,
            lag_drop_ratio=0.5,
            lag_window_factor=5.0,
            max_components=1,
        ):
            """
            Fit a simple exponential decay to the correlation function of a repetition.

            Parameters
            ----------
            data : Data
                Loaded dataset structure from ``Dataloader.load``.
            sample : str
                Sample identifier in ``data.samples``.
            experiment : str
                Experiment identifier under the chosen sample.
            measurement : int or str
                Measurement identifier (as stored in ``data.samples``).
            repetition : str or int
                Repetition identifier (e.g., ``"Repetition 0"``).
            file_name : str, optional
                Correlation function file within the repetition (default
                ``"Correlation Function.csv"``).
            ax : matplotlib.axes.Axes, optional
                Axes to draw on when ``plot=True``. A new figure is created otherwise.
            logx : bool, optional
                Plot the delay axis on a logarithmic scale (default ``True``).
            plot : bool, optional
                Whether to create a plot (default ``True``).
            lag_results : dict, optional
                Output from ``trace_lags`` used to constrain the fit.
            lag_signal_column : str, optional
                Channel name to use when extracting wavelet-lag constraints.
            lag_min_ratio : float, optional
                Minimum ratio of lag to sampling period to accept from wavelet constraints.
            lag_drop_ratio : float, optional
                Power drop ratio for estimating lag uncertainties.
            lag_window_factor : float, optional
                Width multiplier that defines the fitting window around the constrained lag.
            max_components : int, optional
                Maximum number of exponential components to fit. When ``>=2`` a
                double-exponential model is attempted even without explicit
                wavelet constraints (default ``1``).

            Returns
            -------
            dict or (dict, matplotlib.axes.Axes)
                Fit parameters and optionally the axes when ``plot`` is ``True``.
            """

            sample_node = data.samples.get(sample, {})
            experiment_node = sample_node.get("experiments", {}).get(experiment, {})
            measurements = experiment_node.get("measurements", {})
            meas_key = self._resolve_measurement_key(measurements, measurement)
            measurement_node = measurements.get(meas_key, {})
            repetitions = measurement_node.get("repetitions", {})
            rep_key = self._resolve_repetition_name(repetitions, repetition)

            repetition_data = repetitions.get(rep_key, {})
            if file_name not in repetition_data:
                raise KeyError(
                    f"{file_name} not found for measurement '{measurement}' repetition '{rep_key}'."
                )

            corr_df = repetition_data[file_name]
            if corr_df is None or corr_df.empty:
                raise ValueError("Correlation function data frame is empty.")

            # Identify delay/g2 columns
            delay_col = next(
                (
                    col
                    for col in corr_df.columns
                    if "delay" in col.lower() or "time" in col.lower()
                ),
                corr_df.columns[0],
            )
            value_col = next(
                (
                    col
                    for col in corr_df.columns
                    if "g2" in col.lower() or "correlation" in col.lower()
                ),
                corr_df.columns[1] if len(corr_df.columns) > 1 else corr_df.columns[0],
            )

            tau = pd.to_numeric(corr_df[delay_col], errors="coerce").to_numpy(dtype=float)
            g2 = pd.to_numeric(corr_df[value_col], errors="coerce").to_numpy(dtype=float)

            mask = np.isfinite(tau) & np.isfinite(g2)
            tau, g2 = tau[mask], g2[mask]
            if tau.size == 0:
                raise ValueError("No valid correlation data to fit.")

            order = np.argsort(tau)
            tau, g2 = tau[order], g2[order]

            sigma_full = self._estimate_correlation_uncertainty(
                measurement_node,
                rep_key,
                tau,
                g2,
                delay_col,
                value_col,
                file_name=file_name,
            )

            lag_constraint = None
            if lag_results is not None:
                try:
                    constraints = self.wavelet_lag_constraints(
                        lag_results,
                        measurement=measurement,
                        signal_column=lag_signal_column,
                        repetition=repetition,
                        min_ratio=lag_min_ratio,
                        drop_ratio=lag_drop_ratio,
                    )
                    if constraints:
                        lag_constraint = sorted(
                            constraints, key=lambda x: x["lag"]
                        )[0]
                except KeyError:
                    lag_constraint = None

            positive_mask = g2 > 0
            if positive_mask.sum() < 2:
                raise ValueError(
                    "Not enough positive correlation values to perform exponential fit."
                )

            tau_fit = tau[positive_mask]
            y_fit = g2[positive_mask]
            sigma_fit = None
            if sigma_full is not None and len(sigma_full) == len(g2):
                sigma_fit = sigma_full[positive_mask]
                sigma_fit = np.where(
                    np.isfinite(sigma_fit) & (sigma_fit > 0), sigma_fit, 1.0
                )

            amplitude = float(np.max(y_fit))

            multi_fit = False
            allow_double = max_components >= 2
            fit_components = []

            if lag_constraint is not None and np.isfinite(lag_constraint["lag"]):
                tau_c = float(lag_constraint["lag"])
                slope = -1.0 / tau_c if tau_c > 0 else -1.0
                fitted = amplitude * np.exp(slope * tau)
                fit_components.append({"amplitude": amplitude, "tau_c": tau_c})
            else:
                extra_constraints = []
                if lag_results is not None:
                    try:
                        extra_constraints = self.wavelet_lag_constraints(
                            lag_results,
                            measurement=measurement,
                            signal_column=lag_signal_column,
                            repetition=repetition,
                            min_ratio=lag_min_ratio,
                            drop_ratio=lag_drop_ratio,
                        )
                    except Exception:
                        extra_constraints = []

                valid_constraints = [
                    c for c in extra_constraints if np.isfinite(c["lag"])
                ]
                valid_constraints.sort(key=lambda c: c["lag"])

                if allow_double and len(valid_constraints) >= 2:
                    multi_fit = True
                    l1, l2 = valid_constraints[0]["lag"], valid_constraints[1]["lag"]
                    a1 = amplitude * 0.6
                    a2 = amplitude - a1

                    def _residual_double(params):
                        amp1, amp2, tau1, tau2 = params
                        model = amp1 * np.exp(-tau_fit / tau1) + amp2 * np.exp(
                            -tau_fit / tau2
                        )
                        residual = model - y_fit
                        if sigma_fit is not None:
                            residual = residual / sigma_fit
                        return residual

                    from scipy.optimize import least_squares

                    try:
                        result = least_squares(
                            _residual_double,
                            x0=[a1, a2, max(l1, 1e-6), max(l2, 1e-6)],
                            bounds=([0, 0, 1e-8, 1e-8], [np.inf, np.inf, np.inf, np.inf]),
                        )
                        amp1, amp2, tau1, tau2 = result.x
                        fitted = amp1 * np.exp(-tau / tau1) + amp2 * np.exp(-tau / tau2)
                        fit_components.extend(
                            [
                                {"amplitude": amp1, "tau_c": tau1},
                                {"amplitude": amp2, "tau_c": tau2},
                            ]
                        )
                    except Exception:
                        multi_fit = False

                elif allow_double:
                    multi_fit = True

                    def _residual_double(params):
                        amp1, amp2, tau1, tau2 = params
                        model = amp1 * np.exp(-tau_fit / tau1) + amp2 * np.exp(
                            -tau_fit / tau2
                        )
                        residual = model - y_fit
                        if sigma_fit is not None:
                            residual = residual / sigma_fit
                        return residual

                    from scipy.optimize import least_squares

                    n = len(tau_fit)
                    tau1_guess = tau_fit[n // 4] if n >= 4 else tau_fit[0]
                    tau2_guess = tau_fit[min(n - 1, (3 * n) // 4)] if n >= 2 else tau_fit[-1]
                    tau1_guess = max(float(tau1_guess), 1e-6)
                    tau2_guess = max(float(tau2_guess), 1e-6)
                    a1 = amplitude * 0.5
                    a2 = max(amplitude - a1, 1e-8)

                    try:
                        result = least_squares(
                            _residual_double,
                            x0=[a1, a2, tau1_guess, tau2_guess],
                            bounds=([0, 0, 1e-8, 1e-8], [np.inf, np.inf, np.inf, np.inf]),
                        )
                        amp1, amp2, tau1, tau2 = result.x
                        fitted = amp1 * np.exp(-tau / tau1) + amp2 * np.exp(-tau / tau2)
                        fit_components.extend(
                            [
                                {"amplitude": amp1, "tau_c": tau1},
                                {"amplitude": amp2, "tau_c": tau2},
                            ]
                        )
                    except Exception:
                        multi_fit = False

                if not multi_fit:
                    def _residuals(params):
                        amp, tau_c_local = params
                        residual = amp * np.exp(-tau_fit / tau_c_local) - y_fit
                        if sigma_fit is not None:
                            residual = residual / sigma_fit
                        return residual

                    amp0 = amplitude
                    tau0 = tau_fit[np.argmax(y_fit < amp0 / np.e)] if np.any(
                        y_fit < amp0 / np.e
                    ) else tau_fit[-1]
                    tau0 = max(tau0, 1e-6)
                    from scipy.optimize import least_squares

                    result = least_squares(
                        _residuals, x0=[amp0, tau0], bounds=([0, 1e-8], [np.inf, np.inf])
                    )
                    amplitude, tau_c = result.x
                    slope = -1.0 / tau_c
                    fitted = amplitude * np.exp(-tau / tau_c)
                    fit_components.append({"amplitude": amplitude, "tau_c": tau_c})

            residual = (
                y_fit
                - sum(
                    comp["amplitude"] * np.exp(-tau_fit / comp["tau_c"])
                    for comp in fit_components
                )
            )
            ss_res = float(np.sum(residual**2))
            ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            fit_results = {
                "fit_type": "double" if multi_fit else "single",
                "components": fit_components,
                "r2": r2,
                "delay": tau,
                "g2": g2,
                "fitted": fitted,
                "repetition": rep_key,
                "measurement": meas_key,
                "lag_constraint": lag_constraint,
            }
            if fit_components:
                for component in fit_components:
                    tau_val = component.get("tau_c")
                    if tau_val is not None and np.isfinite(tau_val) and tau_val > 0:
                        component["gamma"] = float(1.0 / tau_val)
                    else:
                        component["gamma"] = np.nan
                fit_results["amplitude"] = fit_components[0]["amplitude"]
                fit_results["tau_c"] = fit_components[0]["tau_c"]
                fit_results["gamma"] = fit_components[0]["gamma"]

            if not plot:
                return fit_results

            if ax is None:
                _, ax = plt.subplots()

            ax.plot(tau, g2, "o", ms=4, label="Data")
            ax.plot(tau, fitted, "-", lw=2, label="Exponential fit")

            if logx:
                ax.set_xscale("log")

            ax.set_xlabel(delay_col)
            ax.set_ylabel(value_col)
            ax.set_title(
                f"Correlation fit - {rep_key} (Measurement {measurement})"
            )
            ax.legend()

            return fit_results, ax
        

        def corenn_multi_q(
            self,
            data=None,
            sample=None,
            experiment=None,
            measurements: Optional[Sequence] = None,
            correlation_file="Correlation Function.csv",
            summary_file="Summary.csv",
            beta_col="Intercept",
            q_col="Scattering vector [1/nm]",
            scattering_angle_col="Scattering angle [deg]",
            temperature: Optional[float] = None,
            viscosity: Optional[float] = None,
            datasets: Optional[List[CorennDataSet]] = None,
            radius_range: Tuple[float, float] = (1e-10, 1e-5),
            n_basis: int = 30,
            n_R: int = 200,
            lam_min: float = 1e-12,
            lam_max: float = 1e3,
            n_lam: int = 150,
            regularization: float = 1e-9,
        ) -> Dict[str, object]:
            """
            Perform multi-q CORENN inversion to estimate a particle size distribution.

            Parameters
            ----------
            data : Data, optional
                Dataset returned by :meth:`Dataloader.load`. Required when
                ``datasets`` is ``None``.
            sample : str, optional
                Sample identifier when building datasets automatically.
            experiment : str, optional
                Experiment identifier used to scope the measurements.
            measurements : sequence, optional
                Subset of measurement identifiers to include. Defaults to all.
            correlation_file : str, optional
                Repetition-level CSV containing ``g2`` (default ``"Correlation Function.csv"``).
            summary_file : str, optional
                Summary CSV used for metadata such as ``q`` and ``beta``.
            beta_col : str, optional
                Column storing the coherence factor ``β`` (default ``"Intercept"``).
            q_col : str, optional
                Column storing scattering vector values in ``[1/nm]``.
            scattering_angle_col : str, optional
                Column storing scattering angle in degrees, used when ``q_col`` is absent.
            temperature : float, optional
                Absolute temperature [K]. When ``None`` the value is inferred from
                ``summary_file`` if possible.
            viscosity : float, optional
                Solvent viscosity [Pa·s]. When ``None`` it is inferred from the
                summary file (columns reported in mPa·s are converted).
            datasets : list of :class:`CorennDataSet`, optional
                Pre-constructed datasets. When provided the loader-driven
                extraction is skipped.
            radius_range : tuple, optional
                Lower/upper bounds of the radius grid in metres.
            n_basis, n_R : int, optional
                Number of B-spline basis functions and quadrature points.
            lam_min, lam_max, n_lam : float/int, optional
                Range and resolution of the regularisation parameter scan.
            regularization : float, optional
                Small diagonal damping factor for numerical stability.

            Returns
            -------
            dict
                Dictionary containing the PSD, optimal λ, fitted ``g2`` traces and
                metadata for each repetition.
            """

            if datasets is None:
                if data is None or sample is None or experiment is None:
                    raise ValueError(
                        "data, sample, and experiment must be provided when datasets are not supplied."
                    )
                datasets, inferred_temp, inferred_eta = self._prepare_corenn_datasets(
                    data,
                    sample,
                    experiment,
                    measurements,
                    correlation_file,
                    summary_file,
                    beta_col,
                    q_col,
                    scattering_angle_col,
                )
                if temperature is None:
                    temperature = inferred_temp
                if viscosity is None:
                    viscosity = inferred_eta

            if not datasets:
                raise ValueError("No datasets available for CORENN inversion.")

            if temperature is None or viscosity is None:
                raise ValueError("Temperature and viscosity must be specified.")

            temperature = float(temperature)
            viscosity = float(viscosity)
            if viscosity > 1.0:  # assume input in mPa·s
                viscosity *= 1e-3

            R_min, R_max = radius_range
            R_grid = _build_radius_grid(R_min, R_max, n_R=n_R)
            w_R = _trapezoid_weights(R_grid)
            Phi, Phi_dd, _ = _build_bspline_basis(R_grid, n_basis=n_basis, degree=3)
            B = Phi_dd * np.sqrt(w_R[:, None])
            L_reg = B.T @ B

            diffusion = k_BOLTZMANN * temperature / (6.0 * np.pi * viscosity * R_grid)
            W_Phi = w_R[:, None] * Phi

            K_list = []
            y_blocks = []
            K_blocks = []
            beta_values = []
            metadata = []

            for ds in datasets:
                tau = np.asarray(ds.tau, dtype=float).ravel()
                g_minus = np.asarray(ds.g2, dtype=float).ravel()
                if tau.shape != g_minus.shape:
                    raise ValueError("tau and g2 must share the same shape per dataset.")
                q_si = float(ds.q) * 1e9  # convert from 1/nm to 1/m
                K_d = _build_kernel_for_dataset(q_si, tau, diffusion, W_Phi)
                K_list.append(K_d)
                metadata.append(ds.metadata or {})

                beta_val = ds.beta if ds.beta not in (None, 0) else 1.0
                beta_values.append(beta_val)

                if ds.sigma is not None:
                    sigma = np.asarray(ds.sigma, dtype=float).ravel()
                    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
                    suppress_mask = g_minus < (-2.0 * sigma)
                    if np.any(suppress_mask):
                        g_minus = g_minus.copy()
                        g_minus[suppress_mask] = 0.0
                else:
                    sigma = np.ones_like(g_minus, dtype=float)

                shifted = np.clip(g_minus, 0.0, None) / max(beta_val, 1e-12)
                g_target = np.sqrt(shifted)

                weights = 1.0 / np.clip(sigma, 1e-12, None)
                y_blocks.append(weights * g_target)
                K_blocks.append(weights[:, None] * K_d)

            K_stack = np.vstack(K_blocks)
            y_stack = np.concatenate(y_blocks)
            S = K_stack.T @ K_stack
            b = K_stack.T @ y_stack

            lambda_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lam)
            best_lambda = None
            best_x = None
            best_gcv = np.inf
            gcv_values = []
            n_data = y_stack.size
            ident = np.eye(n_basis)

            for lam in lambda_grid:
                A = S + lam * L_reg + regularization * ident
                x = np.linalg.solve(A, b)
                residual = y_stack - K_stack @ x
                num = float(np.mean(residual**2))
                A_inv_S = np.linalg.solve(A, S)
                d_eff = float(np.trace(A_inv_S))
                denom = (1.0 - d_eff / n_data) ** 2
                gcv = num / denom if denom > 1e-12 else np.inf
                gcv_values.append(gcv)
                if gcv < best_gcv:
                    best_gcv = gcv
                    best_lambda = lam
                    best_x = x

            if best_x is None:
                raise RuntimeError("Failed to determine a valid PSD solution.")

            I_R = Phi @ best_x
            I_R = np.clip(I_R, 0.0, None)

            g2_model_list = []
            dataset_records = []
            for ds, K_d, beta_val, meta in zip(
                datasets, K_list, beta_values, metadata
            ):
                u = K_d @ best_x
                g2_model = beta_val * (u**2)
                measured = np.asarray(ds.g2, dtype=float)
                peak_model = float(np.max(g2_model)) if g2_model.size else 0.0
                peak_target = float(np.max(measured)) if measured.size else 0.0
                if peak_model > 0 and peak_target > 0:
                    g2_model = g2_model * (peak_target / peak_model)
                g2_model_list.append(g2_model)
                dataset_records.append(
                    {
                        "measurement": (meta or {}).get("measurement"),
                        "repetition": (meta or {}).get("repetition"),
                        "q": float(ds.q),
                        "beta": float(beta_val),
                        "tau": np.asarray(ds.tau, dtype=float),
                        "g2": np.asarray(ds.g2, dtype=float),
                        "sigma": None
                        if ds.sigma is None
                        else np.asarray(ds.sigma, dtype=float),
                        "g2_model": g2_model,
                        "metadata": meta or {},
                    }
                )

            return {
                "R_grid": R_grid,
                "Phi": Phi,
                "coefficients": best_x,
                "I_R": I_R,
                "lambda_opt": best_lambda,
                "lambda_grid": lambda_grid,
                "gcv_values": np.asarray(gcv_values, dtype=float),
                "K_list": K_list,
                "g2_model_list": g2_model_list,
                "datasets": dataset_records,
                "temperature": temperature,
                "viscosity": viscosity,
            }
        
        def plot_corenn_relaxation_rates(
            self,
            corenn_results,
            ax=None,
            label=None,
            logx=False,
            logy=False,
            fit_line=True,
            tau_max=None,
            use_model=False,
            show_correlation=False,
            measurement=None,
            repetition=None,
        ):
            """
            Plot CORENN-derived relaxation rates :math:`\\Gamma` versus ``q^2``.

            Parameters
            ----------
            corenn_results : dict
                Output from :meth:`corenn_multi_q`.
            ax : matplotlib.axes.Axes, optional
                Axes to draw on. Creates a new figure when ``None``.
            label : str, optional
                Legend label for the plotted dataset.
            logx, logy : bool, optional
                Use logarithmic scaling on the respective axes.
            fit_line : bool, optional
                If ``True`` draw an origin-constrained linear fit and report its
                slope (diffusivity) in the legend.
            tau_max : float, optional
                Maximum delay time (seconds) to include when estimating
                relaxation rates.
            use_model : bool, optional
                When ``True`` use the fitted CORENN ``g2`` traces; otherwise the
                measured ``g2`` data (default ``False``). The plotting routine
                automatically falls back to the measured trace when the model
                lacks sufficient decay to estimate a relaxation rate.
            show_correlation : bool, optional
                When ``True`` draw a second subplot showing the correlation
                function, fit and uncertainty for a specific measurement.
            measurement : hashable, optional
                Measurement identifier for the correlation subplot. Required
                when ``show_correlation`` is ``True``.
            repetition : hashable, optional
                Repetition identifier for the correlation subplot. When omitted,
                the first repetition for the selected measurement is used.

            Returns
            -------
            matplotlib.axes.Axes or (Axes, dict)
                Axes containing the plot and, when ``fit_line`` is enabled, a
                dictionary describing the fitted slope.
            """

            dataset_records = corenn_results.get("datasets")
            if not dataset_records:
                raise ValueError("corenn_results does not include dataset entries.")

            corr_dataset = None
            if show_correlation:
                if measurement is None:
                    raise ValueError("measurement must be specified when show_correlation=True.")
                for entry in dataset_records:
                    if entry.get("measurement") != measurement:
                        continue
                    if repetition is not None and entry.get("repetition") != repetition:
                        continue
                    corr_dataset = entry
                    break
                if corr_dataset is None:
                    raise ValueError(
                        f"No dataset found for measurement '{measurement}'"
                        + (f", repetition '{repetition}'." if repetition else ".")
                    )

            q_values = []
            gamma_values = []
            for entry in dataset_records:
                q_val = entry.get("q")
                tau = entry.get("tau")
                beta = entry.get("beta", 1.0)
                if q_val in (None, 0) or tau is None:
                    continue

                traces = []
                if use_model:
                    traces.append(entry.get("g2_model"))
                traces.append(entry.get("g2"))
                if not use_model:
                    traces.append(entry.get("g2_model"))

                gamma = None
                for trace in traces:
                    if trace is None:
                        continue
                    gamma = self._estimate_gamma_from_trace(
                        tau,
                        trace,
                        beta,
                        tau_max=tau_max,
                    )
                    if gamma is not None and gamma > 0:
                        break

                if gamma is None:
                    continue
                q_values.append(float(q_val))
                gamma_values.append(float(gamma))

            if not q_values:
                raise ValueError("No valid relaxation rates to plot.")

            q_values = np.asarray(q_values, dtype=float)
            gamma_values = np.asarray(gamma_values, dtype=float)
            q_squared = q_values**2

            corr_ax = None
            if show_correlation:
                if ax is None:
                    _, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
                    ax, corr_ax = axes
                else:
                    if isinstance(ax, (list, tuple)) and len(ax) >= 2:
                        ax, corr_ax = ax[0], ax[1]
                    else:
                        raise ValueError(
                            "ax must contain two axes when show_correlation=True."
                        )
                ax.cla()
                corr_ax.cla()
            elif ax is None:
                _, ax = plt.subplots()
            else:
                ax.cla()

            scatter = ax.errorbar(
                q_squared,
                gamma_values,
                fmt="o",
                ms=5,
                lw=0,
                label=None,
            )

            legend_entries = []
            slope_value = None

            if fit_line and q_squared.size >= 2:
                denom = float(np.dot(q_squared, q_squared))
                if denom > 0:
                    slope_value = float(np.dot(q_squared, gamma_values) / denom)
                    x_fit = np.linspace(np.min(q_squared), np.max(q_squared), 200)
                    color = None
                    if hasattr(scatter, "lines") and scatter.lines:
                        color = scatter.lines[0].get_color()
                    ax.plot(
                        x_fit,
                        slope_value * x_fit,
                        linestyle="--",
                        linewidth=1.5,
                        color=color,
                        alpha=0.8,
                        label=None,
                    )

            final_label = label or "CORENN"
            if slope_value is not None:
                final_label = f"{final_label} (D={slope_value:.3g})"
            scatter.lines[0].set_label(final_label)
            legend_entries.append(final_label)

            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")

            ax.set_xlabel("q^2 [1/nm^2]")
            ax.set_ylabel("Relaxation rate Γ [1/s]")
            ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
            if legend_entries:
                ax.legend()

            extras = None
            if fit_line and slope_value is not None:
                extras = {"slope": slope_value, "diffusivity": slope_value}

            if show_correlation and corr_ax is not None and corr_dataset is not None:
                tau = np.asarray(corr_dataset.get("tau"), dtype=float)
                g_measured = np.asarray(corr_dataset.get("g2"), dtype=float)
                g_fit = np.asarray(corr_dataset.get("g2_model"), dtype=float)
                corr_ax.plot(tau, g_measured, "o", ms=4, label="Measured")
                corr_ax.plot(tau, g_fit, "-", lw=2, label="CORENN fit")
                n_corr = g_measured.size
                if n_corr:
                    lag_indices = np.arange(n_corr, dtype=float)
                    effective_samples = np.maximum(n_corr - lag_indices, 1.0)
                    relative = (lag_indices + 1.0) / effective_samples
                    corr_sigma = np.abs(g_measured) * relative
                    corr_sigma = np.where(np.isfinite(corr_sigma), corr_sigma, 0.0)
                    corr_sigma = np.minimum(corr_sigma, np.abs(g_measured))
                    corr_ax.fill_between(
                        tau,
                        g_measured - corr_sigma,
                        g_measured + corr_sigma,
                        color="C0",
                        alpha=0.2,
                        label="Uncertainty",
                    )
                corr_ax.set_xscale("log")
                corr_ax.set_xlabel("Lag time τ [s]")
                corr_ax.set_ylabel("g₂ - 1")
                corr_ax.set_title(
                    f"Measurement {corr_dataset.get('measurement')}, "
                    f"{corr_dataset.get('repetition')}"
                )
                corr_ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
                corr_ax.legend()

                if extras is not None:
                    return (ax, corr_ax), extras
                return ax, corr_ax

            if extras is not None:
                return ax, extras

            return ax
        
        def fit_diffusivity(
            self,
            data,
            sample,
            experiment,
            correlation_file="Correlation Function.csv",
            summary_file="Summary.csv",
            intensity_col="Laser intensity [mW]",
            q_col="Scattering vector [1/nm]",
            scattering_angle_col="Scattering angle [deg]",
            use_multi=False,
            **fit_kwargs,
        ):
            """
            Fit the diffusion coefficient for each measurement by combining correlation fits.

            Parameters
            ----------
            data : Data
                Loaded dataset structure from ``Dataloader.load``.
            sample : str
                Sample identifier in ``data.samples``.
            experiment : str or iterable of str
                One experiment name or a collection of experiment names to fit
                independently.
            correlation_file : str, optional
                Name of the correlation function file in each repetition folder.
            summary_file : str, optional
                Summary CSV used to obtain scattering angles/laser intensity.
            q_col : str, optional
                Column name containing scattering vectors (fallback).
            scattering_angle_col : str, optional
                Angle column used to compute ``q`` when available.
            use_multi : bool, optional
                When ``True``, request a multi-exponential correlation fit and use
                the fastest relaxation rate (largest ``Γ``) to estimate the
                diffusivity (default ``False``).
            **fit_kwargs : dict, optional
                Keyword arguments forwarded to ``fit_correlation_function`` (e.g. ``lag_results``).

            Returns
            -------
            list of dict or dict of list
                When ``experiment`` is a single name, returns a list of per-measurement
                dictionaries. When multiple experiments are requested, returns a dict
                mapping each experiment name to its corresponding list. Each measurement
                entry includes the average relaxation rate ``gamma`` alongside the
                diffusivity statistics.
            """

            def _coerce_scalar_numeric(value):
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return None
                try:
                    return float(value)
                except Exception:
                    try:
                        coerced = pd.to_numeric(value, errors="coerce")
                        if hasattr(coerced, "item"):
                            coerced = coerced.item()
                        return None if pd.isna(coerced) else float(coerced)
                    except Exception:
                        return None

            def _match_repetition_row(summary_df, repetition_label):
                if summary_df is None or summary_df.empty:
                    return None

                if "Repetition" in summary_df.columns:
                    rep_token = repetition_label
                    match = re.search(r"(\d+)", repetition_label)
                    if match:
                        rep_token = match.group(1)

                    def _matches(value):
                        if pd.isna(value):
                            return False
                        value_str = str(value)
                        return value_str == repetition_label or value_str == rep_token

                    rows = summary_df[summary_df["Repetition"].apply(_matches)]
                    if not rows.empty:
                        return rows.iloc[0]

                return None

            def _resolve_scattering_vector(summary_source, summary_row):
                """Return q-value derived from scattering angle or q columns."""

                def _series_first_numeric(frame, column):
                    if frame is None or column is None:
                        return None
                    try:
                        series = pd.to_numeric(frame[column], errors="coerce").dropna()
                    except Exception:
                        return None
                    if series.empty:
                        return None
                    return _coerce_scalar_numeric(series.iloc[0])

                scattering_angle = None
                q_value = None

                if summary_row is not None:
                    if scattering_angle_col in summary_row:
                        scattering_angle = _coerce_scalar_numeric(
                            summary_row.get(scattering_angle_col)
                        )
                    if q_col in summary_row:
                        q_value = _coerce_scalar_numeric(summary_row.get(q_col))

                columns = getattr(summary_source, "columns", None)
                if scattering_angle is None and summary_source is not None and columns is not None:
                    candidate = None
                    if scattering_angle_col in columns:
                        candidate = scattering_angle_col
                    else:
                        for col in columns:
                            if "angle" in col.lower():
                                candidate = col
                                break
                    if candidate:
                        scattering_angle = _series_first_numeric(summary_source, candidate)

                if scattering_angle is not None:
                    theta = np.radians(scattering_angle)
                    return float(
                        (4 * np.pi / self._default_wavelength()) * np.sin(theta / 2)
                    )

                if q_value is None and summary_source is not None and columns is not None:
                    candidate = None
                    if q_col in columns:
                        candidate = q_col
                    else:
                        for col in columns:
                            lowered = col.lower()
                            if "scattering vector" in lowered or lowered.strip() == "q":
                                candidate = col
                                break
                    if candidate:
                        q_value = _series_first_numeric(summary_source, candidate)

                return q_value

            def _select_summary_frame(summary_dict, preferred_name, allow_any=True):
                """Return a summary dataframe matching ``preferred_name`` if available."""

                if not isinstance(summary_dict, dict) or not summary_dict:
                    return None

                def _is_frame(obj):
                    return hasattr(obj, "columns")

                if preferred_name and preferred_name in summary_dict:
                    candidate = summary_dict[preferred_name]
                    if _is_frame(candidate):
                        return candidate

                preferred_lower = preferred_name.lower() if preferred_name else None
                if preferred_lower:
                    for name, frame in summary_dict.items():
                        if name.lower() == preferred_lower and _is_frame(frame):
                            return frame

                for name, frame in summary_dict.items():
                    if "summary" in name.lower() and _is_frame(frame):
                        return frame

                if allow_any:
                    for frame in summary_dict.values():
                        if _is_frame(frame):
                            return frame

                return None

            sample_node = data.samples.get(sample, {})
            experiment_nodes = sample_node.get("experiments", {})

            if isinstance(experiment, (list, tuple, set)):
                experiments = [str(exp) for exp in experiment]
            else:
                experiments = [experiment]

            multi_experiment = len(experiments) > 1
            aggregated = {}

            for experiment_name in experiments:
                experiment_node = experiment_nodes.get(experiment_name, {})
                measurements = experiment_node.get("measurements", {})
                results = []

                for meas_key in sorted(measurements):
                    measurement = measurements[meas_key]
                    repetitions = measurement.get("repetitions", {})
                    summary_map = measurement.get("summary", {})
                    summary_df = _select_summary_frame(summary_map, summary_file, allow_any=True)
                    measurement_q_default = _resolve_scattering_vector(summary_df, None)

                    q_measurements = []
                    rep_diffs = []
                    rep_gammas = []

                    for repetition, files in repetitions.items():
                        rep_summary_df = _select_summary_frame(
                            {k: v for k, v in files.items()}, summary_file, allow_any=False
                        )
                        summary_source = rep_summary_df if rep_summary_df is not None else summary_df
                        summary_row = _match_repetition_row(summary_source, repetition)
                        q_current = _resolve_scattering_vector(summary_source, summary_row)
                        if q_current is None:
                            q_current = measurement_q_default
                        if q_current is not None:
                            q_measurements.append(q_current)

                        fit_params = {"plot": False}
                        fit_params.update(fit_kwargs)
                        if use_multi and "max_components" not in fit_params:
                            fit_params["max_components"] = 2

                        try:
                            fit_output = self.fit_correlation_function(
                                data,
                                sample,
                                experiment_name,
                                meas_key,
                                repetition,
                                file_name=correlation_file,
                                **fit_params,
                            )
                        except (KeyError, ValueError):
                            continue

                        fit = fit_output[0] if isinstance(fit_output, tuple) else fit_output

                        components = fit.get("components", [])
                        if not components:
                            continue

                        amps = np.array([c["amplitude"] for c in components], dtype=float)
                        taus = np.array([c["tau_c"] for c in components], dtype=float)
                        mask = np.isfinite(amps) & np.isfinite(taus) & (taus > 0)
                        if not np.any(mask):
                            continue

                        gamma_candidates = []
                        for tau_val in taus[mask]:
                            gamma_val = float(1.0 / tau_val) if tau_val > 0 else np.nan
                            if np.isfinite(gamma_val) and gamma_val > 0:
                                gamma_candidates.append(gamma_val)

                        gamma_eff = None
                        if use_multi and gamma_candidates:
                            gamma_eff = float(np.max(gamma_candidates))

                        if gamma_eff is None:
                            tau_eff = float(
                                np.sum(amps[mask] * taus[mask]) / np.sum(amps[mask])
                            )
                            if q_current is None or q_current <= 0 or tau_eff <= 0:
                                continue
                            gamma_eff = float(1.0 / tau_eff)

                        if not np.isfinite(gamma_eff) or gamma_eff <= 0 or q_current is None or q_current <= 0:
                            continue

                        diffusivity = float(gamma_eff / (q_current**2))
                        rep_diffs.append(diffusivity)
                        rep_gammas.append(gamma_eff)

                    if not q_measurements and measurement_q_default is not None:
                        q_measurements.append(measurement_q_default)

                    if not rep_diffs or not q_measurements or not rep_gammas:
                        continue

                    q_value = float(np.mean(q_measurements))
                    rep_diffs = np.asarray(rep_diffs, dtype=float)
                    rep_gammas = np.asarray(rep_gammas, dtype=float)
                    mean_D = float(np.mean(rep_diffs))
                    std_D = float(np.std(rep_diffs, ddof=1)) if rep_diffs.size > 1 else 0.0
                    sem_D = std_D / np.sqrt(rep_diffs.size) if rep_diffs.size > 1 else 0.0
                    gamma_mean = float(np.mean(rep_gammas))
                    gamma_std = (
                        float(np.std(rep_gammas, ddof=1)) if rep_gammas.size > 1 else 0.0
                    )
                    gamma_sem = (
                        gamma_std / np.sqrt(rep_gammas.size) if rep_gammas.size > 1 else 0.0
                    )

                    results.append(
                        {
                            "measurement": meas_key,
                            "experiment": experiment_name,
                            "q": float(q_value),
                            "q_squared": float(q_value**2),
                            "diffusivity": mean_D,
                            "std": std_D,
                            "sem": sem_D,
                            "count": int(rep_diffs.size),
                            "gamma": gamma_mean,
                            "gamma_std": gamma_std,
                            "gamma_sem": gamma_sem,
                        }
                    )

                aggregated[experiment_name] = results

            if multi_experiment:
                return aggregated

            return aggregated.get(experiments[0], [])

        def plot_diffusivity(
            self,
            diffusivity_results,
            ax=None,
            logx=False,
            logy=False,
            label=None,
            fit_line=False,
        ):
            """
            Plot the relaxation rate :math:`\\Gamma = 1/\\tau` versus :math:`q^2`.

            Parameters
            ----------
            diffusivity_results : list of dict or dict[str, list[dict]]
                Output from ``fit_diffusivity``. A dict plots each experiment on
                the same axes using the dict keys as legend labels.
            ax : matplotlib.axes.Axes, optional
                Existing axes to draw on; creates a new figure if ``None``.
            logx, logy : bool, optional
                Use logarithmic scales for the axes.
            label : str, optional
                Legend label for single-series plots.
            fit_line : bool, optional
                When ``True``, fit a straight line ``Γ = D · q² + intercept`` where
                the slope corresponds directly to the diffusivity ``D``.

            Returns
            -------
            matplotlib.axes.Axes or (matplotlib.axes.Axes, dict)
                Axes with the plotted relaxation rates. When ``fit_line`` is
                enabled, also returns a dict of fit parameters keyed by series label.
            """

            if isinstance(diffusivity_results, dict):
                datasets = [
                    (exp_label, data)
                    for exp_label, data in diffusivity_results.items()
                    if data
                ]
            else:
                datasets = [(label or "Diffusivity", diffusivity_results)]

            datasets = [(lbl, data) for lbl, data in datasets if data]
            if not datasets:
                raise ValueError("No diffusivity data to plot.")

            if ax is None:
                _, ax = plt.subplots()

            legend_labels = []
            fit_summary = {} if fit_line else None

            for idx, (series_label, data) in enumerate(datasets, start=1):
                q_squared = np.array(
                    [r.get("q_squared", np.nan) for r in data], dtype=float
                )
                gamma = np.array(
                    [r.get("gamma", np.nan) for r in data], dtype=float
                )
                gamma_err = np.array(
                    [r.get("gamma_sem", 0.0) for r in data], dtype=float
                )

                mask = np.isfinite(q_squared) & np.isfinite(gamma)
                if not np.any(mask):
                    continue

                base_label = series_label or (label if len(datasets) == 1 else f"Series {idx}")
                eb = ax.errorbar(
                    q_squared[mask],
                    gamma[mask],
                    yerr=gamma_err[mask],
                    fmt="o-",
                    ms=5,
                    lw=1.5,
                    label=None,
                )
                slope_value = None
                if fit_line and np.count_nonzero(mask) >= 2:
                    x_vals = q_squared[mask]
                    y_vals = gamma[mask]
                    denom = float(np.dot(x_vals, x_vals))
                    if denom > 0:
                        slope = float(np.dot(x_vals, y_vals) / denom)
                        x_fit = np.linspace(np.min(x_vals), np.max(x_vals), 200)
                        color = None
                        if hasattr(eb, "lines") and eb.lines:
                            color = eb.lines[0].get_color()
                        ax.plot(
                            x_fit,
                            slope * x_fit,
                            linestyle="--",
                            color=color,
                            linewidth=1.5,
                            alpha=0.8,
                        )
                        slope_value = slope
                        fit_summary[base_label] = {
                            "slope": slope,
                            "intercept": 0.0,
                            "diffusivity": float(slope),
                        }

                final_label = base_label
                if slope_value is not None and np.isfinite(slope_value):
                    final_label = f"{base_label} (D={slope_value:.3g})"
                if eb.lines:
                    eb.lines[0].set_label(final_label)
                if final_label:
                    legend_labels.append(final_label)

            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")

            ax.set_xlabel("q^2 [1/nm^2]")
            ax.set_ylabel("Relaxation rate Γ [1/s]")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

            if legend_labels:
                ax.legend()

            if fit_line:
                return ax, fit_summary

            return ax
        
        def plot_wavelet(
            self,
            wavelet_results,
            measurement,
            signal_column,
            repetition=None,
            ax=None,
            cmap="viridis",
            log_scale=True,
            colorbar=True,
            lag_results=None,
            show_lag_axis=True,
            annotate_lags=True,
            concatenate=False,
        ):
            """
            Visualise the wavelet scalogram for a given measurement/repetition.

            Parameters
            ----------
            wavelet_results : dict
                Output from ``trace`` containing per-measurement wavelet data.
            measurement : int or str
                Measurement identifier (matches keys returned by ``trace``).
            signal_column : str
                Column name that was analysed (e.g., ``"CR CHA [kHz]"``).
            repetition : str or None, optional
                Repetition identifier, e.g., ``"Repetition 0"``. Required unless
                ``concatenate=True``.
            ax : matplotlib.axes.Axes, optional
                Axes to draw on. Creates a new figure when ``None``.
            cmap : str, optional
                Matplotlib colormap for the scalogram (default ``"viridis"``).
            log_scale : bool, optional
                Display the y-axis (scale/frequency) on a logarithmic scale
                (default ``True``).
            colorbar : bool, optional
                Add a colorbar describing the coefficient magnitude (default ``True``).
            lag_results : dict, optional
                Output from ``trace_lags`` used to annotate lag information.
            show_lag_axis : bool, optional
                Display the secondary y-axis with lag values (default ``True``).
            annotate_lags : bool, optional
                Retained for API compatibility; currently no annotations are drawn.
            concatenate : bool, optional
                When ``True``, concatenate all repetitions for the measurement and
                plot the wavelet transform of the combined signal.

            Returns
            -------
            matplotlib.axes.Axes
                The axes containing the plotted scalogram.
            """

            meas_key = self._resolve_measurement_key(wavelet_results, measurement)
            measurement_data = wavelet_results[meas_key]

            if concatenate:
                concatenated = self._concatenate_measurement_wavelets(measurement_data)
                if not concatenated:
                    raise ValueError(
                        f"No data available to concatenate for measurement '{measurement}'."
                    )
                measurement_data = concatenated
                repetition = self.CONCATENATED_REPETITION_NAME

            if repetition is None and not concatenate:
                raise ValueError("repetition must be provided unless concatenate=True.")

            repetition_key = self._resolve_repetition_name(measurement_data, repetition)

            repetition_data = measurement_data[repetition_key]
            if signal_column not in repetition_data:
                raise KeyError(
                    f"Signal column '{signal_column}' not available for measurement '{meas_key}', repetition '{repetition_key}'."
                )

            entry = repetition_data[signal_column]
            lag_entry = None
            if lag_results is not None:
                try:
                    lag_meas_key = self._resolve_measurement_key(
                        lag_results, measurement
                    )
                    lag_measurement = lag_results.get(lag_meas_key, {})
                    lag_rep_key = self._resolve_repetition_name(
                        lag_measurement, repetition_key
                    )
                    lag_repetition = lag_measurement.get(lag_rep_key, {})
                    lag_entry = lag_repetition.get(signal_column)
                except KeyError:
                    lag_entry = None

            coefficients = entry["coefficients"]
            scales = entry["scales"]
            sampling_period = entry["sampling_period"]
            signal = entry["signal"]

            times = np.arange(signal.size) * sampling_period
            power = np.abs(coefficients) ** 2
            with np.errstate(divide="ignore"):
                power_log = np.log10(np.maximum(power, 1e-20))

            if ax is None:
                _, ax = plt.subplots()

            extent = [times[0], times[-1], scales[-1], scales[0]]
            im = ax.imshow(
                power_log,
                extent=extent,
                aspect="auto",
                cmap=cmap,
                origin="lower",
            )

            ax.set_xlabel("Time")
            ax.set_ylabel("Scale")

            if log_scale:
                ax.set_yscale("log")

            ax.set_title(f"{signal_column} - {repetition_key} (Measurement {meas_key})")

            ax_right = None

            def _scale_to_lag(scale_values):
                scale_values = np.asarray(scale_values, dtype=float)
                freq = pywt.scale2frequency(entry["wavelet"], scale_values) / sampling_period
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where(freq > 0, 1.0 / freq, np.nan)

            if show_lag_axis:
                ax_right = ax.twinx()
                ax_right.set_ylim(ax.get_ylim())
                ax_right.set_yscale(ax.get_yscale())
                ax_right.set_ylabel("Lag [s]", labelpad=14)

                scale_ticks = np.asarray(ax.get_yticks(), dtype=float)
                scale_ticks = scale_ticks[scale_ticks > 0]
                lag_values = _scale_to_lag(scale_ticks)
                def _format_power_of_ten(val):
                    if not np.isfinite(val) or val <= 0:
                        return ""
                    exponent = int(np.round(np.log10(val)))
                    return rf"$10^{{{exponent}}}$"

                labels = [_format_power_of_ten(val) for val in lag_values]
                ax_right.set_yticks(scale_ticks)
                ax_right.set_yticklabels(labels)


            if colorbar:
                fig = ax.figure
                cbar_axes = [ax_right, ax] if ax_right is not None else ax
                cbar = fig.colorbar(im, ax=cbar_axes, pad=0.08, fraction=0.03)
                cbar.set_label(r"$\log_{10}(|\mathrm{Coefficient}|^2)$")

            return ax
            
        
        def plot_sq(
            self,
            q,
            intensity,
            intensity_err=None,
            logx=False,
            logy=False,
            ax=None,
            label=None,
            autoscale_limits=True,
            pad_frac=0.05,
            fit=None,
            fit_style=None,
            scale_parameters=None,
            scale_err=None,
            scale_label="Negative binomial dispersion",
            scale_scale=1.0,
            cv_parameters=None,
            cv_err=None,
            cv_label="Coefficient of variation",
            cv_scale=1.0,
            **kwargs,
        ):
            """Plot S(q) scaled intensity versus scattering vector.

            Parameters
            ----------
            q : array-like
                Scattering vector values (e.g., output of ``sq()[2]``).
            intensity : array-like
                Scaled intensity values (e.g., output of ``sq()[0]``).
            intensity_err : array-like or None, optional
                Standard errors for the intensities (e.g., output of ``sq()[1]``).
            logx : bool, optional
                Plot the x-axis on a logarithmic scale (default ``False``).
            logy : bool, optional
                Plot the y-axis on a logarithmic scale (default ``False``).
            ax : matplotlib.axes.Axes, optional
                Existing axes to plot on. If ``None`` a new figure and axes are created.
            label : str, optional
                Series label for legend entries.
            autoscale_limits : bool, optional
                If ``True``, tighten x-limits to finite data (respecting log/linear)
                with a small padding (default ``True``).
            pad_frac : float, optional
                Fractional padding applied to the data span when ``autoscale_limits``
                is enabled (default ``0.05``).
            fit : dict or None, optional
                Fit result as returned by ``sq_fit``. When provided, overlays the
                fitted line over the plotted data.
            fit_style : dict or None, optional
                Styling overrides for the fit line (passed to ``ax.plot``).
            scale_parameters : array-like or None, optional
                Negative binomial dispersion sizes (e.g., from ``sq``) to plot on an adjacent subplot.
            scale_err : array-like or None, optional
                Standard errors for ``scale_parameters`` (fifth output of ``sq``).
            scale_label : str, optional
                Label for the dispersion subplot (default ``"Negative binomial dispersion"``).
            scale_scale : float, optional
                Scaling factor applied to the dispersion values before plotting.
            cv_parameters : array-like or None, optional
                Coefficient of variation values (sixth output of ``sq``) to plot on a third subplot.
            cv_err : array-like or None, optional
                Standard deviations for the CV values (seventh output of ``sq``).
            cv_label : str, optional
                Label for the CV subplot (default ``"Coefficient of variation"``).
            cv_scale : float, optional
                Scaling factor applied to the CV values before plotting.
            **kwargs : dict
                Additional keyword arguments passed to ``Axes.errorbar`` or ``Axes.plot``.

            Returns
            -------
            matplotlib.axes.Axes or tuple of Axes
                The primary axes (and secondary subplot if ``scale_parameters`` is provided).
            """
            q_arr = np.asarray(q, dtype=float)
            i_arr = np.asarray(intensity, dtype=float)
            err_arr = (
                np.asarray(intensity_err, dtype=float)
                if intensity_err is not None
                else None
            )

            finite_mask = np.isfinite(q_arr) & np.isfinite(i_arr)
            if logx:
                finite_mask &= q_arr > 0
            if err_arr is not None:
                finite_mask &= np.isfinite(err_arr)

            if not np.any(finite_mask):
                if ax is None:
                    _, ax = plt.subplots()
                return ax

            q_plot = q_arr[finite_mask]
            i_plot = i_arr[finite_mask]
            err_plot = err_arr[finite_mask] if err_arr is not None else None

            extra_keys = []
            if scale_parameters is not None:
                extra_keys.append("scale")
            if cv_parameters is not None:
                extra_keys.append("cv")

            extra_axes = {}
            if ax is None:
                if extra_keys:
                    width = [3] + [1.5] * len(extra_keys)
                    fig, axes = plt.subplots(
                        1,
                        1 + len(extra_keys),
                        figsize=(10 + 2 * (len(extra_keys) - 1), 4.5),
                        width_ratios=width,
                        sharex=True,
                    )
                    ax = axes[0]
                    for key, axis in zip(extra_keys, axes[1:]):
                        extra_axes[key] = axis
                else:
                    fig, ax = plt.subplots()
            else:
                if isinstance(ax, (list, tuple)):
                    axes_list = list(ax)
                    if len(axes_list) < 1 + len(extra_keys):
                        raise ValueError(
                            "Not enough axes provided for the requested subplots."
                        )
                    base_ax = axes_list[0]
                    for key, axis in zip(extra_keys, axes_list[1:]):
                        extra_axes[key] = axis
                    ax = base_ax
                else:
                    if extra_keys:
                        raise ValueError(
                            "Pass (ax_main, ax_scale, ax_cv) when supplying custom axes."
                        )

            data_handle = None
            if err_plot is not None:
                ax.errorbar(
                    q_plot,
                    i_plot,
                    yerr=err_plot,
                    fmt=kwargs.pop("fmt", "o"),
                    label=label,
                    **kwargs,
                )
            else:
                fmt = kwargs.pop("fmt", "o")
                data_handle = ax.plot(q_plot, i_plot, fmt, label=label, **kwargs)[0]

            ax.set_xlabel("Scattering vector q [1/nm]")
            ax.set_ylabel("Scaled intensity (S(q))")

            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")

            fit_handle = None
            if fit is not None:
                fit_kwargs = {
                    "color": kwargs.get("color", None)
                    if kwargs.get("color", None) is not None
                    else (data_handle.get_color() if data_handle else None),
                    "linestyle": "--",
                }
                slope = fit.get("slope")
                intercept = fit.get("intercept")
                if (
                    slope is not None
                    and intercept is not None
                    and np.isfinite(slope)
                    and np.isfinite(intercept)
                ):
                    fit_kwargs.update(fit_style or {})
                    if "label" not in fit_kwargs or fit_kwargs.get("label") is None:
                        if label:
                            fit_kwargs["label"] = f"{label} fit (m={slope:.3g})"
                        else:
                            fit_kwargs["label"] = f"fit (m={slope:.3g})"

                    q_fit_line = np.linspace(np.min(q_plot), np.max(q_plot), 100)
                    log_q_line = np.log10(q_fit_line)
                    log_i_line = slope * log_q_line + intercept
                    i_fit_line = 10 ** log_i_line
                    fit_handle = ax.plot(q_fit_line, i_fit_line, **fit_kwargs)[0]

            if autoscale_limits:
                q_min, q_max = np.min(q_plot), np.max(q_plot)
                if logx:
                    span = np.log10(q_max) - np.log10(q_min) if q_min > 0 else 1.0
                    pad = pad_frac * span
                    ax.set_xlim(
                        10 ** (np.log10(q_min) - pad),
                        10 ** (np.log10(q_max) + pad),
                    )
                else:
                    span = q_max - q_min
                    pad = pad_frac * span if span > 0 else abs(q_min) * pad_frac
                    ax.set_xlim(q_min - pad, q_max + pad)

            shape_line = None
            cv_line = None
            if scale_parameters is not None:
                scale_values = np.asarray(scale_parameters, dtype=float)[finite_mask]
                if scale_values.size and np.any(np.isfinite(scale_values)):
                    scale_ax = extra_axes.get("scale")
                    if scale_ax is None:
                        raise ValueError(
                            "Scale parameters requested but no axis provided/created."
                        )
                    scaled_vals = scale_values * scale_scale
                    scale_ax.set_xlabel(ax.get_xlabel())
                    scale_ax.set_ylabel(scale_label)
                    scale_ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
                    if scale_err is not None:
                        err_vals = np.asarray(scale_err, dtype=float)[finite_mask] * scale_scale
                        shape_line = scale_ax.errorbar(
                            q_plot,
                            scaled_vals,
                            yerr=err_vals,
                            fmt="s-",
                            color="tab:orange",
                            markersize=4,
                            label=scale_label,
                        )
                    else:
                        shape_line = scale_ax.plot(
                            q_plot,
                            scaled_vals,
                            color="tab:orange",
                            linestyle="-",
                            marker="s",
                            markersize=4,
                            label=scale_label,
                        )[0]

            if cv_parameters is not None:
                cv_values = np.asarray(cv_parameters, dtype=float)[finite_mask]
                if cv_values.size and np.any(np.isfinite(cv_values)):
                    cv_ax = extra_axes.get("cv")
                    if cv_ax is None:
                        raise ValueError(
                            "CV parameters requested but no axis provided/created."
                        )
                    scaled_cv = cv_values * cv_scale
                    cv_ax.set_xlabel(ax.get_xlabel())
                    cv_ax.set_ylabel(cv_label)
                    cv_ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
                    if cv_err is not None:
                        err_cv = np.asarray(cv_err, dtype=float)[finite_mask] * cv_scale
                        cv_line = cv_ax.errorbar(
                            q_plot,
                            scaled_cv,
                            yerr=err_cv,
                            fmt="d-",
                            color="tab:green",
                            markersize=4,
                            label=cv_label,
                        )
                    else:
                        cv_line = cv_ax.plot(
                            q_plot,
                            scaled_cv,
                            color="tab:green",
                            linestyle="-",
                            marker="d",
                            markersize=4,
                            label=cv_label,
                        )[0]

            handles, labels = ax.get_legend_handles_labels()
            if shape_line is not None:
                handles.append(shape_line)
                labels.append(scale_label)
            if cv_line is not None:
                handles.append(cv_line)
                labels.append(cv_label)
            elif fit_handle and fit_handle.get_label() and fit_handle not in handles:
                handles.append(fit_handle)
                labels.append(fit_handle.get_label())

            if handles:
                ax.legend(handles, labels)

            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

            # Secondary x-axis for real-space length
            secax = ax.secondary_xaxis(
                "top", functions=(lambda q: self._real_space_length(q), lambda L: self._inverse_real_space_length(L))
            )
            secax.set_xlabel("spatial scale [nm]")

            if extra_keys:
                ordered_axes = [ax]
                for key in extra_keys:
                    ordered_axes.append(extra_axes.get(key))
                return tuple(ordered_axes)
            return ax
            
        
        def sq_fit(self, q, intensity, intensity_err=None, q_range=None):
            """Fit log10(intensity) vs log10(q) with a straight line over a q-range.

            Parameters
            ----------
            q : array-like
                Scattering vector values.
            intensity : array-like
                Scaled intensities corresponding to ``q``.
            intensity_err : array-like or None, optional
                Standard errors for the intensities; if provided, uses weighted
                least squares with log-domain uncertainties propagated from
                ``d(log10 I) = dI / (I * ln 10)`` (default ``None``).
            q_range : tuple(float, float) or None, optional
                Inclusive q-range (min, max) to restrict the fit. If ``None``,
                all valid points are used.

            Returns
            -------
            dict
                Dictionary with keys:
                    - ``slope``: slope of log-log fit.
                    - ``intercept``: intercept of log-log fit.
                    - ``r2``: coefficient of determination (nan if <2 points).
                    - ``q_used``: q values included in the fit.
                    - ``intensity_used``: intensity values included in the fit.
            """
            q_arr = np.asarray(q, dtype=float)
            i_arr = np.asarray(intensity, dtype=float)
            err_arr = (
                np.asarray(intensity_err, dtype=float)
                if intensity_err is not None
                else None
            )

            mask = np.isfinite(q_arr) & np.isfinite(i_arr) & (q_arr > 0) & (i_arr > 0)
            if err_arr is not None:
                mask &= np.isfinite(err_arr) & (err_arr >= 0)

            if q_range is not None:
                q_min, q_max = q_range
                mask &= (q_arr >= q_min) & (q_arr <= q_max)

            if not np.any(mask):
                raise ValueError("No finite, positive data points available for fit.")

            q_fit = q_arr[mask]
            i_fit = i_arr[mask]
            log_q = np.log10(q_fit)
            log_i = np.log10(i_fit)

            weights = None
            if err_arr is not None:
                err_fit = err_arr[mask]
                sigma_log = err_fit / (i_fit * np.log(10))
                # Avoid zero weights; mask invalid propagated errors.
                valid_sigma = np.isfinite(sigma_log) & (sigma_log > 0)
                if np.any(valid_sigma):
                    log_q = log_q[valid_sigma]
                    log_i = log_i[valid_sigma]
                    sigma_log = sigma_log[valid_sigma]
                    q_fit = q_fit[valid_sigma]
                    i_fit = i_fit[valid_sigma]
                    weights = 1.0 / sigma_log

            if log_q.size < 2:
                raise ValueError("At least two valid points are required for fitting.")

            if weights is not None:
                coeffs = np.polyfit(log_q, log_i, 1, w=weights)
            else:
                coeffs = np.polyfit(log_q, log_i, 1)

            slope, intercept = coeffs[0], coeffs[1]
            fit_vals = slope * log_q + intercept
            ss_res = np.sum((log_i - fit_vals) ** 2)
            ss_tot = np.sum((log_i - np.mean(log_i)) ** 2)
            r2 = 1 - ss_res / ss_tot if log_q.size > 1 and ss_tot != 0 else np.nan

            return {
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "q_used": q_fit,
                "intensity_used": i_fit,
            }


            
        def _concatenate_measurement_wavelets(self, measurement_data):
            """
            Concatenate signals for all repetitions in a measurement and recompute CWT.

            Parameters
            ----------
            measurement_data : dict
                Nested dict of wavelet entries for each repetition.

            Returns
            -------
            dict
                Measurement data with a single ``Concatenated`` repetition containing
                recomputed coefficients for each signal column.
            """
            aggregated = {}
            for repetition_data in measurement_data.values():
                for column, entry in repetition_data.items():
                    if "signal" not in entry or entry["signal"] is None:
                        continue
                    aggregated.setdefault(column, []).append(entry)

            concatenated_channels = {}
            for column, entries in aggregated.items():
                signals = [np.asarray(e["signal"], dtype=float) for e in entries if e["signal"] is not None]
                if not signals:
                    continue
                signal = np.concatenate(signals)

                sampling_periods = [float(e.get("sampling_period", 1.0)) for e in entries]
                sampling_period = sampling_periods[0]
                if not np.allclose(sampling_periods, sampling_period):
                    sampling_period = float(np.mean(sampling_periods))

                wavelet = entries[0].get("wavelet", "morl")
                scales = entries[0].get("scales")
                if scales is None or len(scales) == 0:
                    max_scale = max(2, min(signal.size // 2, 256))
                    scales_arr = np.linspace(1, max_scale, num=max_scale, endpoint=True)
                else:
                    scales_arr = np.asarray(scales, dtype=float)

                coefficients, frequencies = pywt.cwt(
                    signal, scales_arr, wavelet, sampling_period=sampling_period
                )

                concatenated_channels[column] = {
                    "signal_column": column,
                    "signal": signal,
                    "sampling_period": sampling_period,
                    "scales": scales_arr,
                    "wavelet": wavelet,
                    "coefficients": coefficients,
                    "frequencies": frequencies,
                }

            if not concatenated_channels:
                return {}

            return {self.CONCATENATED_REPETITION_NAME: concatenated_channels}

        def _resolve_measurement_key(self, data_dict, requested):
            """Return the canonical measurement key matching the requested identifier."""
            if requested in data_dict:
                return requested

            str_request = str(requested)
            for key in data_dict:
                if str(key) == str_request:
                    return key

            match_req = re.search(r"(\d+)", str_request)
            if match_req:
                req_num = int(match_req.group(1))
                for key in data_dict:
                    match_key = re.search(r"(\d+)", str(key))
                    if match_key and int(match_key.group(1)) == req_num:
                        return key

            raise KeyError(f"Measurement '{requested}' not found.")

        def _resolve_repetition_name(self, measurement_data, requested):
            """
            Normalize repetition identifiers (e.g., 1 -> 'Repetition 1').
            """
            if requested in measurement_data:
                return requested

            for rep_key in measurement_data:
                if str(rep_key) == str(requested):
                    return rep_key

            try:
                req_num = int(re.search(r"(\d+)", str(requested)).group(1))
            except Exception:
                raise KeyError(
                    f"Repetition '{requested}' not found in measurement data."
                )

            for rep_key in measurement_data:
                match = re.search(r"(\d+)", str(rep_key))
                if match and int(match.group(1)) == req_num:
                    return rep_key

            raise KeyError(f"Repetition '{requested}' not found in measurement data.")

        def _fit_gamma_from_values(
            self,
            values,
            count_bins=None,
            max_value=None,
            initial_shift=None,
            fit_shift=True,
            shift_grid=64,
        ):
            """
            Fit Gamma parameters directly from a 1D array of samples, optionally
            optimising a baseline shift.
            """
            samples = np.asarray(values, dtype=float)
            samples = samples[np.isfinite(samples)]
            if max_value is not None:
                samples = np.clip(samples, None, float(max_value))
            samples = samples[samples >= 0]
            if samples.size == 0:
                raise ValueError("No valid samples available for Gamma fitting.")

            raw_mean = float(np.mean(samples))
            raw_var = float(np.var(samples))
            raw_var = max(raw_var, 1e-12)
            raw_scale = raw_var / raw_mean if raw_mean > 0 else np.nan

            max_shift = float(samples.min())
            shift = 0.0
            if initial_shift is not None:
                shift = min(max(float(initial_shift), 0.0), max_shift)
                fit_shift = False

            best = None
            residual_mean = None
            if fit_shift and max_shift > 0:
                grid = np.linspace(0.0, max_shift, int(max(2, shift_grid)))
                best_score = -np.inf
                for candidate in grid:
                    residual = samples - candidate
                    residual = residual[residual > 0]
                    if residual.size < 2:
                        continue
                    r_mean = float(np.mean(residual))
                    if r_mean <= 0:
                        continue
                    r_var = float(np.var(residual))
                    r_var = max(r_var, 1e-12)
                    shape = r_mean**2 / r_var
                    scale = r_var / r_mean
                    ll = self._gamma_log_likelihood(residual, shape, scale)
                    if ll > best_score:
                        best_score = ll
                        best = (candidate, shape, scale, r_mean)

                if best is not None:
                    shift, shape, scale, residual_mean = best
                else:
                    fit_shift = False

            if not fit_shift:
                residual = samples - shift
                residual = residual[residual > 0]
                if residual.size == 0:
                    residual = samples
                    shift = min(shift, samples.min())
                residual_mean = float(np.mean(residual))
                if residual_mean <= 0:
                    raise ValueError("Gamma mean must be positive.")
                residual_var = float(np.var(residual))
                residual_var = max(residual_var, 1e-12)
                shape = residual_mean**2 / residual_var
                scale = residual_var / residual_mean

            hist_data = None
            if count_bins and count_bins > 1:
                count_bins = max(int(count_bins), 2)
                vmin, vmax = samples.min(), samples.max()
                if np.isclose(vmin, vmax):
                    vmin = max(0.0, vmin - 0.5)
                    vmax = vmax + 0.5
                edges = np.linspace(vmin, vmax, count_bins + 1)
                density, edges = np.histogram(samples, bins=edges, density=True)
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                hist_data = {
                    "bin_centers": bin_centers,
                    "density": density,
                    "edges": edges,
                }

            vmin, vmax = samples.min(), samples.max()
            if np.isclose(vmin, vmax):
                vmin = max(0.0, vmin - 0.5)
                vmax = vmax + 0.5
            num_points = max(200, samples.size)
            eval_points = np.linspace(vmin, vmax, num_points)
            residual_points = np.maximum(eval_points - shift, 1e-12)
            log_pdf = (
                (shape - 1) * np.log(residual_points)
                - (residual_points / scale)
                - shape * np.log(scale)
                - math.lgamma(shape)
            )
            gamma_pdf = np.exp(log_pdf)
            gamma_pdf[eval_points < shift] = 0.0

            mean = float(residual_mean + shift)

            return {
                "shape": shape,
                "scale": scale,
                "raw_mean": raw_mean,
                "raw_scale": raw_scale,
                "mean": mean,
                "shift": shift,
                "histogram": hist_data,
                "gamma": {
                    "x": eval_points,
                    "pdf": gamma_pdf,
                },
            }

        def _fit_negative_binomial_from_values(self, values):
            """
            Fit a negative binomial distribution from discrete count samples.
            """
            counts = np.asarray(values, dtype=float)
            counts = counts[np.isfinite(counts)]
            counts = counts[counts >= 0]
            if counts.size == 0:
                raise ValueError("No valid samples available for negative binomial fitting.")

            counts = np.round(counts).astype(int)
            counts = counts[counts >= 0]
            if counts.size == 0:
                raise ValueError("No valid integer samples available for negative binomial fitting.")

            mean = float(np.mean(counts))
            var = float(np.var(counts))
            if mean <= 0:
                raise ValueError("Negative binomial mean must be positive.")

            if var <= mean:
                size = np.inf
                prob = 1.0
            else:
                size = mean**2 / (var - mean)
                prob = size / (size + mean)

            return {
                "mean": mean,
                "variance": var,
                "size": size,
                "prob": prob,
            }

        def _negative_binomial_pmf(self, k_values, size, prob, mean):
            k_values = np.asarray(k_values, dtype=float)
            pmf = np.zeros_like(k_values, dtype=float)
            if not np.isfinite(size) or size <= 0 or prob <= 0 or prob >= 1:
                # Poisson limit when size -> inf and prob -> 1 keeping mean fixed.
                mean = max(float(mean), 0.0)
                pmf = np.exp(-mean) * np.power(mean, k_values) / np.maximum(
                    1.0, np.vectorize(math.factorial)(k_values.astype(int))
                )
                return pmf

            if gammaln is None:
                comb = np.vectorize(math.comb)
                coeff = comb((k_values + size - 1).astype(int), k_values.astype(int))
                pmf = coeff * (1 - prob) ** k_values * prob**size
            else:
                log_pmf = (
                    gammaln(k_values + size)
                    - gammaln(size)
                    - gammaln(k_values + 1)
                    + size * np.log(prob)
                    + k_values * np.log(1 - prob)
                )
                pmf = np.exp(log_pmf)
            return pmf
        
        def _scale_to_lag_value(self, scale, wavelet, sampling_period):
            if scale is None or not np.isfinite(scale):
                return np.nan
            try:
                freq = pywt.scale2frequency(wavelet, scale) / sampling_period
            except Exception:
                return np.nan
            return np.inf if freq <= 0 else 1.0 / freq

        def _estimate_lag_uncertainty(self, entry, peak_scale, drop_ratio):
            powers = np.asarray(entry.get("average_power"))
            scales = np.asarray(entry.get("scales"))
            wavelet = entry.get("wavelet", "morl")
            sampling_period = float(entry.get("sampling_period", 1.0))

            if (
                peak_scale is None
                or powers.size == 0
                or scales.size == 0
                or powers.size != scales.size
            ):
                return np.nan

            idx = np.argmin(np.abs(scales - peak_scale))
            target = powers[idx] * drop_ratio

            left = idx
            while left > 0 and powers[left] > target:
                left -= 1
            right = idx
            while right < len(powers) - 1 and powers[right] > target:
                right += 1

            lag_peak = self._scale_to_lag_value(peak_scale, wavelet, sampling_period)
            lag_low = self._scale_to_lag_value(scales[left], wavelet, sampling_period)
            lag_high = self._scale_to_lag_value(
                scales[right], wavelet, sampling_period
            )

            deltas = [
                lag_peak - lag_low if np.isfinite(lag_low) else np.nan,
                lag_high - lag_peak if np.isfinite(lag_high) else np.nan,
            ]
            finite_deltas = [d for d in deltas if np.isfinite(d)]
            if not finite_deltas:
                return np.nan
            return float(max(finite_deltas))

        def _gamma_log_likelihood(self, samples, shape, scale):
            if shape <= 0 or scale <= 0:
                return -np.inf
            samples = np.asarray(samples, dtype=float)
            samples = samples[samples > 0]
            if samples.size == 0:
                return -np.inf
            log_pdf = (
                (shape - 1) * np.log(samples)
                - (samples / scale)
                - shape * np.log(scale)
                - math.lgamma(shape)
            )
            return float(np.sum(log_pdf))
