import os
from typing import Any, Dict

import pandas as pd


def load_dataset(root_path: str, extension: str = ".csv") -> Dict[str, Dict[str, Any]]:
    """Load CSV files in a sample/experiment/measurement/repetition hierarchy.

    Parameters
    ----------
    root_path : str
        Directory containing sample folders.
    extension : str, optional
        CSV file extension to search for (default ".csv").

    Returns
    -------
    dict
        Nested dictionary mapping ``sample -> experiment -> measurement`` where
        each measurement contains separate dictionaries for the file types
        ``CORENN Gamma results.csv``, ``Correlation Function.csv`` and
        ``Count Trace.csv``.  Within each file type dictionary the repetitions
        are loaded as :class:`pandas.DataFrame` instances.  Experiment level
        information is stored under the key ``"info"`` and measurement summaries
        are stored under ``"summary"``.
    """
    dataset: Dict[str, Dict[str, Any]] = {}

    for sample in sorted(os.listdir(root_path)):
        sample_path = os.path.join(root_path, sample)
        if not os.path.isdir(sample_path):
            continue

        sample_dict: Dict[str, Any] = {"experiments": {}}

        for experiment in sorted(os.listdir(sample_path)):
            exp_path = os.path.join(sample_path, experiment)
            if not os.path.isdir(exp_path):
                continue

            exp_dict: Dict[str, Any] = {"info": None, "measurements": {}}

            exp_csvs = [f for f in os.listdir(exp_path) if f.endswith(extension)]
            if exp_csvs:
                info_path = os.path.join(exp_path, exp_csvs[0])
                try:
                    exp_dict["info"] = pd.read_csv(info_path)
                except Exception:
                    exp_dict["info"] = None

            for measurement in sorted(os.listdir(exp_path)):
                meas_path = os.path.join(exp_path, measurement)
                if not os.path.isdir(meas_path):
                    continue

                meas_dict: Dict[str, Any] = {
                    "summary": None,
                    "gamma_results": {},
                    "correlation_function": {},
                    "count_trace": {},
                }
                summary_path = os.path.join(meas_path, f"Summary{extension}")
                if os.path.exists(summary_path):
                    try:
                        meas_dict["summary"] = pd.read_csv(summary_path)
                    except Exception:
                        meas_dict["summary"] = None

                file_types = {
                    f"CORENN Gamma results{extension}": "gamma_results",
                    f"Correlation Function{extension}": "correlation_function",
                    f"Count Trace{extension}": "count_trace",
                }

                for rep_file in sorted(os.listdir(meas_path)):
                    if rep_file == f"Summary{extension}" or not rep_file.endswith(extension):
                        continue

                    rep_path = os.path.join(meas_path, rep_file)
                    for suffix, key in file_types.items():
                        if rep_file.endswith(suffix):
                            rep_name = rep_file[: -len(suffix)].rstrip(" -_") or rep_file
                            try:
                                meas_dict[key][rep_name] = pd.read_csv(rep_path)
                            except Exception:
                                meas_dict[key][rep_name] = None
                            break

                exp_dict["measurements"][measurement] = meas_dict

            sample_dict["experiments"][experiment] = exp_dict

        dataset[sample] = sample_dict

    return dataset
