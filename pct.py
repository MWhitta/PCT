import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data:
    """Object for holding data, metadata, and analysis."""

    def __init__(self):
        self.path = None
        self.ext = None
        # dictionary keyed by experiment name. Each experiment contains
        #   - info: dataframe of experiment information
        #   - measurements: dict keyed by measurement number
        self.experiments = {}



class Dataloader():
    """ load PCT data and structure it for processing
    """

    def __init__(self, path):
        """ stuff happens
        """
        self.path = path

    def load(self, ext='.csv'):
        """Load photon correlation spectroscopy data organised in a dataset/experiment/measurement/repetition
        hierarchy.

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

        # iterate through experiments
        for experiment in sorted(os.listdir(self.path)):
            exp_path = os.path.join(self.path, experiment)
            if not os.path.isdir(exp_path):
                continue

            # load experiment level csv (meta data)
            exp_info = None
            exp_csvs = [f for f in os.listdir(exp_path) if f.endswith(ext)]
            if exp_csvs:
                info_path = os.path.join(exp_path, exp_csvs[0])
                try:
                    exp_info = pd.read_csv(
                        info_path, skiprows=3, usecols=["Name", "Correlation Type"]
                    )
                except Exception:
                    exp_info = None

            experiment_dict = {"info": exp_info, "measurements": {}}

            # iterate through measurements in the experiment
            for measurement in sorted(os.listdir(exp_path)):
                meas_path = os.path.join(exp_path, measurement)
                if not os.path.isdir(meas_path):
                    continue

                # parse measurement number from folder name
                match = re.search(r"(\d+)", measurement)
                meas_num = int(match.group(1)) if match else measurement

                # load measurement summary
                summary_path = os.path.join(meas_path, "Summary.csv")
                summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

                measurement_dict = {"summary": summary_df, "repetitions": {}}

                # iterate through repetition folders
                for repetition in sorted(os.listdir(meas_path)):
                    rep_path = os.path.join(meas_path, repetition)
                    if not os.path.isdir(rep_path):
                        continue

                    ct_path = os.path.join(rep_path, "Count Trace.csv")
                    if os.path.exists(ct_path):
                        ct_df = pd.read_csv(ct_path)
                        measurement_dict["repetitions"][repetition] = ct_df

                experiment_dict["measurements"][meas_num] = measurement_dict

            data.experiments[experiment] = experiment_dict

        return data