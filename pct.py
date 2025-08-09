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


    def remove_trailing_comma(self, filepath):
        with open(filepath, 'r+', encoding='utf-8') as file:
            lines = file.readlines()
            if lines[-1].rstrip().endswith(','):
                lines[-1] = lines[-1].rstrip().rstrip(',') + '\n'
                file.seek(0)
                file.writelines(lines)
                file.truncate()


    def load(self, ext='.csv', delimiter=','):
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
                measurement_dict = {"summary": {}, "repetitions": {}}

                # iterate through repetition folders
                for repetition in sorted(os.listdir(meas_path)):
                    if repetition == "Count Trace.csv":
                        rep_path = os.path.join(meas_path, repetition)
                        ct_df = pd.read_csv(rep_path, skiprows=2)
                        measurement_dict["repetitions"][repetition] = ct_df

                    if repetition == "Correlation Function.csv":
                        rep_path = os.path.join(meas_path, repetition)
                        ct_df = pd.read_csv(rep_path, skiprows=2)
                        measurement_dict["repetitions"][repetition] = ct_df

                    if repetition == "CORENN Gamma Results.csv":
                        rep_path = os.path.join(meas_path, repetition)
                        ct_df = pd.read_csv(rep_path, skiprows=5)
                        measurement_dict["repetitions"][repetition] = ct_df
                
                experiment_dict["measurements"][meas_num] = measurement_dict
            data.experiments[experiment] = experiment_dict

        return data
    

    class Analyzer():
        """
        """
        
        def __init__(self):
            """
            """


        def g20(self, tau, g2, g2err, q):
            """ fit """
            
        
        def analyze(self, data):
            """
            Analyze the loaded data.

            Parameters
            ----------
            data : Data
                The structured data object containing experiments, measurements, and repetitions.
            """
            # Example analysis: Calculate mean and standard deviation of each measurement's summary
            