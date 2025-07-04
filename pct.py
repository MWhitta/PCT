import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data():
    """ object for holding data, metadata, and analysis
    """



class Dataloader():
    """ load PCT data and structure it for processing
    """

    def __init__(self, path):
        """ stuff happens
        """
        self.path = path

    def load(self, ext='.csv'):
        """ load data and store in a Data object
        """
        data = Data()
        data.path = self.path
        data.ext = ext
        data.files = [f for f in os.listdir(self.path) if f.endswith(ext)]
        data.data = []
        
        for file in data.files:
            file_path = os.path.join(self.path, file)
            file_data = pd.read_csv(file_path, delimiter=',', skiprows=3)
            

        return data

