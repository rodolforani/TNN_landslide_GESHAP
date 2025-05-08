import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import os
from scipy import signal

class readTransformerData:
    def __init__(self, dataprepinargs):
        self.dataparam = dataprepinargs

    def readfilesTerrain(self):
        covardir = self.dataparam["covar"]
        responsedir = self.dataparam["response"]
        self.covar = np.load(covardir)
        self.response = np.expand_dims(np.load(responsedir), axis=-1)
        self.time = self.response

    def readfiles(self):
        timedir = self.dataparam["time"]
        covardir = self.dataparam["covar"]
        responsedir = self.dataparam["response"]
        self.time = np.expand_dims(np.load(timedir), axis=-1)
        self.covar = np.load(covardir)
        self.response = np.expand_dims(np.load(responsedir), axis=-1)
    
    def preparedata(self):
        
        terrain = self.dataparam.get("terrain", False)
        
        if terrain:
            self.readfilesTerrain()
        else:
            self.readfiles()

        # if self.dataparam["removezeros"]:
        #     idx = np.where(self.response > 0)[0]
        #     self.time = self.time[idx]
        #     self.covar = self.covar[idx]


        # # Remove samples where all values in self.time are zero
        # if self.dataparam["removezeros"]:
        #     idx = np.where(np.any(self.time != 0, axis=(1, 2)))[0]
        #     self.time = self.time[idx]
        #     self.covar = self.covar[idx]
        #     self.response = self.response[idx]


        # Remove samples where all values in specified columns of self.time are zero
        if self.dataparam["removezeros"]:
            # Compute the mean of each column (time step)
            mean_per_column = np.mean(self.time, axis=0).squeeze()
            # Compute the percentile of these means
            overall_perc = np.percentile(mean_per_column, 75)
            # Identify columns where the mean is greater than the overall mean
            relevant_columns = mean_per_column > overall_perc
            
            # Print debug information
            print("Mean per column (shape):", mean_per_column.shape)
            print(mean_per_column)
            print("Overall mean:", overall_perc)
            print("Relevant columns (shape):", relevant_columns.shape)
            print(relevant_columns)
            
            #  Keep only rows where all values are non-zero in the relevant columns = remove SU with at least one value = 0
            idx = np.where(np.all(self.time[:, relevant_columns, :] != 0, axis=(1, 2)))[0]
            # Keep only rows where there is at least one non-zero value in the relevant columns = remove SU with all values are 0
            # idx = np.where(np.any(self.time[:, relevant_columns, :] != 0, axis=(1, 2)))[0]

            self.time = self.time[idx]
            self.covar = self.covar[idx]
            self.response = self.response[idx]

        if self.dataparam["inference"]:
            self.Xt = self.time
            self.Xc = self.covar
            self.Y = self.response
        else:
            (
                self.Xt_train,
                self.Xt_test,
                self.Xc_train,
                self.Xc_test,
                self.Y_train,
                self.Y_test,
            ) = train_test_split(
                self.time,
                self.covar,
                self.response,
                test_size=self.dataparam["testsize"],
                random_state=420,
            )
        self.covars = None
        self.time = None
        self.response = None