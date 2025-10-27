import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
from ibicus.debias import QuantileMapping, CDFt, DeltaChange, QuantileDeltaMapping, ScaledDistributionMapping, LinearScaling, ISIMIP, ECDFM
from ibicus.evaluate.metrics import *
import data.valid_crd as valid_crd

##### Needs Refactoring from Unit Manager; REFACTOR reference loading
class BiasCorrectionBenchmark:
    def __init__(self, clim, ref, hist_period, test_period,
                 scenario, clim_var, correction_methods, model_path, test_path):
        """
        A class to manage and benchmark multiple bias correction methods.

        Parameters:
            hist_period (list): [start, end] years for historical period.
            future_period (list): [start, end] years for future period.
            scenario (str): Climate scenario (e.g., 'ssp5_8_5').
            clim_var (str): Climate variable (e.g., 'pr').
            correction_methods (list): List of bias correction methods to apply.
            dataset_path (str): Path to dataset directory.
            exp_path (str): Path where processed data is stored.
            trend_analysis (bool): Whether to perform trend analysis.
        """
        self.clim = clim
        self.ref = ref
        self.hist_period = hist_period
        self.test_period = test_period
        self.scenario = scenario
        self.clim_var = clim_var
        self.correction_methods = correction_methods
        self.model_path = model_path
        self.test_path = test_path

        # Time series setup
        self.ref_time = pd.date_range(start=f"{self.hist_period[0]}-01-01", end=f"{self.hist_period[1]}-12-31", freq="D").to_numpy()

        # Load data
        self.load_data()

    def load_data(self):
        """Loads the historical and future climate model data and reference dataset."""
        print("Loading dataset for benchmarking...")
        self.hist_time =  torch.load(f'{self.model_path}/time.pt', weights_only=False)
        self.hist_model = torch.load(f'{self.model_path}/x.pt', map_location='cpu', weights_only=True).numpy()

        self.test_time =  torch.load(f'{self.test_path}/time.pt', weights_only=False)
        self.test_model = torch.load(f'{self.test_path}/x.pt', map_location='cpu', weights_only=True).numpy()

        self.reference = torch.load(f'{self.model_path}/y.pt', map_location='cpu', weights_only=True).numpy()

        ## needed for perfect model approach
        if os.path.exists(f'{self.model_path}/time_y.pt'):
            self.ref_time = torch.load(f'{self.model_path}/time_y.pt', weights_only=True)


        ## may need to refactor better
        if self.clim_var == 'pr': ##  mm/day to mm/s or kg/m2/s
            self.hist_model = self.hist_model / 86400 
            self.test_model = self.test_model / 86400
            self.reference = self.reference / 86400 



    def apply_correction(self):
        """Applies multiple bias correction methods and saves results."""
        for method in self.correction_methods:
            print(f"Applying bias correction: {method}")

            save_path = f'benchmark/{method}/conus/{self.clim}-{self.ref}'
            os.makedirs(save_path, exist_ok=True)
            save_file = f'{save_path}/{self.hist_period}_{self.scenario}_{self.test_period}.pt'

            if os.path.exists(save_file):
                print(f"Debiased data for {method} already exists at {save_file}, skipping...")
                continue

            # Select bias correction method
            if method == "QuantileMapping":
                debiaser = QuantileMapping.from_variable(self.clim_var, mapping_type="parametric")
            elif method == "CDFt":
                debiaser = CDFt.from_variable(self.clim_var)
            elif method == "DeltaChange":
                debiaser = DeltaChange.from_variable(self.clim_var)
            elif method == "QuantileDeltaMapping":
                debiaser = QuantileDeltaMapping.from_variable(self.clim_var)
            elif method == "ScaledDistributionMapping":
                debiaser = ScaledDistributionMapping.from_variable(self.clim_var)
            elif method == "LinearScaling": 
                debiaser = LinearScaling.from_variable(self.clim_var)
            elif method == "ISIMIP":
                debiaser = ISIMIP.from_variable(self.clim_var)
            elif method == "ECDFM":
                debiaser = ECDFM.from_variable(self.clim_var)
            else:
                print(f"Unknown method: {method}, skipping...")
                continue

            # Apply correction
            debiased_future = debiaser.apply(
                self.reference, self.hist_model, self.test_model,
                time_obs=self.ref_time, time_cm_hist=self.hist_time, time_cm_future=self.test_time
            )

    
            # Save results        
            torch.save(debiased_future, save_file)
            print(f"Saved debiased data to {save_file}")
