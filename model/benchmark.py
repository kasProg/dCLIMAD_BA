import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
from ibicus.debias import QuantileMapping, CDFt, DeltaChange, QuantileDeltaMapping, ScaledDistributionMapping, LinearScaling
from ibicus.evaluate.metrics import *
import data.valid_crd as valid_crd

##### Needs Refactoring from Unit Manager


class BiasCorrectionBenchmark:
    def __init__(self, cmip6_dir, clim, ref, model_type, layers, degree, quantile, hist_period, test_period,
                 scenario, trend_future_period, clim_var, correction_methods, dataset_path, exp_path, trend_analysis=True):
        """
        A class to manage and benchmark multiple bias correction methods.

        Parameters:
            clim (str): Climate model name.
            ref (str): Reference dataset name.
            model_type (str): Model type used in bias correction.
            layers (int): Number of layers in ANN model.
            degree (int): Degree of transformation.
            quantile (float): Target quantile for correction.
            hist_period (list): [start, end] years for historical period.
            future_period (list): [start, end] years for future period.
            scenario (str): Climate scenario (e.g., 'ssp5_8_5').
            trend_future_period (list): [start, end] years for future trend analysis.
            clim_var (str): Climate variable (e.g., 'pr').
            correction_methods (list): List of bias correction methods to apply.
            dataset_path (str): Path to dataset directory.
            exp_path (str): Path where processed data is stored.
            trend_analysis (bool): Whether to perform trend analysis.
        """
        self.cmip6_dir = cmip6_dir
        self.clim = clim
        self.ref = ref
        self.model_type = model_type
        self.layers = layers
        self.degree = degree
        self.quantile = quantile
        self.hist_period = hist_period
        self.test_period = test_period
        self.scenario = scenario
        self.trend_future_period = trend_future_period
        self.clim_var = clim_var
        self.correction_methods = correction_methods
        self.dataset_path = dataset_path
        self.exp_path = exp_path
        self.trend_analysis = trend_analysis

        # Paths
        self.save_path = f'benchmark/{self.clim}-{self.ref}'
        os.makedirs(self.save_path, exist_ok=True)

        # Time series setup
        self.ref_time = pd.date_range(start=f"{self.hist_period[0]}-01-01", end=f"{self.hist_period[1]}-12-31", freq="D").to_numpy()
        # self.future_time = pd.date_range(start=f"{self.future_period[0]}-01-01", end=f"{self.future_period[1]}-12-31", freq="D").to_numpy()
        # self.trend_future_time = pd.date_range(start=f"{self.trend_future_period[0]}-01-01", end=f"{self.trend_future_period[1]}-12-31", freq="D").to_numpy()

        # Load data
        self.load_data()

    def load_data(self):
        """Loads the historical and future climate model data and reference dataset."""
        print("Loading dataset...")
        ds_sample = xr.open_dataset(f"{self.dataset_path}prec.1980.nc")
        valid_coords = valid_crd.valid_lat_lon(ds_sample)


        hist_path = f'{self.cmip6_dir}/{self.clim}/historical/{self.var}/clipped_US.nc'
        test_path = f'{self.cmip6_dir}/{self.clim}/{self.scenario}/{self.var}/clipped_US.nc'

        
        # Load future data
        x_hist = xr.open_dataset(hist_path)
        self.hist_time = x_hist.time.values
        x_hist = x_hist[self.clim_var].sel(
            lat=xr.DataArray(valid_coords[:, 0], dims='points'),
            lon=xr.DataArray(valid_coords[:, 1], dims='points'),
            method='nearest'
        ).sel(time=slice(f'{self.hist_period[0]}', f'{self.hist_period[1]}'))

        x_hist = x_hist.values
        self.hist_model = np.expand_dims(x_hist, axis=-1)

        x_test = xr.open_dataset(test_path)
        self.test_time = x_test.time.values
        x_test = x_test[self.clim_var].sel(
            lat=xr.DataArray(valid_coords[:, 0], dims='points'),
            lon=xr.DataArray(valid_coords[:, 1], dims='points'),
            method='nearest'
        ).sel(time=slice(f'{self.test_period[0]}', f'{self.test_period[1]}'))

        x_test = x_test.values
        self.test_model = np.expand_dims(x_test, axis=-1)


        ## may need to refactor better
        self.reference = torch.load(f'{self.exp_path}/y.pt', weights_only=False).cpu().numpy() / 86400



    def apply_correction(self):
        """Applies multiple bias correction methods and saves results."""
        for method in self.correction_methods:
            print(f"Applying bias correction: {method}")

            # Select bias correction method
            if method == "QuantileMapping":
                debiaser = QuantileMapping.from_variable(self.var, mapping_type="parametric")
            elif method == "CDFt":
                debiaser = CDFt.from_variable(self.var)
            elif method == "DeltaChange":
                debiaser = DeltaChange.from_variable(self.var)
            elif method == "QuantileDeltaMapping":
                debiaser = QuantileDeltaMapping.from_variable(self.var)
            elif method == "ScaledDistributionMapping":
                debiaser = ScaledDistributionMapping.from_variable(self.var)
            elif method == "LinearScaling":
                debiaser = LinearScaling.from_variable(self.var)
            else:
                print(f"Unknown method: {method}, skipping...")
                continue

            # Apply correction
            debiased_future = debiaser.apply(
                self.reference, self.hist_model, self.test_model,
                time_obs=self.ref_time, time_cm_hist=self.hist_time, time_cm_future=self.test_time
            )

            # Save results
            save_file = f'{self.save_path}/{method}_{self.hist_period}__{self.scenario}_{self.test_period}.pt'
            torch.save(debiased_future * 86400, save_file)
            print(f"Saved debiased data to {save_file}")
