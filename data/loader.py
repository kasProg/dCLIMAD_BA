import os
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
import valid_crd
import process  # Corrected import

### Some limitations: 
# Loyal to CONUS region, eg files named clipped_US

class DataLoaderWrapper:
    def __init__(self, clim, ref, train_period, test_period, dataset_path, cmip6_dir, 
                 elev_path, input_x, input_attrs, num='all', batch_size=100, 
                 train=False, trend_analysis=False, trend_future_period=None, device=0):
        """
        Customizable climate data loader with future projection support.
        """
        self.clim = clim
        self.ref = ref
        self.train_period = train_period
        self.test_period = test_period
        self.dataset_path = dataset_path
        self.cmip6_dir = cmip6_dir
        self.elev_path = elev_path
        self.input_x = input_x
        self.input_attrs = input_attrs
        self.num = num
        self.batch_size = batch_size
        self.train = train
        self.trend_analysis = trend_analysis
        self.trend_future_period = trend_future_period

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        # Define periods
        self.period = self.train_period if self.train else self.test_period

        # Set save paths
        self.save_path = self.get_save_path()

        # Load data
        self.valid_coords = self.get_valid_coords()
        self.attrs_data = self.load_attrs()
        self.x_data, self.time_x = self.load_dynamic_inputs()
        self.y_data = self.load_y_data()
        self.attr_tensor = self.get_attr_tensor()
        self.input_norm_tensor = self.normalize_data()

        # Future trend data
        if self.trend_analysis:
            self.input_norm_tensor_future, self.x_future = self.load_future_data()

    def get_save_path(self):
        """Generates the save path based on training/testing configuration."""
        base_path = f'jobs/{self.clim}-{self.ref}/QM_ANN_{self.num}/{self.train_period[0]}_{self.train_period[1]}/'
        if not self.train:
            base_path += f'{self.test_period[0]}_{self.test_period[1]}/'
            os.makedirs(base_path + f'ep40', exist_ok=True)
        os.makedirs(base_path, exist_ok=True)
        return base_path

    def get_valid_coords(self):
        """Extracts valid latitude-longitude pairs."""
        ds_sample = xr.open_dataset(f"{self.dataset_path}prec.1980.nc")
        return valid_crd.valid_lat_lon(ds_sample)

    def load_attrs(self):
        """Loads static attributes (elevation, land type, etc.)."""
        attrs_data = {}
        if "elevation" in self.input_attrs:
            elev = xr.open_dataset(self.elev_path)
            attrs_data["elevation"] = elev["elevation"].sel(
                lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                method='nearest'
            ).values
        return attrs_data

    def load_dynamic_inputs(self):
        """Loads dynamic inputs (precipitation, wind, etc.)."""
        x_data = []
        time_x = None

        for var in self.input_x:
            print(f"Processing {var}...")
            ds = xr.open_dataset(f"{self.cmip6_dir}/{self.clim}/historical/{var}/clipped_US.nc")
            x_var = ds[var].sel(
                lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                method='nearest'
            ).sel(time=slice(f"{self.period[0]}", f"{self.period[1]}")).values * 86400
            x_var = torch.tensor(x_var).to(self.device)
            x_data.append(x_var.unsqueeze(-1))
            if time_x is None:
                time_x = ds.time.values

        return torch.cat(x_data, dim=-1), time_x

    def load_y_data(self):
        """Loads reference precipitation data (Livneh or other)."""
        print("Processing y data...")
        y = process.process_data(self.dataset_path, self.period, self.valid_coords, 
                                 self.num, self.device, var="prec")
        return y.to(self.x_data.dtype)

    def get_attr_tensor(self):
        """Creates a tensor for static attributes."""
        attr_tensors = []
        for var in self.input_attrs:
            if var in self.attrs_data:
                attr_tensors.append(torch.tensor(self.attrs_data[var]).to(self.x_data.dtype).to(self.device).unsqueeze(-1))
        return torch.cat(attr_tensors, dim=-1)

    def normalize_data(self):
        """Normalizes input data."""
        print("Normalizing data...")
        statDict = process.getStatDic(flow_regime=0, seriesLst=self.input_x, seriesdata=self.x_data, 
                                      attrLst=self.input_attrs, attrdata=self.attr_tensor)
        attr_norm = process.transNormbyDic(self.attr_tensor, self.input_attrs, statDict, toNorm=True, flow_regime=0)
        attr_norm[torch.isnan(attr_norm)] = 0.0
        series_norm = process.transNormbyDic(self.x_data, self.input_x, statDict, toNorm=True, flow_regime=0)
        series_norm[torch.isnan(series_norm)] = 0.0

        attr_norm_tensor = attr_norm.unsqueeze(0).expand(series_norm.shape[0], -1, -1)
        return torch.cat((series_norm, attr_norm_tensor), dim=2).permute(1, 0, 2)

    def load_future_data(self):
        """Loads and normalizes future projections for trend analysis."""
        print("Processing future projections...")
        x_future_data = []
        for var in self.input_x:
            ds_future = xr.open_dataset(f"{self.cmip6_dir}/{self.clim}/ssp5_8_5/{var}/clipped_US.nc")
            x_future_var = ds_future[var].sel(
                lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                method='nearest'
            ).sel(time=slice(f"{self.trend_future_period[0]}", f"{self.trend_future_period[1]}")).values * 86400
            x_future_data.append(torch.tensor(x_future_var).to(self.device).unsqueeze(-1))

        x_future = torch.cat(x_future_data, dim=-1)
        x_in_future = x_future.unsqueeze(-1)

        # Normalize future projections using training statistics
        series_norm_future = process.transNormbyDic(x_in_future, self.input_x, 
                                                    process.getStatDic(flow_regime=0, seriesLst=self.input_x, 
                                                                       seriesdata=x_in_future, 
                                                                       attrLst=self.input_attrs, 
                                                                       attrdata=self.attr_tensor), 
                                                    toNorm=True, flow_regime=0)

        attr_norm_tensor_future = self.attr_tensor.unsqueeze(0).expand(series_norm_future.shape[0], -1, -1)
        return torch.cat((series_norm_future, attr_norm_tensor_future), dim=2).permute(1, 0, 2), x_future.T

    def get_dataloader(self):
        """Returns a PyTorch DataLoader."""
        dataset = TensorDataset(self.input_norm_tensor, self.x_data.T, self.y_data.T)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class DataLoaderWrapper:
    def __init__(self, clim, ref, period, ref_path, cmip6_dir, 
                 input_x, input_attrs, target_y, save_path, crd='all', batch_size=100, device=0):
        """
        Customizable climate data loader with future projection support.
        """
        self.clim = clim
        self.ref = ref
        self.period = period
        self.ref_path = ref_path
        self.cmip6_dir = cmip6_dir
        self.input_x = input_x
        self.target_y = target_y
        self.input_attrs = input_attrs
        self.crd = crd
        self.batch_size = batch_size

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        # Define periods
        # self.period = self.train_period if self.train else self.test_period

        # Set save paths
        # self.save_path = self.get_save_path()
        self.save_path = save_path


        # Load data
        self.valid_coords = self.get_valid_coords()
        self.attrs_data = self.load_attrs()
        self.x_data, self.time_x = self.load_dynamic_inputs()
        self.y_data = self.load_y_data()
        self.attr_tensor = self.get_attr_tensor()
        self.input_norm_tensor = self.normalize_data()

        # Future trend data
        # if self.trend_analysis:
        #     self.input_norm_tensor_future, self.x_future = self.load_future_data()

    # def get_save_path(self):
    #     """Generates the save path based on training/testing configuration."""
    #     base_path = f'jobs/{self.clim}-{self.ref}/QM_ANN_{self.num}/{self.train_period[0]}_{self.train_period[1]}/'
    #     if not self.train:
    #         base_path += f'{self.test_period[0]}_{self.test_period[1]}/'
    #         os.makedirs(base_path + f'ep40', exist_ok=True)
    #     os.makedirs(base_path, exist_ok=True)
    #     return base_path

    def get_valid_coords(self):
        """Extracts valid latitude-longitude pairs."""
        ds_sample = xr.open_dataset(f"{self.ref_path}*1980*.nc")
        return valid_crd.valid_lat_lon(ds_sample)

    def load_attrs(self):
        """Loads static attributes (elevation, land type, etc.)."""
        attrs_data = {}
        for var, possible_vars in self.input_attrs.items():
            print(f"Processing Attribute {var}...")
            if var == "elevation":
                path = self.cmip6_dir + f'{self.clim}/elev.nc'
                elev = xr.open_dataset(path)
                attrs_data["elevation"] = elev["elevation"].sel(
                    lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                    lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                    method='nearest'
                ).values
        return attrs_data

    def load_dynamic_inputs(self):
        """Loads dynamic inputs (precipitation, wind, etc.)."""
        x_data = []
        time_x = None

        for var, possible_vars in self.input_x.items():
            print(f"Processing Climate {var}...")

            # Open the NetCDF file (assuming a generic variable name for directory structure)
            ds = xr.open_dataset(f"{self.cmip6_dir}/{self.clim}/historical/{var}/clipped_US.nc")

            # Find the first available variable from the list
            matched_var = next((v for v in possible_vars if v in ds.variables), None)

            if matched_var:
                print(f"Using '{matched_var}' for '{var}'")
                x_var = ds[matched_var].sel(
                    lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                    lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                    method='nearest'
                ).sel(time=slice(f"{self.period[0]}", f"{self.period[1]}")).values * 86400
                x_var = torch.tensor(x_var).to(self.device)
                x_data.append(x_var.unsqueeze(-1))
                if time_x is None:
                    time_x = ds.time.values
            else:
                print(f"Variable '{var}' not found in the Climate NetCDF file. Available variables: {list(ds.variables.keys())}")

       
        return torch.cat(x_data, dim=-1), time_x

    def load_y_data(self):
        """Loads reference precipitation data (Livneh)."""
        print("Processing y data...")
        y_data = []
        for var, possible_vars in self.input_y.items():
            selected_var = None
            path = self.ref_path + f'{var}/{self.clim}'

            # Try each possible variable name in the NetCDF file
            for candidate in possible_vars:
                try:
                    y = process.process_data(path, self.period, self.valid_coords, 
                                            self.num, self.device, var=candidate)
                    selected_var = candidate  # Store the successfully processed variable
                    break  # Stop checking once a valid variable is found
                except FileNotFoundError:
                    continue  # If the file isn't found, try the next candidate
                except Exception as e:
                    print(f"Error processing '{candidate}': {e}")
                    continue

            if selected_var:
                print(f"For Reference (Y) data, using '{selected_var}' for '{var}'")
            else:
                raise ValueError(f"None of the variables {possible_vars} were found in the NetCDF file.")
            
            y_data.append(y.unsqueeze(-1))
            
        y_data = torch.cat(y_data, dim=-1)

        return y_data.to(self.x_data.dtype)

    def get_attr_tensor(self):
        """Creates a tensor for static attributes."""
        attr_tensors = []
        for var in self.input_attrs:
            if var in self.attrs_data:
                attr_tensors.append(torch.tensor(self.attrs_data[var]).to(self.x_data.dtype).to(self.device).unsqueeze(-1))
        return torch.cat(attr_tensors, dim=-1)

    def normalize_data(self):
        """Normalizes input data."""
        print("Normalizing data...")
        statDict = process.getStatDic(flow_regime=0, seriesLst=self.input_x.keys(), seriesdata=self.x_data, 
                                      attrLst=self.input_attrs.keys(), attrdata=self.attr_tensor)
        attr_norm = process.transNormbyDic(self.attr_tensor, self.input_attrs, statDict, toNorm=True, flow_regime=0)
        attr_norm[torch.isnan(attr_norm)] = 0.0
        series_norm = process.transNormbyDic(self.x_data, self.input_x, statDict, toNorm=True, flow_regime=0)
        series_norm[torch.isnan(series_norm)] = 0.0

        attr_norm_tensor = attr_norm.unsqueeze(0).expand(series_norm.shape[0], -1, -1)
        return torch.cat((series_norm, attr_norm_tensor), dim=2).permute(1, 0, 2)

    # def load_future_data(self):
    #     """Loads and normalizes future projections for trend analysis."""
    #     print("Processing future projections...")
    #     x_future_data = []
    #     for var in self.input_x:
    #         ds_future = xr.open_dataset(f"{self.cmip6_dir}/{self.clim}/ssp5_8_5/{var}/clipped_US.nc")
    #         x_future_var = ds_future[var].sel(
    #             lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
    #             lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
    #             method='nearest'
    #         ).sel(time=slice(f"{self.trend_future_period[0]}", f"{self.trend_future_period[1]}")).values * 86400
    #         x_future_data.append(torch.tensor(x_future_var).to(self.device).unsqueeze(-1))

    #     x_future = torch.cat(x_future_data, dim=-1)
    #     x_in_future = x_future.unsqueeze(-1)

    #     # Normalize future projections using training statistics
    #     series_norm_future = process.transNormbyDic(x_in_future, self.input_x, 
    #                                                 process.getStatDic(flow_regime=0, seriesLst=self.input_x, 
    #                                                                    seriesdata=x_in_future, 
    #                                                                    attrLst=self.input_attrs, 
    #                                                                    attrdata=self.attr_tensor), 
    #                                                 toNorm=True, flow_regime=0)

    #     attr_norm_tensor_future = self.attr_tensor.unsqueeze(0).expand(series_norm_future.shape[0], -1, -1)
    #     return torch.cat((series_norm_future, attr_norm_tensor_future), dim=2).permute(1, 0, 2), x_future.T

    def get_dataloader(self):
        """Returns a PyTorch DataLoader."""
        dataset = TensorDataset(self.input_norm_tensor, self.x_data.T, self.y_data.T)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
