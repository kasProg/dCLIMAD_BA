import os
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import data.valid_crd as valid_crd
import data.process as process  # Corrected import
from data.helper import UnitManager
import numpy as np

### Some limitations: 
# Loyal to CONUS region, eg files named clipped_US

class DataLoaderWrapper:
    def __init__(self, clim, scenario, ref, period, ref_path, cmip6_dir, shapefile_filter_path,
                 input_x, input_attrs, ref_var, save_path, stat_save_path, crd='all', batch_size=100, train=True, autoregression = False, lag=3, 
                 chunk=False, chunk_size = 365, stride = 90, device=0):
        """
        Customizable climate data loader with future projection support.
        """
        self.clim = clim
        self.scenario = scenario
        self.ref = ref
        self.period = period
        self.ref_path = ref_path
        self.cmip6_dir = cmip6_dir
        self.input_x = input_x
        self.ref_var = ref_var
        self.input_attrs = input_attrs
        self.crd = crd
        self.batch_size = batch_size
        self.train=train
        self.shapefile_filter_path = shapefile_filter_path
        self.device = device

        self.chunk = chunk
        self.chunk_size = chunk_size
        self.stride = stride
        
        self.save_path = save_path
        self.stat_save_path = stat_save_path

        self.autoregression = autoregression
        self.lag = lag

        self.valid_coords = self.get_valid_coords()

        self.attrs_data = self.load_attrs()
        self.x_data, self.time_x = self.load_dynamic_inputs()
        if self.scenario=='historical' or self.ref not in ['livneh', 'gridmet']:
            self.y_data, time_y = self.load_y_data()

        self.attr_tensor = self.get_attr_tensor()
        self.input_norm_tensor = self.normalize_data()

    def get_valid_coords(self):
        """Extracts valid latitude-longitude pairs."""
        if self.clim ==  'ensemble':
            y_clim = 'access_cm2'
        else:
            y_clim = self.clim
        
        
        ds_sample = xr.open_dataset(f"{self.cmip6_dir}/{self.clim}/historical/precipitation/clipped_US.nc")
        
        return valid_crd.valid_lat_lon(ds = ds_sample, var_name = 'pr', 
                                       shapefile_path = self.shapefile_filter_path, attr='OBJECTID', attrList = self.crd)

    
    def load_attrs(self):
        """Loads static attributes (elevation, land type, etc.)."""
        attrs_data = {}
        for var in self.input_attrs:
            print(f"Processing Attribute {var}...")
            # if var == "elevation":
            path = os.path.join(self.cmip6_dir, f'{self.clim}/{var}.nc')
            attr = xr.open_dataset(path)
            attrs_data[var] = attr[var].sel(
                lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                method='nearest'
            ).values
        return attrs_data

    def load_dynamic_inputs(self):
        """Loads dynamic inputs (precipitation, wind, etc.)."""
        x_data = []
        time_x = None
        print('Processing x data...')
        for var, possible_vars in self.input_x.items():
            print(f"Processing x: Climate {var}...")

            # Open the NetCDF file (assuming a generic variable name for directory structure)
            ds = xr.open_dataset(f"{self.cmip6_dir}/{self.clim}/{self.scenario}/{var}/clipped_US.nc")
             # Find the first available variable from the list
            matched_var = next((v for v in possible_vars if v in ds.variables), None)

            unit_identifier = UnitManager(ds)
            units = unit_identifier.get_units()            
           
            if matched_var:
                print(f"Using '{matched_var}' for '{var}'")
                x_var = ds[matched_var].sel(
                    lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                    lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                    method='nearest'
                ).sel(time=slice(f"{self.period[0]}", f"{self.period[1]}"))
                time_x = x_var.time.values
                x_var = x_var.values

                #managing units
                x_var = unit_identifier.convert(x_var, var, units[matched_var]) 

                x_var = torch.tensor(x_var).to(self.device)
                x_data.append(x_var.unsqueeze(-1))
                    
            else:
                print(f"Variable '{var}' not found in the Climate NetCDF file. Available variables: {list(ds.variables.keys())}")

        x_data = torch.cat(x_data, dim=-1)
        torch.save(time_x, f'{self.save_path}/time.pt')
        torch.save(x_data, f'{self.save_path}/x.pt')

        return x_data.to(torch.float32), time_x

    def load_y_data(self):
        y_data = []
        print("Processing y data...")
        if self.ref in ['livneh', 'gridmet']:
            """Loads reference data"""
            # for var, possible_vars in self.ref_var.items():
            print(f'Processing y: Reference {self.ref_var}...')
            if self.clim ==  'ensemble':
                y_clim = 'access_cm2'
            else:
                y_clim = self.clim
            
            path = os.path.join(self.ref_path, f'{self.ref_var}/{y_clim}')
            y = process.process_multi_year_data(path, self.period, self.valid_coords, 
                                                self.device, var=self.ref_var)           
            y_data.append(y.unsqueeze(-1))
                
            y_data = torch.cat(y_data, dim=-1)

            torch.save(y_data, f'{self.save_path}/y.pt')

            return y_data.to(self.x_data.dtype), 'nil'
            
        else:
            ### Used for Perfect Model Framework
            time_y = None
            for var, possible_vars in self.ref_var.items():
                print(f"Processing y: Climate {var}...")

                # Open the NetCDF file (assuming a generic variable name for directory structure)
                ds = xr.open_dataset(f"{self.ref_path}/{self.scenario}/{var}/{self.clim}/clipped_US.nc")
                # Find the first available variable from the list
                matched_var = next((v for v in possible_vars if v in ds.variables), None)

                unit_identifier = UnitManager(ds)
                units = unit_identifier.get_units()            
            
                if matched_var:
                    print(f"Using '{matched_var}' for '{var}'")
                    y_var = ds[matched_var].sel(
                        lat=xr.DataArray(self.valid_coords[:, 0], dims='points'),
                        lon=xr.DataArray(self.valid_coords[:, 1], dims='points'),
                        method='nearest'
                    ).sel(time=slice(f"{self.period[0]}", f"{self.period[1]}"))
                    time_y = y_var.time.values
                    y_var = y_var.values

                    #managing units
                    y_var = unit_identifier.convert(y_var, var, units[matched_var]) 

                    y_var = torch.tensor(y_var).to(self.device)
                    y_data.append(y_var.unsqueeze(-1))
                        
                else:
                    print(f"Variable '{var}' not found in the Climate NetCDF file. Available variables: {list(ds.variables.keys())}")

            y_data = torch.cat(y_data, dim=-1)
            torch.save(time_y, f'{self.save_path}/time_y.pt')
            torch.save(y_data, f'{self.save_path}/y.pt')

            return y_data.to(torch.float32), time_y

    def get_attr_tensor(self):
        """Creates a tensor for static attributes."""
        attr_tensors = []
        for var in self.input_attrs:
            if var in self.attrs_data:
                attr_tensors.append(torch.tensor(self.attrs_data[var]).to(self.x_data.dtype).to(self.device).unsqueeze(-1))
        
        if attr_tensors:
            attr_tensor = torch.cat(attr_tensors, dim=-1).to(torch.float32)
        else:
             # Create an empty tensor with the expected number of dimensions
            attr_tensor = torch.empty(0, dtype=torch.float32).to(self.device)
        return attr_tensor

    def normalize_data(self):
        """Normalizes input data."""
        print("Normalizing data...")
        if self.train:
            statDict = process.getStatDic(flow_regime=0, seriesLst=list(self.input_x.keys()), seriesdata=self.x_data, 
                                      attrLst=list(self.input_attrs), attrdata=self.attr_tensor)
            process.save_dict(statDict, f'{self.stat_save_path}/statDict.json')
        else:
            statDict = process.load_dict(f'{self.stat_save_path}/statDict.json')
        series_norm = process.transNormbyDic(self.x_data, list(self.input_x.keys()), statDict, toNorm=True, flow_regime=0)
        series_norm[torch.isnan(series_norm)] = 0.0

        if self.attr_tensor.numel()!=0:
            attr_norm = process.transNormbyDic(self.attr_tensor, self.input_attrs, statDict, toNorm=True, flow_regime=0)
            attr_norm[torch.isnan(attr_norm)] = 0.0  
            attr_norm_tensor = attr_norm.unsqueeze(0).expand(series_norm.shape[0], -1, -1)

            return torch.cat((series_norm, attr_norm_tensor), dim=2).permute(1, 0, 2)
        
        else:
            return series_norm.permute(1, 0, 2)

    def get_dataloader(self):
        """Returns a PyTorch DataLoader."""
        x = self.x_data.squeeze().T
        y = self.y_data.squeeze().T

        norm_input = self.input_norm_tensor

        if self.autoregression:
            self.input_norm_tensor = self.build_autoregressive_dataset(norm_input=norm_input, k=self.lag)
            # x = torch.tensor(x).to(self.device)
            # y = torch.tensor(y).to(self.device)
        if self.chunk:
            norm_input, y, x = self.chunk_sequence(norm_input, x, y, chunk_size=self.chunk_size, stride=self.stride)

        dataset = TensorDataset(norm_input, x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def get_dataloader_future(self):
        """Returns a PyTorch DataLoader."""
        x = self.x_data.squeeze().T
        norm_input = self.input_norm_tensor

        if self.autoregression:
            self.input_norm_tensor = self.build_autoregressive_dataset(norm_input=norm_input, k=self.lag)

        if self.chunk:
            norm_input, y, x = self.chunk_sequence(norm_input, x, None, chunk_size=self.chunk_size, stride=self.stride)

        dataset = TensorDataset(norm_input, x)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def build_autoregressive_dataset(self, norm_input, k, tv_idx=0, static_idxs=[1, 2, 3, 4]):
        """
        norm_input: (coords, time, features)
        k: number of lags (uses past k+1 values)
        tv_idx: index of time-varying feature
        static_idxs: indices of time-invariant features
        
        Returns:
            X: (coords, time, input_dim) -- with padded lags
        """
        coords, time, features = norm_input.shape
        device = norm_input.device

        # Time-varying feature
        prcp = norm_input[:, :, tv_idx]  # (coords, time)

        # Pad on the left (k zeros) for lagging
        prcp_padded = F.pad(prcp, pad=(k, 0), mode='constant', value=0)  # (coords, time + k)

        # Build lag window
        lagged = []
        for j in reversed(range(k + 1)):
            lagged.append(prcp_padded[:, j:j + time])  # (coords, time)
        
        prcp_stack = torch.stack(lagged, dim=-1)  # (coords, time, k+1)

        # Static features
        static_feat = norm_input[:, 0, static_idxs]  # (coords, static_dim)
        static_feat = static_feat.unsqueeze(1).expand(-1, time, -1)  # (coords, time, static_dim)

        # Combine
        X = torch.cat([prcp_stack, static_feat], dim=-1)  # (coords, time, input_dim)

        return X
    
    def chunk_sequence(self, data, x=None, y=None, chunk_size=365, stride=90):
        coords, time, features = data.shape
        chunks, x_chunks, target_chunks = [], [], []

        for start in range(0, time - chunk_size + 1, stride):
            end = start + chunk_size
            chunks.append(data[:, start:end, :])  # (coords, chunk_size, features)
            if y is not None:
                target_chunks.append(y[:, start:end])  # (coords, chunk_size)
            if x is not None:
                x_chunks.append(x[:, start:end])  # (coords, chunk_size)

        data_chunks = torch.cat(chunks, dim=0)  # (coords * n_chunks, chunk_len, features)
        target_chunks = torch.cat(target_chunks, dim=0) if y is not None else None
        x_chunks = torch.cat(x_chunks, dim=0) if x is not None else None

        return data_chunks, target_chunks, x_chunks
