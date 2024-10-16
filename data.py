import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import os


def load_and_process_year(path, year, valid_coords, num):

    ## this is just to manage some difference in file name
    if not (path.endswith(f'prec_{year}.nc') or path.endswith(f'prec.{year}.nc')):
        # If not, try adding both formats and check if the file exists
        file_prec_underscore = os.path.join(path, f'prec_{year}.nc')
        file_prec_dot = os.path.join(path, f'prec.{year}.nc')
        # Check which file exists, and use the correct one
        if os.path.exists(file_prec_underscore):
            path = file_prec_underscore
        elif os.path.exists(file_prec_dot):
            path = file_prec_dot


    x_year = xr.open_dataset(path)

    if num == 'all':
        lat_coords = valid_coords[:, 0]
        lon_coords = valid_coords[:, 1]
    else:
        lat_coords = valid_coords[:num, 0]
        lon_coords = valid_coords[:num, 1]

        # Use the selected coordinates for both cases
    x_data = x_year['prec'].sel(lat=xr.DataArray(lat_coords, dims='points'),
                                lon=xr.DataArray(lon_coords, dims='points'),
                                method='nearest').values

    return x_data


def process_data(path, train_period, valid_coords, num, device):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_and_process_year,
                                    [path] * (train_period[1] - train_period[0]),  # Add path to argument list
                                    range(train_period[0], train_period[1]),
                                    [valid_coords] * (train_period[1] - train_period[0]),
                                    [num] * (train_period[1] - train_period[0])))

    x_list = results

    x = np.concatenate(x_list, axis=0)

    return torch.tensor(x).to(device)
