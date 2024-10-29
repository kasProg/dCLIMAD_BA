import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import os


def load_and_process_year(path, year, valid_coords, num, var):

    ## this is just to manage some difference in file name
    if not (path.endswith(f'prec_{year}.nc') or path.endswith(f'prec.{year}.nc')):
        # If not, try adding both formats and check if the file exists
        file_prec_underscore = os.path.join(path, f'prec_{year}.nc')
        file_prec_dot = os.path.join(path, f'prec.{year}.nc')
        file_wind = os.path.join(path, f'wind.{year}.nc')
        # Check which file exists, and use the correct one
        if os.path.exists(file_prec_underscore):
            path = file_prec_underscore
        elif os.path.exists(file_prec_dot):
            path = file_prec_dot
        elif os.path.exists(file_wind):
            path = file_wind


    x_year = xr.open_dataset(path)

    if num == 'all':
        lat_coords = valid_coords[:, 0]
        lon_coords = valid_coords[:, 1]
    else:
        lat_coords = valid_coords[:num, 0]
        lon_coords = valid_coords[:num, 1]

        # Use the selected coordinates for both cases
    x_data = x_year[var].sel(lat=xr.DataArray(lat_coords, dims='points'),
                                lon=xr.DataArray(lon_coords, dims='points'),
                                method='nearest').values

    return x_data


def process_data(path, train_period, valid_coords, num, device, var):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_and_process_year,
                                    [path] * (train_period[1] - train_period[0]),  # Add path to argument list
                                    range(train_period[0], train_period[1]),
                                    [valid_coords] * (train_period[1] - train_period[0]),
                                    [num] * (train_period[1] - train_period[0]),
                                    [var] * (train_period[1] - train_period[0])))

    x_list = results

    x = np.concatenate(x_list, axis=0)

    return torch.tensor(x).to(device)

def calStat(x):
    a = x.flatten()
    b = a[~np.isnan(a)]
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def calStatgamma(x):  # for daily streamflow and precipitation
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def getStatDic(flow_regime, attrLst=None, attrdata=None, seriesLst=None, seriesdata=None):
    statDict = dict()
    # series data
    if seriesLst is not None:
        for k in range(len(seriesLst)):
            var = seriesLst[k]
            if flow_regime==0:
                if var in ["prcp", "Precip", "runoff", "Runoff", "Runofferror"]:
                    statDict[var] = calStatgamma(seriesdata[:, :, k])
                else:
                    statDict[var] = calStat(seriesdata[:, :, k])
            elif flow_regime==1:
                statDict[var] = calStat(seriesdata[:, :, k])
    # const attribute
    if attrLst is not None:
        for k in range(len(attrLst)):
            var = attrLst[k]
            statDict[var] = calStat(attrdata[:, k])
    return statDict

