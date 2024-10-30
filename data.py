import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import os
import json

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


def calStatgamma(x):
    x = x.cpu().detach().numpy()
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out NaN
    b = np.log10(torch.sqrt(b) + 0.1)  # transformation to change gamma characteristics
    p10 = np.percentile(b, 0.1).float()
    p90 = np.percentile(b, 0.9).float()
    mean = np.mean(b).float()
    std = np.std(b).float()
    if std < 0.001:
        std = np.tensor(1.0)
    return [p10, p90, mean, std]

def calStat(data):
    data = data.cpu().detach().numpy()
    data = data.flatten()
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    pct10 = np.percentile(data, 0.1)
    pct90 = np.percentile(data, 0.9)
    return [pct10, pct90, mean, std]

def getStatDic(flow_regime, attrLst=None, attrdata=None, seriesLst=None, seriesdata=None):
    statDict = {}
    # series data
    if seriesLst is not None:
        for k in range(len(seriesLst)):
            var = seriesLst[k]
            if flow_regime == 0:
                if var in ["prcp", "Precip", "runoff", "Runoff", "Runofferror", "noisy_prcp"]:
                    statDict[var] = calStatgamma(seriesdata[:, :, k])
                else:
                    statDict[var] = calStat(seriesdata[:, :, k])
            elif flow_regime == 1:
                statDict[var] = calStat(seriesdata[:, :, k])
    # const attribute
    if attrLst is not None:
        for k in range(len(attrLst)):
            var = attrLst[k]
            statDict[var] = calStat(attrdata[:, k])
    return statDict


def transNormbyDic(x, varLst, staDic, toNorm, flow_regime):
    if isinstance(varLst, str):
        varLst = [varLst]
    out = torch.zeros_like(x)

    special_vars = [
        "prcp", "usgsFlow", "Precip", "runoff", "Runoff", "Runofferror", "noisy_prcp"
    ]

    for k, var in enumerate(varLst):
        stat = staDic[var]
        if toNorm:
            if x.dim() == 3:
                if flow_regime == 0 and var in special_vars:
                    temp = torch.log10(torch.sqrt(x[:, :, k]) + 0.1)
                    out[:, :, k] = (temp - stat[2]) / stat[3]
                else:
                    out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif x.dim() == 2:
                if flow_regime == 0 and var in special_vars:
                    temp = torch.log10(torch.sqrt(x[:, k]) + 0.1)
                    out[:, k] = (temp - stat[2]) / stat[3]
                else:
                    out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if x.dim() == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if flow_regime == 0 and var in special_vars:
                    temptrans = torch.pow(10, out[:, :, k]) - 0.1
                    temptrans = torch.clamp(temptrans, min=0)  # set negative as zero
                    out[:, :, k] = temptrans ** 2
            elif x.dim() == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if flow_regime == 0 and var in special_vars:
                    temptrans = torch.pow(10, out[:, k]) - 0.1
                    temptrans = torch.clamp(temptrans, min=0)
                    out[:, k] = temptrans ** 2
    return out


# Saving the dictionary
def save_dict(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f)

# Loading the dictionary
def load_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)