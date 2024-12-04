
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

### This code regrids data


clim_path = '/data/kas7897/Livneh/upscale_1by4_linear'
obs_path = '/data/kas7897/Livneh/'
path = f'/data/kas7897/Livneh/upscale_1by4'
os.makedirs(path, exist_ok=True)


def interpolate_time_slice(slice_data, lat_A, lon_A, lat_B, lon_B):
    lon_A_2d, lat_A_2d = np.meshgrid(lon_A, lat_A)
    valid_mask = ~np.isnan(slice_data)
    points = np.column_stack((lat_A_2d[valid_mask], lon_A_2d[valid_mask]))
    values = slice_data[valid_mask]
    lon_B_2d, lat_B_2d = np.meshgrid(lon_B, lat_B)
    return griddata(points, values, (lat_B_2d, lon_B_2d), method='nearest')


for year in range(1980, 2008):

    prcp_ds = xr.open_dataset(f'{obs_path}/prec.{year}.nc')
    ds_og = xr.open_dataset(f"{clim_path}/prec_{year}.nc")


    # Create a new dataset with noisy precipitation
    noisy_prcp_ds = prcp_ds.copy()
    time = noisy_prcp_ds.time.values
    prcp_n = noisy_prcp_ds.prec.values

    # ## Bicubic interpolation
    lat_og = ds_og.lat.values
    lon_og = ds_og.lon.values
    lat_A = noisy_prcp_ds.lat.values
    lon_A = noisy_prcp_ds.lon.values
    prcp_final = np.zeros((len(time), len(lat_og), len(lon_og)))

    # Perform interpolation
    for t in range(len(time)):
        prcp_final[t] = interpolate_time_slice(prcp_n[t], lat_A, lon_A, lat_og, lon_og)
    


    ds_C = xr.Dataset(
        data_vars={
            'prec': (['time', 'lat', 'lon'], prcp_final)
        },
        coords={
            'time': time,
            'lat': lat_og,
            'lon': lon_og
        }
    )


    # # Add attributes if necessary
    # ds_C.prec.attrs = noisy_prcp_ds.prec.attrs
    ds_C['prec'] = ds_C['prec'].where(ds_C['prec'] >= 0, 0)

    ds_C['prec'] = ds_C['prec'].where(ds_og.prec == ds_og.prec, np.nan)
 


    # Save the new dataset as a NetCDF file
    ds_C.to_netcdf(f'{path}/prec_{year}.nc')
    print(year)






