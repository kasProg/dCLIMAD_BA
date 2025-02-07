
import xarray as xr
import numpy as np
import os
import cftime
import pandas as pd
from helper import interpolate_time_slice


########------THIS CODE REGRIDS UNSPLIT-LIVNEH TO CLIMATE-MODEL RESOLUTION-------##########

cmip6_dir = "/pscratch/sd/k/kas7897/cmip6"
obs_path = '/pscratch/sd/k/kas7897/Livneh/unsplit/precipitation'
obs_year_range = [1950, 2014]

clim_models = ['gfdl_esm4', 'ipsl_cm6a_lr', 'miroc6', 'mpi_esm1_2_lr', 'mri_esm2_0', 'access_cm2']
for clim in clim_models:

    clim_path = f'{cmip6_dir}/{clim}/historical/precipitation/clipped_US.nc'
    
    path = f'{obs_path}/{clim}'
    os.makedirs(path, exist_ok=True)


    for year in range(obs_year_range[0], obs_year_range[1]+1):

        prcp_ds = xr.open_dataset(f'{obs_path}/livneh_unsplit_precip.2021-05-02.{year}.nc')
        ds_og = xr.open_dataset(f"{clim_path}")


        # Create a new dataset with noisy precipitation
        noisy_prcp_ds = prcp_ds.copy()
        time = noisy_prcp_ds.Time.values
        prcp_n = noisy_prcp_ds.PRCP.values

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

        if isinstance(ds_og['time'].values[0], cftime.datetime):
                    # Convert cftime to pandas datetime
                    new_time = pd.to_datetime([t.strftime('%Y-%m-%d') for t in ds_og['time'].values])

                    # Replace the time coordinate
                    ds_og = ds_og.assign_coords(time=new_time)

        ref_ds_aligned = ds_og.reindex(time=ds_C['time'], method='nearest')
        # ds_interp[variables[var]] = ds_interp[variables[var]].where(ref_ds_aligned.prec == ref_ds_aligned.prec, np.nan)
        # ref_ds_aligned = ref_ds.reindex(time=ds_interp['time'], method='nearest')
        # ds_interp[variables[var]] = ds_interp[variables[var]].where(ref_ds_aligned.prec == ref_ds_aligned.prec, np.nan)

        ds_C['prec'] = ds_C['prec'].where(ref_ds_aligned.pr == ref_ds_aligned.pr, np.nan)
    


        # Save the new dataset as a NetCDF file
        ds_C.to_netcdf(f'{path}/prec.{year}.nc')
        print(year)






