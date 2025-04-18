import xarray as xr
import os
import cftime
import numpy as np
import pandas as pd
import json
from data.helper import find_missing_dates, interpolate_missing_day


#######--------THIS CODE COMPUTES ENSEMBLE OF CLIMATE MODELS (RIGHT NOW MODIFIED FOR diffDOWNSCALE)----------#########

clim_models = {'access_cm2':'ACCESS-CM2',
               'gfdl_esm4': 'GFDL-ESM4',
               'ipsl_cm6a_lr': 'IPSL-CM6A-LR',
               'miroc6': 'MIROC6',
               'mpi_esm1_2_lr': 'MPI-ESM1-2-LR',
               'mri_esm2_0': 'MRI-ESM2_0'}

degree = {'access_cm2':'1',
               'gfdl_esm4': '2',
               'ipsl_cm6a_lr': '1',
               'miroc6': '2',
               'mpi_esm1_2_lr': '2',
               'mri_esm2_0': '1'}
testep = {'access_cm2':'50',
               'gfdl_esm4': '50',
               'ipsl_cm6a_lr': '50',
               'miroc6': '50',
               'mpi_esm1_2_lr': '200',
               'mri_esm2_0': '50'}
emph_quantile = {'access_cm2':'0.9',
               'gfdl_esm4': '0.5',
               'ipsl_cm6a_lr': '0.9',
               'miroc6': '0.9',
               'mpi_esm1_2_lr': '0.9',
               'mri_esm2_0': '0.5'}

variables = {'precipitation':'pr'}
# exps = {'historical': 'historical'}
exps = {'diffDownscale': 'diffDownscale'}
period = [1981, 1995]
ensemble_path = f'/pscratch/sd/k/kas7897/cmip6/ensemble/1950_1980/{period[0]}_{period[1]}/'


def create_noon_date_array(start_date, end_date):
    # Create a date range with dates only
    date_array = np.arange(np.datetime64(start_date), np.datetime64(end_date) + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
    # Add 12 hours (noon) to each date
    noon_array = date_array + np.timedelta64(12, 'h')
    return noon_array



for var in variables:
    for exp in exps:
        clim_array = []
        save_path = f'{ensemble_path}/{exp}/{var}'
        os.makedirs(save_path, exist_ok=True)

        if exp=='historical':
            time_ref = create_noon_date_array("1950-01-01", "2014-12-31")
        elif exp == 'diffDownscale':
            time_ref = create_noon_date_array("1981-01-01", "1995-12-31")

        else:
            time_ref = create_noon_date_array("2015-01-01", "2099-12-31")


        for clim in clim_models:
            # clim_ds = xr.open_dataset(f'/data/kas7897/diffDownscale/cmip6/{clim}/{exp}/{var}/clipped_US.nc')
            # clim_ds = xr.open_dataset(f'/data/kas7897/diffDownscale/cmip6/{clim}/{exp}/{var}/loca/coarse_USclip.nc')
            clim_ds = xr.open_dataset(f'/pscratch/sd/k/kas7897/diffDownscale/jobs/{clim}-livneh/QM_ANN_layers4_degree{degree[clim]}_quantile{emph_quantile[clim]}/all/1950_1980/{period[0]}_{period[1]}/ep{testep[clim]}/xt.nc')

            if isinstance(clim_ds['time'].values[0], cftime.datetime):
                # Convert cftime to pandas datetime
                new_time = pd.to_datetime([t.strftime('%Y-%m-%d 12:00:00') for t in clim_ds['time'].values])

                # Replace the time coordinate
                clim_ds = clim_ds.assign_coords(time=new_time)

            var_clim = clim_ds[variables[var]].values 


            time_clim = clim_ds.time.values

            # Find missing dates in clim_ds
            missing_dates = find_missing_dates(time_clim, time_ref)
            # Handle missing dates by interpolation
            for missing_date in missing_dates:
                interpolated_day = interpolate_missing_day(var_clim, time_clim, missing_date)
                insert_idx = np.searchsorted(time_clim, missing_date)  # Find where to insert the missing date
                time_clim = np.insert(time_clim, insert_idx, missing_date)
                var_clim = np.insert(var_clim, insert_idx, interpolated_day, axis=0)

            lat_ref = clim_ds.lat.values
            lon_ref = clim_ds.lon.values
            
            ds_interp = xr.Dataset(
                        data_vars={
                            variables[var]: (['time', 'lat', 'lon'], var_clim)
                        },
                        coords={
                            'time': time_clim,
                            'lat': lat_ref,
                            'lon': lon_ref
                        }
                    )
            
            clim_array.append(ds_interp[variables[var]])
        

        reference_clim = clim_array[0]  # Choose first dataset as the common grid
        common_lat = reference_clim["lat"]
        common_lon = reference_clim["lon"]

        interpolated_clim_array = [
                da.interp(lat=common_lat, lon=common_lon, method="nearest") for da in clim_array
            ]
        ensemble = xr.concat(interpolated_clim_array, dim="stacked").mean(dim="stacked")

        ensemble.to_netcdf(f'{save_path}/ensemble.nc')
        print(f'Yo Yo Ensemble Done Boy {exp}__{var}')


## logging climate models in ensembles
combined = {
    "clim_models": clim_models,
    "degree": degree,
    "testep": testep,
    "emph_quantile": emph_quantile,
}

with open(f"{ensemble_path}/clim_models.json", "w") as f:
    json.dump(combined, f, indent=4)