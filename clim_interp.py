import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import cftime
import pandas as pd


def get_calendar_type(time_variable):
    """Retrieve the calendar type from the dataset's time variable"""
    if hasattr(time_variable, 'calendar'):
        return time_variable.calendar
    elif isinstance(time_variable.values[0], cftime.datetime):
        return time_variable.values[0].calendar
    else:
        return 'standard'

def find_missing_dates(time_ds_A, time_ds_B):
    """Find missing dates in prcp_ds (time_ds_A) compared to ds_B (time_ds_B)"""
    missing_dates = np.setdiff1d(time_ds_B, time_ds_A)
    return missing_dates

def get_time_units(time_variable):
    """Retrieve the units from the time variable's attributes"""
    if 'units' in time_variable.attrs:
        return time_variable.attrs['units']
    else:
        # Default to a common unit system if not specified
        return 'days since 1900-01-01'

def interpolate_missing_day(prcp_data, time_A, missing_date):
    """Interpolate missing day between the adjacent dates"""
    # Create cftime objects for the previous and next day
    previous_day = missing_date - np.timedelta64(1, 'D')
    next_day = missing_date + np.timedelta64(1, 'D')

    # Ensure the neighbors are in the available time data
    prev_idx = np.where(time_A == previous_day)[0][0] if previous_day in time_A else None
    next_idx = np.where(time_A == next_day)[0][0] if next_day in time_A else None

    # If both neighbors exist, interpolate
    if prev_idx is not None and next_idx is not None:
        interpolated_day = (prcp_data[prev_idx] + prcp_data[next_idx]) / 2.0
        return interpolated_day
    else:
        raise ValueError(f"Cannot interpolate for {missing_date}, missing neighbors.")


def interpolate_time_slice(slice_data, lat_A, lon_A, lat_B, lon_B):
    lon_A_2d, lat_A_2d = np.meshgrid(lon_A, lat_A)
    valid_mask = ~np.isnan(slice_data)
    points = np.column_stack((lat_A_2d[valid_mask], lon_A_2d[valid_mask]))
    values = slice_data[valid_mask]
    lon_B_2d, lat_B_2d = np.meshgrid(lon_B, lat_B)
    return griddata(points, values, (lat_B_2d, lon_B_2d), method='cubic')

### Interpolating GFDL-ESM4 to Livneh crds

for year in range(1983,1996):
    prcp_ds = xr.open_dataset(f"/data/kas7897/GFDL-ESM4/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_{year}_v1.1.nc")
    ds_B = xr.open_dataset(f"/data/kas7897/Livneh/prec.{year}.nc")

    if isinstance(prcp_ds['time'].values[0], cftime.datetime):
        # Convert cftime to pandas datetime
        new_time = pd.to_datetime([t.strftime('%Y-%m-%d') for t in prcp_ds['time'].values])

        # Replace the time coordinate
        prcp_ds = prcp_ds.assign_coords(time=new_time)
    # # Check if any time coordinate uses cftime
    # uses_cftime = any(isinstance(prcp_ds[var].values[0], cftime.datetime)
    #                   for var in prcp_ds.coords
    #                   if prcp_ds[var].dtype == 'O')
    # if uses_cftime:
    #    prcp_ds = xr.decode_cf(prcp_ds)

    ## Bicubic interpolation
    lat_A = prcp_ds.lat.values
    lon_A = prcp_ds.lon.values
    time_A = prcp_ds.time.values

    prcp_A = prcp_ds.pr.values*86400 ##converting to mm/day

    # Extract target lat and lon from file B
    lat_B = ds_B.lat.values
    lon_B = ds_B.lon.values
    time_B = ds_B.time.values




    # Find missing dates in prcp_ds
    missing_dates = find_missing_dates(time_A, time_B)

    # Handle missing dates by interpolation
    for missing_date in missing_dates:
        interpolated_day = interpolate_missing_day(prcp_A, time_A, missing_date)
        insert_idx = np.searchsorted(time_A, missing_date)  # Find where to insert the missing date
        time_A = np.insert(time_A, insert_idx, missing_date)
        prcp_A = np.insert(prcp_A, insert_idx, interpolated_day, axis=0)

    # Create meshgrid for target coordinates
    lon_mesh, lat_mesh = np.meshgrid(lon_B, lat_B)
    prcp_B = np.zeros((len(time_B), len(lat_B), len(lon_B)))

    # Perform interpolation
    for t in range(len(time_B)):
        prcp_B[t] = interpolate_time_slice(prcp_A[t], lat_A, lon_A, lat_B, lon_B)

    ds_C = xr.Dataset(
        data_vars={
            'prec': (['time', 'lat', 'lon'], prcp_B)
        },
        coords={
            'time': time_B,
            'lat': lat_B,
            'lon': lon_B
        }
    )
    # Add attributes if necessary
    # ds_C.prec.attrs = noisy_prcp_ds.prec.attrs
    ds_C['prec'] = ds_C['prec'].where(ds_C['prec'] >= 0, 0)
    ds_C['prec'] = ds_C['prec'].where(ds_B.prec == ds_B.prec, np.nan)

    # Save the new dataset as a NetCDF file
    ds_C.to_netcdf(f"/data/kas7897/GFDL-ESM4/livneh_bci/prec.{year}.nc")
    # noisy_prcp_ds.to_netcdf(f'/data/kas7897/Livneh/noisy_new/prec_{year}.nc')
    print(year)