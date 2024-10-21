import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def valid_lat_lon(ds):
    lat = ds.lat.values
    lon = ds.lon.values

    # Create meshgrid for target coordinates
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    # Create a mask for valid (non-NaN) values
    valid_mask = ds['prec'].values[0] == ds['prec'].values[0]

    # Extract valid latitude and longitude pairs
    valid_lats = lat_mesh[valid_mask]
    valid_lons = lon_mesh[valid_mask]
    valid_coords = np.column_stack((valid_lats, valid_lons))

    return valid_coords


## To test files
#
# ds_B = xr.open_dataset(f"/data/kas7897/Livneh/prec.1980.nc")
# ds_A = xr.open_dataset(f"/data/kas7897/Livneh/upscale_1by4_LOGnoisy001d_bci/prec_1980.nc")
# # ds_A = xr.open_dataset("/data/kas7897/GFDL-ESM4/livneh_bci/prec.1980.nc")
#
# valid_coords = valid_lat_lon(ds_B)
# #
# # ## calling precpitation values
# #
# prec_B = ds_B['prec'].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'), lon=xr.DataArray(valid_coords[:, 1], dims='points'), method='nearest').values
# prec_A = ds_A['prec'].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'), lon=xr.DataArray(valid_coords[:, 1], dims='points'), method='nearest').values
#
# # Finding num of crd with all zeros
# print(np.sum(np.all(prec_B == 0, axis=0)))
# print(np.sum(np.all(prec_A == 0, axis=0)))
#
#
#
# plt.figure(figsize=(12, 6))
# plt.hist(prec_A[:,400], bins=30, alpha=0.6, label="Noisy")
# plt.hist(prec_B[:,400], bins=30, alpha=0.6, label="Original", color='orange')
# plt.title("Original Gaussian and Target Skewed Distributions")
# plt.legend()
plt.show()