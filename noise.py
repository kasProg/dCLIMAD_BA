import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

### This code adds noise to the coarsened data

noise_factor = 0.01
noise_type = 'red4'

def generate_elevation_based_noise(elevation, noise_type='red', noise_factor=0.1):
    if noise_type not in ['red', 'white', 'mixed', 'log', 'red2', 'red3', 'red4']:
        raise ValueError("Invalid noise_type. Choose from 'red', 'white', 'mixed', 'log', or 'red2'.")

    # Generate random normal noise
    noise_multiplicative = np.random.normal(0, 1, elevation.shape)
    noise_additive = np.random.normal(0, 1, elevation.shape)

    # Apply different noise types
    if noise_type == 'red':
        # Elevation-based multiplicative noise
        scaled_noise = noise_multiplicative * elevation * noise_factor
    elif noise_type == 'red2':
        # Elevation-based multiplicative noise to constrain -0.2 to 0.2
        new_max = 0.2
        new_min = -0.2
        scaled_noise = (noise_multiplicative * elevation)
        scaled_noise = (scaled_noise - scaled_noise.min()) / \
                       (scaled_noise.max() - scaled_noise.min()) * (new_max - new_min) + new_min
    elif noise_type == 'red3':
        scaled_noise = noise_factor * (elevation - elevation.min()) * noise_multiplicative
    elif noise_type == 'red4':
        new_max = 0.2
        new_min = -0.2
        scaled_noise = noise_factor * (elevation - elevation.mean()) + noise_multiplicative
        scaled_noise = (scaled_noise - scaled_noise.min()) / \
                       (scaled_noise.max() - scaled_noise.min()) * (new_max - new_min) + new_min
    elif noise_type == 'white':
        # Additive noise independent of elevation
        scaled_noise = noise_additive + (elevation*noise_factor)
    elif noise_type == 'mixed':
        # Combination of independent additive and multiplicative noise
        scaled_noise = (noise_multiplicative * elevation * noise_factor) + noise_additive + (elevation*noise_factor)
    elif noise_type == 'log':
        # base = 2
        scaled_noise = noise_multiplicative * (np.exp(elevation * noise_factor))

    return scaled_noise

def interpolate_time_slice(slice_data, lat_A, lon_A, lat_B, lon_B):
    lon_A_2d, lat_A_2d = np.meshgrid(lon_A, lat_A)
    valid_mask = ~np.isnan(slice_data)
    points = np.column_stack((lat_A_2d[valid_mask], lon_A_2d[valid_mask]))
    values = slice_data[valid_mask]
    lon_B_2d, lat_B_2d = np.meshgrid(lon_B, lat_B)
    return griddata(points, values, (lat_B_2d, lon_B_2d), method='cubic')

# Load the elevation and precipitation datasets
# prec_sample = xr.open_dataset(f'/data/kas7897/Livneh/upscale_1by4/prec_1980.nc')
elevation_ds = xr.open_dataset('/data/kas7897/diffDownscale/elev_Livneh.nc')
# elevation_ds['elevation'] = elevation_ds['elevation'].where(prec_sample.prec == prec_sample.prec, np.nan)
noise = generate_elevation_based_noise(elevation_ds.elevation, noise_type=noise_type, noise_factor=noise_factor)

plt.scatter(elevation_ds.elevation.values, noise.values, c ='green', s =10)
plt.show()

for year in range(1980, 2008):
    # prcp_ds = xr.open_dataset(f'/data/kas7897/Livneh/upscale_1by4/prec_{year}.nc')
    prcp_ds = xr.open_dataset(f'/data/kas7897/Livneh/prec.{year}.nc')
    ds_B = xr.open_dataset(f"/data/kas7897/Livneh/prec.{year}.nc")
    noise = noise.broadcast_like(prcp_ds.prec)
    ## This block adds dynamic Elevation Noise
    # Broadcast elevation to match prcp dimensions
    # elevation_broadcasted = elevation_ds.elevation.broadcast_like(prcp_ds.prec)
    # Generate noise
    # noise = generate_elevation_based_noise(elevation_broadcasted, noise_type=noise_type, noise_factor=noise_factor)
    # Create a mask for rainy days (where precipitation > 0)
    rainy_days_mask = prcp_ds.prec > 0
    # Apply noise only to rainy days
    if noise_type == 'red2' or noise_type == 'red4':
        noisy_prcp = prcp_ds.prec*(1 + noise)
    else:
        noisy_prcp = xr.where(rainy_days_mask,
                              prcp_ds.prec + noise,
                              prcp_ds.prec)

    # Add noise to precipitation data
    # noisy_prcp = prcp_ds.prec + noise


    noisy_prcp = xr.where(noisy_prcp < 0, 0, noisy_prcp)

    # prcp_ds.prec.values[prcp_ds.prec.values == 0] = np.nan
    # noisy_prcp.values[noisy_prcp.values == 0] = np.nan
    # n = noisy_prcp.values / (prcp_ds.prec.values) - 1
    # plt.scatter(elevation_ds.elevation.values, np.nanmean(n, axis=0), c='red', s=10)
    # plt.show()


    # Create a new dataset with noisy precipitation
    noisy_prcp_ds = prcp_ds.copy()
    noisy_prcp_ds['prec'] = noisy_prcp
    #
    # ## Bicubic interpolation
    # lat_A = noisy_prcp_ds.lat.values
    # lon_A = noisy_prcp_ds.lon.values
    time = noisy_prcp_ds.time.values
    #
    prcp_A = noisy_prcp_ds.prec.values
    #
    # # Extract target lat and lon from file B
    lat_B = ds_B.lat.values
    lon_B = ds_B.lon.values
    #
    # # Create meshgrid for target coordinates
    # lon_mesh, lat_mesh = np.meshgrid(lon_B, lat_B)
    # prcp_B = np.zeros((len(time), len(lat_B), len(lon_B)))
    #
    # # Perform interpolation
    # for t in range(len(time)):
    #     prcp_B[t] = interpolate_time_slice(prcp_A[t], lat_A, lon_A, lat_B, lon_B)

    # ds_C = xr.Dataset(
    #     data_vars={
    #         'prec': (['time', 'lat', 'lon'], prcp_B)
    #     },
    #     coords={
    #         'time': time,
    #         'lat': lat_B,
    #         'lon': lon_B
    #     }
    # )

    ds_C = xr.Dataset(
        data_vars={
            'prec': (['time', 'lat', 'lon'], prcp_A)
        },
        coords={
            'time': time,
            'lat': lat_B,
            'lon': lon_B
        }
    )
    # # Add attributes if necessary
    ds_C.prec.attrs = noisy_prcp_ds.prec.attrs
    ds_C['prec'] = ds_C['prec'].where(ds_C['prec'] >= 0, 0)
    ds_C['prec'] = ds_C['prec'].where(ds_B.prec == ds_B.prec, np.nan)

    # ds_B.prec.values[ds_B.prec.values == 0] = np.nan
    # ds_C.prec.values[ds_C.prec.values == 0] = np.nan
    # n = ds_C.prec.values / (ds_B.prec.values) - 1
    # elev = xr.open_dataset('/data/kas7897/diffDownscale/elev_Livneh.nc')
    # plt.scatter(elev.elevation.values, np.nanmean(n, axis=0), c='red', s=10)
    # plt.show()

    # Save the new dataset as a NetCDF file
    ds_C.to_netcdf(f'/data/kas7897/Livneh/R4noisyStatic001d/prec_{year}.nc')
    # noisy_prcp_ds.to_netcdf(f'/data/kas7897/Livneh/noisy_new/prec_{year}.nc')
    print(year)



