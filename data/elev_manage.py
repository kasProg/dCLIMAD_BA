from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import netCDF4 as nc
import rasterio
import numpy as np
from scipy.interpolate import griddata


#######-------THIS CODE INTERPOLATES (LINEARLY) ELEVATION or ANY ATTRI. TIFF FILE TO NC FILE PROVIDED--------########

# Function to convert longitudes from 0-360 to -180 to 180
def convert_lon_to_neg180(lons):
    return np.where(lons > 180, lons - 360, lons)

clim_models = ['access_cm2', 'miroc6', 'ipsl_cm6a_lr', 'mpi_esm1_2_lr', 'mri_esm2_0', 'gfdl_esm4']
cmip6_dir = '/pscratch/sd/k/kas7897/cmip6'
tiff_file = "/pscratch/sd/k/kas7897/slope_0.01.tif"

for clim in clim_models:
    # Step 1: Extract lat/lon from the precipitation NetCDF file and convert longitude
    precip_nc_file = f'{cmip6_dir}/{clim}/historical/precipitation/clipped_US.nc'

    #save_path
    attri_nc_file = f'{cmip6_dir}/{clim}/elev.nc'

    with nc.Dataset(precip_nc_file, 'r') as precip_ds:
        lats = precip_ds.variables['lat'][:]
        lons = precip_ds.variables['lon'][:]

        # Convert longitudes from 0-360 to -180 to 180 for interpolation
        lons_converted = convert_lon_to_neg180(lons)

    # Step 2: Read the attri data from the TIFF file

    # tiff_file = "/data/kas7897/diffDownscale/elevation_1KMmn_GMTEDmn_with_Antarctica_from_World_e-Atlas.tif"
    with rasterio.open(tiff_file) as attri_ds:
        attri_data = attri_ds.read(1)  # Read the first band (assuming it's attri)
        tiff_transform = attri_ds.transform
        tiff_bounds = attri_ds.bounds
        tiff_res = attri_ds.res

        # Create coordinate arrays for the attri data
        tiff_lon = np.arange(tiff_bounds.left, tiff_bounds.right, tiff_res[0])
        tiff_lat = np.arange(tiff_bounds.top, tiff_bounds.bottom, -tiff_res[1])
        tiff_lon_grid, tiff_lat_grid = np.meshgrid(tiff_lon, tiff_lat)

    # Step 3: Interpolate the attri at the lat/lon points from the precipitation NetCDF
    lon_grid, lat_grid = np.meshgrid(lons_converted, lats)
    attri_interp = griddata((tiff_lon_grid.flatten(), tiff_lat_grid.flatten()),
                                attri_data.flatten(),
                                (lon_grid, lat_grid),
                                method='linear')

    # Step 4: Create a new NetCDF file to store the interpolated attri data
    with nc.Dataset(attri_nc_file, 'w', format='NETCDF4') as attri_ds:
        # Create dimensions
        lat_dim = attri_ds.createDimension('lat', len(lats))
        lon_dim = attri_ds.createDimension('lon', len(lons))

        # Create coordinate variables
        latitudes = attri_ds.createVariable('lat', np.float32, ('lat',))
        longitudes = attri_ds.createVariable('lon', np.float32, ('lon',))

        # Assign values to lat/lon
        latitudes[:] = lats
        longitudes[:] = lons  # Keep the original 0-360 longitudes

        # Create the attri variable
        attri = attri_ds.createVariable('slope', np.float32, ('lat', 'lon'))
        attri.units = 'degree'
        attri.long_name = 'Slope at the given lat/lon'

        # Assign interpolated elevation values
        attri[:] = attri_interp
