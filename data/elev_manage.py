from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import netCDF4 as nc
import rasterio
import numpy as np
from scipy.interpolate import griddata


#######-------THIS CODE INTERPOLATES (LINEARLY) ELEVATION TIFF FILE TO NC FILE PROVIDED--------########

# Function to convert longitudes from 0-360 to -180 to 180
def convert_lon_to_neg180(lons):
    return np.where(lons > 180, lons - 360, lons)

clim = 'mri_esm2_0'
# Step 1: Extract lat/lon from the precipitation NetCDF file and convert longitude
precip_nc_file = f'/data/kas7897/diffDownscale/cmip6/{clim}/historical/precipitation/clipped_US.nc'

#save_path
elevation_nc_file = f'/data/kas7897/diffDownscale/cmip6/{clim}/elev.nc'

with nc.Dataset(precip_nc_file, 'r') as precip_ds:
    lats = precip_ds.variables['lat'][:]
    lons = precip_ds.variables['lon'][:]

    # Convert longitudes from 0-360 to -180 to 180 for interpolation
    lons_converted = convert_lon_to_neg180(lons)

# Step 2: Read the elevation data from the TIFF file
tiff_file = "/data/wxt42/raw_data/Topography/DEM_Attr/mean_elev.tif"
# tiff_file = "/data/kas7897/diffDownscale/elevation_1KMmn_GMTEDmn_with_Antarctica_from_World_e-Atlas.tif"
with rasterio.open(tiff_file) as elevation_ds:
    elevation_data = elevation_ds.read(1)  # Read the first band (assuming it's elevation)
    tiff_transform = elevation_ds.transform
    tiff_bounds = elevation_ds.bounds
    tiff_res = elevation_ds.res

    # Create coordinate arrays for the elevation data
    tiff_lon = np.arange(tiff_bounds.left, tiff_bounds.right, tiff_res[0])
    tiff_lat = np.arange(tiff_bounds.top, tiff_bounds.bottom, -tiff_res[1])
    tiff_lon_grid, tiff_lat_grid = np.meshgrid(tiff_lon, tiff_lat)

# Step 3: Interpolate the elevation at the lat/lon points from the precipitation NetCDF
lon_grid, lat_grid = np.meshgrid(lons_converted, lats)
elevation_interp = griddata((tiff_lon_grid.flatten(), tiff_lat_grid.flatten()),
                            elevation_data.flatten(),
                            (lon_grid, lat_grid),
                            method='linear')

# Step 4: Create a new NetCDF file to store the interpolated elevation data
with nc.Dataset(elevation_nc_file, 'w', format='NETCDF4') as elevation_ds:
    # Create dimensions
    lat_dim = elevation_ds.createDimension('lat', len(lats))
    lon_dim = elevation_ds.createDimension('lon', len(lons))

    # Create coordinate variables
    latitudes = elevation_ds.createVariable('lat', np.float32, ('lat',))
    longitudes = elevation_ds.createVariable('lon', np.float32, ('lon',))

    # Assign values to lat/lon
    latitudes[:] = lats
    longitudes[:] = lons  # Keep the original 0-360 longitudes

    # Create the elevation variable
    elevation = elevation_ds.createVariable('elevation', np.float32, ('lat', 'lon'))
    elevation.units = 'meters'
    elevation.long_name = 'Elevation at the given lat/lon'

    # Assign interpolated elevation values
    elevation[:] = elevation_interp
