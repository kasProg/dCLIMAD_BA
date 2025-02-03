
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import warnings
import numpy as np
import regionmask
import glob
from helper import clip_netcdf, search_path

#######-------THIS CODE CLIPS CLIMATE-MODELS TO CONUS SHAPEFILE-------#######

models= {'gfdl_esm4':'GFDL-ESM4', 'access_cm2':'ACCESS-CM2'}
variables = {'precipitation':'pr', 'near_surface_air_temperature':'tas'}
exps = {'historical': 'historical', 'ssp2_4_5':'ssp245'}

for model in models:
    for var in variables:
      for exp in exps:

        if exp == 'historical':
                start_date, end_date = "19500101", "20141231"
        else:
            start_date, end_date = "20150101", "20991231"

        # Construct the search pattern for files with any grid type (*)
        search_pattern = f"/data/kas7897/diffDownscale/cmip6/{model}/{exp}/{var}/{variables[var]}_day_{models[model]}_{exps[exp]}_r1i1p1f1_*_{start_date}-{end_date}.nc"
        netcdf_file = search_path(search_pattern)
          
        shapefile_path = "/data/kas7897/conus/conus.shp"

        # Step 1: Open the NetCDF file
        clipped_ds = clip_netcdf(nc_file = netcdf_file, shape_file = shapefile_path)

        # Step 5: Save the clipped dataset
        output_path = f"/data/kas7897/diffDownscale/cmip6/{model}/{exp}/{var}/clipped_US.nc"
        clipped_ds[variables[var]].to_netcdf(output_path)

        print(f"Clipped NetCDF saved to {output_path}")




