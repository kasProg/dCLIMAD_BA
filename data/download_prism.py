import os
import requests
import zipfile
import io
from datetime import datetime, timedelta
from tqdm import tqdm
from glob import glob
import xarray as xr

def download_prism_webservice_daily(
    start_date="1981-01-01",
    end_date="2014-12-31",
    out_dir="PRISM_daily_nc",
    fmt="nc"
):
    base_url = "https://services.nacse.org/prism/data/get"
    region = "us"
    resolution = "800m"
    element = "ppt"

    os.makedirs(out_dir, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.now() if end_date is None else datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    while current <= end:
        ymd = current.strftime("%Y%m%d")
        nc_filename = f"{element}_{ymd}.{fmt}"
        nc_filepath = os.path.join(out_dir, nc_filename)

        if os.path.exists(nc_filepath):
            print(f"[SKIP] {nc_filename} already exists.")
        else:
            url = f"{base_url}/{region}/{resolution}/{element}/{ymd}?format={fmt}"
            print(f"Downloading: {url}")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Extract .nc file from ZIP
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        for member in z.namelist():
                            if member.endswith(".nc"):
                                 z.extractall(out_dir)
                    print(f"[OK] Extracted all files for {ymd}")
                else:
                    print(f"[{response.status_code}] Skipped {ymd}")
            except Exception as e:
                print(f"[ERROR] Failed {ymd}: {e}")

        current += timedelta(days=1)


def merge_prism_daily_to_yearly(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Match PRISM daily NetCDF files
    pattern = os.path.join(input_dir, "prism_ppt_us_30s_*.nc")
    nc_files = sorted(glob(pattern))

    # Organize files by year
    yearly_files = {}
    for file in nc_files:
        basename = os.path.basename(file)
        date_str = basename.replace("prism_ppt_us_30s_", "").replace(".nc", "")
        year = date_str[:4]

        if year not in yearly_files:
            yearly_files[year] = []
        yearly_files[year].append((date_str, file))

    for year, file_list in yearly_files.items():
        file_list.sort()  # Ensure chronological order
        output_path = os.path.join(output_dir, f"prism_ppt_us_30s_{year}.nc")

        if os.path.exists(output_path):
            print(f"[SKIP] {output_path} already exists.")
            continue

        datasets = []
        for date_str, file in file_list:
            try:
                ds = xr.open_dataset(file)
                ds = ds.expand_dims("time")
                ds["time"] = [datetime.strptime(date_str, "%Y%m%d")]
                datasets.append(ds)
            except Exception as e:
                print(f"[ERROR] Skipping {file}: {e}")

        if datasets:
            combined = xr.concat(datasets, dim="time")
            combined.to_netcdf(output_path)
            print(f"[OK] Saved {output_path}")
        else:
            print(f"[WARN] No valid data for {year}")

if __name__ == "__main__":
    # download_prism_webservice_daily(out_dir='/pscratch/sd/k/kas7897/PRISM/ppt')
    input_dir = "/pscratch/sd/k/kas7897/PRISM/ppt"
    output_dir = "/pscratch/sd/k/kas7897/PRISM/ppt/yearly"
    merge_prism_daily_to_yearly(input_dir, output_dir)