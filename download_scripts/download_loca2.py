import os
import requests
from bs4 import BeautifulSoup

# ---------------------- USER CONFIGURABLE PARAMETERS ----------------------
BASE_URL = "https://cirrus.ucsd.edu/~pierce/LOCA2/NAmer/"


models= { 
         'mpi_esm1_2_lr':'MPI-ESM1-2-LR', 
         'mri_esm2_0':'MRI-ESM2-0', 
         'access_cm2':'ACCESS-CM2', 
         'miroc6':'MIROC6',
         'gfdl_esm4':'GFDL-ESM4'}
variables = {'precipitation':'pr'}
exps = {'ssp2_4_5': 'ssp245'}

for model in models:
    for var in variables:
      for exp in exps:

        SAVE_DIR = f"/data/kas7897/diffDownscale/cmip6/{model}/{exp}/{var}/loca"  # Directory to save NetCDF files

        # Define dataset parameters
        clim_model = models[model]      # Example: IPSL-CM6A-LR
        resolution = "0p0625deg"      # Grid resolution
        member = "r1i1p1f1"          # clim_model member
        experiment = exps[exp]     # Example: historical, ssp245, ssp585
        variable = variables[var]              # Example: pr (precipitation), tasmax, tasmin, etc.

        # Construct the URL for the dataset's directory
        URL = f"{BASE_URL}{clim_model}/{resolution}/{member}/{experiment}/{variable}/"

        # Ignore seasonal/monthly/yearly variations to download only daily
        EXCLUDED_PATTERNS = ["DJF", "JJA", "MAM", "SON", "monthly", "yearly"]

        print(f"📡 Searching for the correct file in: {URL}")

        # ---------------------- DOWNLOAD FUNCTION ----------------------
        def download_best_match(url, save_path):
            """Find and download the most relevant NetCDF file dynamically."""
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"❌ Failed to access {url} (Status {response.status_code})")
                return

            # Parse the HTML to extract NetCDF file links
            soup = BeautifulSoup(response.text, "html.parser")
            all_files = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".nc")]

            # Filter out unwanted seasonal, monthly, or yearly files
            valid_files = [
                file for file in all_files
                if clim_model in file and experiment in file and not any(pattern in file for pattern in EXCLUDED_PATTERNS)
            ]

            if not valid_files:
                print(f"⚠️ No valid NetCDF file found for clim_model: {clim_model}, experiment: {experiment}.")
                return
            
            # Download each selected file
            for file in valid_files:
                file_url = url + file
                file_path = os.path.join(save_path, file)
                os.makedirs(save_path, exist_ok=True)
                if os.path.exists(file_path):
                    print(f"✅ {file} already exists, skipping...")
                    continue  # Skip already downloaded files

                print(f"⬇️ Downloading: {file}")
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            # # Use the first valid match (assuming the main dataset is listed first)
            # selected_file = valid_files[0]
            # file_url = url + selected_file
            # file_path = os.path.join(save_path, selected_file)

            # Ensure the directory exists
            # os.makedirs(save_path, exist_ok=True)

            # if os.path.exists(file_path):
            #     print(f"✅ {selected_file} already exists, skipping download...")
            #     return

            # # Download the selected file
            # print(f"⬇️ Downloading: {selected_file}")
            # with requests.get(file_url, stream=True) as r:
            #     r.raise_for_status()
            #     with open(file_path, "wb") as f:
            #         for chunk in r.iter_content(chunk_size=8192):
            #             f.write(chunk)

                print(f"✅ Saved: {file_path}")

        # ---------------------- EXECUTE DOWNLOAD ----------------------
        save_path = os.path.join(SAVE_DIR, clim_model, resolution, member, experiment, variable)
        download_best_match(URL, save_path)

        print("🎉 Download complete!")
