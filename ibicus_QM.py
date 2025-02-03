import torch
import os
from loss import compare_distributions, rmse, distributional_loss_interpolated
import matplotlib.pyplot as plt
import valid_crd
import xarray as xr
from scipy.stats import norm, laplace, gamma, beta

from ibicus.variables import *
from ibicus.utils import gen_PrecipitationHurdleModel, gen_PrecipitationGammaLeftCensoredModel

from ibicus.debias import ECDFM, ISIMIP, QuantileMapping, DeltaChange, CDFt
from ibicus.debias import QuantileDeltaMapping, ScaledDistributionMapping, LinearScaling

from ibicus.evaluate import assumptions, correlation, marginal, multivariate, trend
from ibicus.evaluate.metrics import *

# debiaser = QuantileMapping.from_variable("pr")
debiaser = QuantileMapping.for_precipitation(mapping_type="parametric")
clim = 'miroc6'
ref = 'livneh'
model_type = 'ANN'
layers = 4
degree = 1
num = 'all'
hist_period = [1980, 1990]
future_period = [1991, 2000]
testepoch = 40
quantile = 0.5
clim_var = 'pr'
exp_path = f'models/{clim}-{ref}/QM_{model_type}_layers{layers}_degree{degree}_quantile{quantile}/{num}/{hist_period[0]}_{hist_period[1]}'

### FOR TREND ANALYSIS
scenario = 'ssp2_4_5'
trend_future_period = [2015, 2044]
future_path = f'/data/kas7897/diffDownscale/cmip6/{clim}/{scenario}/precipitation/clipped_US.nc'
trend_analysis = True

dataset = f'/data/kas7897/Livneh/unsplit/{clim}/'


hist_time = pd.date_range(start=f"{hist_period[0]}-01-01", end=f"{hist_period[1]}-12-31", freq="D").to_numpy()
future_time = pd.date_range(start=f"{future_period[0]}-01-01", end=f"{future_period[1]}-12-31", freq="D").to_numpy()
trend_future_time = pd.date_range(start=f"{trend_future_period[0]}-01-01", end=f"{trend_future_period[1]}-12-31", freq="D").to_numpy()


save_path = f'benchmark/QM_parameteric_ibicus/conus/{clim}-{ref}'
os.makedirs(save_path, exist_ok=True)

# Load the .pt files
reference = torch.load(f'{exp_path}/y.pt', weights_only=False)
target_reference = torch.load(f'{exp_path}/{future_period[0]}_{future_period[1]}/y.pt', weights_only=False)

hist_model = torch.load(f'{exp_path}/x.pt', weights_only=False)
future_model = torch.load(f'{exp_path}/{future_period[0]}_{future_period[1]}/x.pt', weights_only=False)
# model = torch.load(f'{dataset}/QM_input/m{period}{num}_{noise_type}.pt', weights_only=False)

# Ensure the tensors are on the CPU and convert to numpy arrays
ref_prcp = reference.unsqueeze(-1).cpu().numpy()/86400
future_ref = target_reference.unsqueeze(-1).cpu().numpy()/86400
hist_prcp = hist_model.unsqueeze(-1).cpu().numpy()/86400
future_prcp = future_model.unsqueeze(-1).cpu().numpy()/86400

# debiased_future = debiaser.apply(ref_prcp, hist_prcp, future_prcp, time_obs = hist_time, time_cm_hist = hist_time, time_cm_future = future_time)
# torch.save(debiased_future*86400, f'{save_path}/{hist_period}_{future_period}{num}.pt')

if trend_analysis:
    ds_sample = xr.open_dataset(f"{dataset}prec.1980.nc")
    valid_coords = valid_crd.valid_lat_lon(ds_sample)
    x_future = xr.open_dataset(future_path)
    future_time = x_future.time.values
    x_future = x_future[clim_var].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                                        lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                                        method='nearest')
    x_future = x_future.sel(time =slice(f'{trend_future_period[0]}', f'{trend_future_period[1]}'))

    x_future = x_future.values
    x_future = np.expand_dims(x_future, axis=-1)

    debiased_future = debiaser.apply(ref_prcp, hist_prcp, x_future, time_obs = hist_time, time_cm_hist = hist_time, time_cm_future = future_time)
    torch.save(debiased_future*86400, f'{save_path}/{scenario}_{trend_future_period}_{num}.pt')
