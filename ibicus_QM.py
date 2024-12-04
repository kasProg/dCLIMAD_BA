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
# dataset = '/data/kas7897/Livneh/'
# ds_sample = xr.open_dataset(f"{dataset}prec.1980.nc")
# valid_coords = valid_crd.valid_lat_lon(ds_sample)

dataset = '/data/kas7897/Livneh/upscale_1by4/'
clim_model = '/data/kas7897/GFDL-ESM4/'
noise_type = 'livneh025d_interp'
hist_period = [1980, 1990]
future_period = [1991, 1995]

hist_time = pd.date_range(start=f"{hist_period[0]}-01-01", end=f"{hist_period[1]}-12-31", freq="D").to_numpy()
future_time = pd.date_range(start=f"{future_period[0]}-01-01", end=f"{future_period[1]}-12-31", freq="D").to_numpy()


num = 2000
QM = True
save_path = f'benchmark/QM_ibicus/'
os.makedirs(save_path, exist_ok=True)

# Load the .pt files
reference = torch.load(f'{dataset}QM_input/y{hist_period}{num}.pt', weights_only=False)
target_reference = torch.load(f'{dataset}QM_input/y{future_period}{num}.pt', weights_only=False)

hist_model = torch.load(f'{clim_model}QM_input/x{hist_period}{num}_{noise_type}.pt', weights_only=False)
future_model = torch.load(f'{clim_model}QM_input/x{future_period}{num}_{noise_type}.pt', weights_only=False)
# model = torch.load(f'{dataset}/QM_input/m{period}{num}_{noise_type}.pt', weights_only=False)

# Ensure the tensors are on the CPU and convert to numpy arrays
ref_prcp = reference.unsqueeze(-1).cpu().numpy()
future_ref = target_reference.unsqueeze(-1).cpu().numpy()
hist_prcp = hist_model.unsqueeze(-1).cpu().numpy()
future_prcp = future_model.unsqueeze(-1).cpu().numpy()

debiased_future = debiaser.apply(ref_prcp, hist_prcp, future_prcp, time_obs = hist_time, time_cm_hist = hist_time, time_cm_future = future_time)
torch.save(debiased_future, f'{save_path}/{future_period}{num}.pt')

# debiased_future = debiased_future.squeeze()
# future_prcp = future_prcp.squeeze()
# future_ref = future_ref.squeeze()

# quantile_rmse_model = torch.sqrt(distributional_loss_interpolated(torch.tensor(future_prcp), torch.tensor(future_ref), device='cpu', num_quantiles=100))
# quantile_rmse_bs = torch.sqrt(distributional_loss_interpolated(torch.tensor(debiased_future), torch.tensor(future_ref), device='cpu', num_quantiles=100))

# print('Future Metric:\n')
# avg_improvement, individual_improvements = compare_distributions(debiased_future, future_prcp, future_ref)
# print(f"Average distribution improvement: {avg_improvement:.4f}")

# print(f"Quantile RMSE between Model and Target: {quantile_rmse_model}")
# print(f"Quantile RMSE between Corrected and Target: {quantile_rmse_bs}")

# print(f"RMSE Diff: {quantile_rmse_model - quantile_rmse_bs}")


# best_ind, best_improv = max(enumerate(individual_improvements), key=lambda x: x[1])

# # Step 6: Plotting the original and transformed distributions
# plt.figure(figsize=(12, 6))
# plt.suptitle(f'Quantile Mapping (Delta) \n WS Distance Improvement:{best_improv}')
# plt.subplot(1, 2, 1)
# plt.hist(future_prcp[:, best_ind], bins=30, alpha=0.6, label="Model x")
# plt.hist(future_ref[:, best_ind], bins=30, alpha=0.6, label="Target Y", color='orange')
# plt.title("Modeled x and Target Distributions")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.hist(debiased_future[:, best_ind], bins=30, alpha=0.6, label="Transformed x", color='green')
# plt.hist(future_ref[:, best_ind], bins=30, alpha=0.6, label="Target y", color='orange')
# plt.title("Transformed x vs Target y")
# plt.legend()

# plt.savefig('fig.png')


pr_metrics = [dry_days, wet_days, R10mm, R20mm]


pr_marginal_bias_data = marginal.calculate_marginal_bias(metrics = pr_metrics, 
                                                         statistics = ['mean', 0.05, 0.95],
                                                        percentage_or_absolute = 'percentage',
                                                         obs = future_ref,
                                                         raw = future_prcp, 
                                                         QM = debiased_future)

pr_marginal_bias_plot = marginal.plot_marginal_bias(variable = 'pr', 
                                                    bias_df = pr_marginal_bias_data,
                                                   remove_outliers = True,
                                                   outlier_threshold_statistics = 10,
                                                   metrics_title = 'Absolute bias [days / year]',
                                                   statistics_title = 'Absolute bias [mm]')

pr_marginal_bias_plot.savefig('fig.png')


spelllength_dry = dry_days.calculate_spell_length(minimum_length= 3,obs = future_ref,
                                                         raw = future_prcp, 
                                                         QM = debiased_future)

spatiotemporal_dry = dry_days.calculate_spatiotemporal_clusters(obs = future_ref,
                                                         raw = future_prcp, 
                                                         QM = debiased_future)

spatial_dry = dry_days.calculate_spatial_extent(obs = future_ref,
                                                         raw = future_prcp, 
                                                         QM = debiased_future)

spatiotemporal_fig = marginal.plot_spatiotemporal(data = [spelllength_dry, spatiotemporal_dry, spatial_dry])

spatiotemporal_fig.savefig('fig1.png')