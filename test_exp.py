 
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import xarray as xr
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import numpy as np
from model.model import QuantileMappingModel_, QuantileMappingModel, QuantileMappingModel_Poly2
from model.loss import rainy_day_loss, distributional_loss_interpolated, compare_distributions, rmse, kl_divergence_loss, wasserstein_distance_loss, trend_loss
import data.process as process
from sklearn.preprocessing import StandardScaler
from ibicus.evaluate import assumptions, correlation, marginal, multivariate, trend
from ibicus.evaluate.metrics import *
from data.loader import DataLoaderWrapper
from model.benchmark import BiasCorrectionBenchmark
import data.valid_crd as valid_crd
import argparse
import yaml
import data.helper as helper


###-----The code is currently accustomed to CMIP6-Livneh Data format ----###


parser = argparse.ArgumentParser(description='Quantile Mapping Model Configuration')

parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--logging', action='store_true')
parser.add_argument('--clim', type=str, default='miroc6')
parser.add_argument('--ref', type=str, default='livneh')
parser.add_argument('--train', action='store_true')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--model_type', type=str, default='ANN')
parser.add_argument('--degree', type=int, default=1)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--time_scale', type=str, default='seasonal')
parser.add_argument('--emph_quantile', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--testepoch', type=int, default=50)
parser.add_argument('--num', type=str, default='all')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--trend_analysis', action='store_true')
parser.add_argument('--benchmarking', action='store_true')
parser.add_argument('--train_start', type=int, default=1950)
parser.add_argument('--train_end', type=int, default=1980)
parser.add_argument('--test_start', type=int, default=1981)
parser.add_argument('--test_end', type=int, default=1995)
parser.add_argument('--scneario', type=str, default='ssp5_8_5')
parser.add_argument('--trend_start', type=int, default=2075)
parser.add_argument('--trend_end', type=int, default=2099)

args = parser.parse_args()

torch.manual_seed(42)
cuda_device = args.cuda_device

logging = args.logging


clim = args.clim
ref = args.ref
train = args.train
# validation = args.validation


### FOR TREND ANALYSIS
trend_analysis = args.trend_analysis
scenario = 'ssp5_8_5'
trend_future_period = [args.trend_start, args.trend_analysis]


train_period = [args.train_start, args.train_end]
test_period = [args.test_start, args.test_end]
epochs = args.epochs
testepoch = args.testepoch
benchmarking = args.benchmarking

# model params
model_type = args.model_type
# resume = False
degree = args.degree
layers = args.layers
time_scale = args.time_scale
emph_quantile = args.emph_quantile


##number of coordinates; if all then set to 'all'
num = args.num
# slice = 0 #for spatial test; set 0 otherwise
batch_size = args.batch_size



####------FIXED INPUTS------------####

if cuda_device == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
    else:
        raise RuntimeError(f"CUDA device {cuda_device} requested but CUDA is not available.")


cmip6_dir = '/pscratch/sd/k/kas7897/cmip6'
ref_path = '/pscratch/sd/k/kas7897/Livneh/unsplit/'

input_x = {'precipitation': ['pr', 'prec', 'prcp' 'PRCP', 'precipitation']}
clim_var = 'pr'
target_y = {'precipitation': ['pr', 'prec', 'prcp', 'PRCP', 'precipitation']}
input_attrs = {'elevation': ['elev', 'elevation']}

## loss params
w1 = 1
w2 = 0
# ny = 4 # number of params

seriesLst = input_x.keys()
attrLst =input_attrs.keys()


###-------- Developer section here -----------###

if logging:
    exp = f'conus/{clim}/{model_type}_{layers}Layers_{degree}degree_quantile{emph_quantile}_scale{time_scale}'
    writer = SummaryWriter(f"runs/{exp}")


save_path = f'jobs/{clim}-{ref}/QM_{model_type}_layers{layers}_degree{degree}_quantile{emph_quantile}_scale{time_scale}/{num}/{train_period[0]}_{train_period[1]}/'
model_save_path = save_path

val_save_path =  save_path + f'{test_period[0]}_{test_period[1]}/'

test_save_path = val_save_path + f'ep{testepoch}'
os.makedirs(test_save_path, exist_ok=True)

args_dict = vars(args)  # Converts argparse Namespace into a dictionary
with open(os.path.join(test_save_path, "test_config.yaml"), "w") as f:
    yaml.dump(args_dict, f)

data_loader = DataLoaderWrapper(
    clim=clim, scenario='historical', ref=ref, period=test_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, target_y=target_y, save_path=save_path, stat_save_path = model_save_path,
    crd='all', batch_size=batch_size, train=train, device=device)

dataloader = data_loader.get_dataloader()

if trend_analysis:
    future_save_path = model_save_path + f'{scenario}_{trend_future_period[0]}_{trend_future_period[1]}/'
    os.makedirs(future_save_path, exist_ok=True)
    data_loader_future = DataLoaderWrapper(
    clim=clim, scenario = scenario, ref=ref, period=trend_future_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, target_y={}, save_path=future_save_path, stat_save_path = model_save_path, 
    crd='all', batch_size=100, train=train, device=device)

    dataloader_future = data_loader_future.get_dataloader_future()
    with open(os.path.join(future_save_path, "future_config.yaml"), "w") as f:
        yaml.dump(args_dict, f)

valid_coords = data_loader.get_valid_coords()
_, time_x = data_loader.load_dynamic_inputs()
nx = len(input_x)+ len(input_attrs)

if time_scale!= 'daily':
    time_labels = helper.extract_time_labels(time_x, label_type=time_scale)
    if trend_analysis:
        _, time_x_future = data_loader_future.load_dynamic_inputs()
        time_labels_future = helper.extract_time_labels(time_x_future, label_type=time_scale)
else:
    time_labels = 'daily'
    time_labels_future = 'daily'


model = QuantileMappingModel(nx=nx, degree=degree, hidden_dim=64, num_layers=layers, modelType=model_type).to(device)


model.load_state_dict(torch.load(f'{model_save_path}/model_{testepoch}.pth', weights_only=True))
model.eval()
transformed_x = []
transformed_x_future = []
x_future = []
x = []
y = []
with torch.no_grad():
    for batch in dataloader:
        batch_input_norm, batch_x, batch_y = [b.to(device) for b in batch]

        # Forward pass
        predictions = model(batch_x, batch_input_norm, time_labels)

        # Store predictions
        transformed_x.append(predictions.cpu())

        y.append(batch_y.cpu())
        x.append(batch_x.cpu())
    
    if trend_analysis:
        for batch in dataloader_future:
            batch_input_norm, batch_x = [b.to(device) for b in batch]

            # Forward pass
            predictions = model(batch_x, batch_input_norm, time_labels_future)

            # Store predictions
            transformed_x_future.append(predictions.cpu())

            x_future.append(batch_x.cpu())
        

## no batch exp
# transformed_x = model(x, input_norm_tensor).cpu().detach().numpy()
# x = x.cpu().detach().numpy()
# y = y.cpu().detach().numpy()


transformed_x = torch.cat(transformed_x, dim=0).numpy().T
transformed_x_nc = valid_crd.reconstruct_nc(transformed_x, valid_coords, time_x, input_x['precipitation'][0])
transformed_x_nc.to_netcdf(f'{test_save_path}/xt.nc')
x = torch.cat(x, dim=0).numpy().T
y = torch.cat(y, dim=0).numpy().T        

torch.save(transformed_x, f'{test_save_path}/xt.pt')
avg_improvement, individual_improvements = compare_distributions(transformed_x, x, y)

quantile_rmse_model = torch.sqrt(distributional_loss_interpolated(torch.tensor(x), torch.tensor(y), device='cpu', num_quantiles=1000, emph_quantile=None))
quantile_rmse_bs = torch.sqrt(distributional_loss_interpolated(torch.tensor(transformed_x), torch.tensor(y), device='cpu', num_quantiles=1000, emph_quantile=None))
print(f"Average distribution improvement: {avg_improvement:.4f}")

print(f"Quantile RMSE between Model and Target: {quantile_rmse_model}")
print(f"Quantile RMSE between Corrected and Target: {quantile_rmse_bs}")
print(f"Quantile RMSE Improvement: {quantile_rmse_model - quantile_rmse_bs}")

if benchmarking:
    print("processing LOCA for benchmarking...")
    loca = xr.open_dataset(f'{cmip6_dir}/{clim}/historical/precipitation/loca/coarse_USclip.nc')
    loca = loca[input_x['precipitation'][0]].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                                        lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                                        method='nearest')
    loca = loca.sel(time =slice(f'{test_period[0]}', f'{test_period[1]}')).values


    
    QM_bench = f'benchmark/QuantileMapping/conus/{clim}-{ref}/{train_period}_historical_{test_period}.pt'
    if os.path.exists(QM_bench):
        QM_debiased = torch.load(QM_bench, weights_only=False)
    else:
        bench = BiasCorrectionBenchmark(clim = clim,
                                        ref = ref,
                                        hist_period = train_period, 
                                        test_period = test_period, 
                                        scenario = 'historical', 
                                        clim_var = clim_var, 
                                        correction_methods = ['QuantileMapping'],  
                                        model_path = model_save_path, 
                                        test_path = save_path)  
        bench.apply_correction()
        QM_debiased = torch.load(QM_bench, weights_only=False)

    x = np.expand_dims(x, axis=-1)
    loca = np.expand_dims(loca, axis=-1)
    y = np.expand_dims(y, axis=-1)
    transformed_x = np.expand_dims(transformed_x, axis=-1)

    #ibicus plots
    pr_metrics = [dry_days, wet_days, R10mm]

    x = x/86400
    y = y/86400
    transformed_x = transformed_x/86400
    # QM_debiased =  QM_debiased/ 86400
    

    pr_marginal_bias_data = marginal.calculate_marginal_bias(metrics = pr_metrics, 
                                                            statistics = ['mean', 0.95],
                                                            percentage_or_absolute = 'percentage',
                                                            obs = y,
                                                            raw = x, 
                                                            QM = QM_debiased,
                                                            LOCA2 = loca,
                                                            diffDownscale = transformed_x)

    pr_marginal_bias_plot = marginal.plot_marginal_bias(variable = 'pr', 
                                                        bias_df = pr_marginal_bias_data,
                                                    remove_outliers = True,
                                                    outlier_threshold_statistics = 10,
                                                    metrics_title = 'Percentage bias [days]',
                                                    statistics_title = 'Percentage bias')

    pr_marginal_bias_plot.savefig(f'{test_save_path}/ibicus_fig.png')

    spelllength_dry = dry_days.calculate_spell_length(minimum_length= 3, obs = y,
                                                            raw = x, 
                                                            QM = QM_debiased,
                                                            LOCA2 = loca, 
                                                            delCLIMD_BA = transformed_x)

    spatiotemporal_dry = dry_days.calculate_spatiotemporal_clusters(obs = y,
                                                            raw = x, 
                                                            QM = QM_debiased,
                                                            LOCA2 = loca,
                                                            delCLIMD_BA = transformed_x)

    spatial_dry = dry_days.calculate_spatial_extent(obs = y,
                                                    raw = x, 
                                                    QM = QM_debiased,
                                                    LOCA2 = loca,
                                                    delCLIMD_BA = transformed_x)

    spatiotemporal_fig = marginal.plot_spatiotemporal(data = [spelllength_dry, spatiotemporal_dry, spatial_dry])

    spatiotemporal_fig.savefig(f'{test_save_path}/ibicus_fig1.png')


    if trend_analysis:
        transformed_x_future = torch.cat(transformed_x_future, dim=0).numpy().T
        torch.save(transformed_x_future, f'{future_save_path}/xt.pt')
        x_future = torch.cat(x_future, dim=0).numpy().T

        
        QM_bench_future = f'benchmark/QuantileMapping/conus/{clim}-{ref}/{train_period}_{scenario}_{trend_future_period}.pt'
        if os.path.exists(QM_bench_future):
            QM_debiased_future = torch.load(QM_bench_future, weights_only=False)
        else:
            bench = BiasCorrectionBenchmark(clim = clim,
                                        ref = ref,
                                        hist_period = train_period, 
                                        test_period = trend_future_period, 
                                        scenario = scenario, 
                                        clim_var = clim_var, 
                                        correction_methods = ['QuantileMapping'],  
                                        model_path = model_save_path, 
                                        test_path = future_save_path)
            bench.apply_correction()
            QM_debiased_future = torch.load(QM_bench_future, weights_only=False) 
    

        loca_future = xr.open_dataset(f'{cmip6_dir}/{clim}/{scenario}/precipitation/loca/coarse_USclip.nc')
        loca_future = loca_future[input_x['precipitation'][0]].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                                        lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                                        method='nearest')
        loca_future = loca_future.sel(time =slice(f'{trend_future_period[0]}', f'{trend_future_period[1]}')).values

        loca_future = np.expand_dims(loca_future, axis=-1)
        transformed_x_future = np.expand_dims(transformed_x_future, axis=-1)
        x_future = np.expand_dims(x_future, axis=-1)

        # QM_debiased_future = QM_debiased_future/86400
        x_future = x_future/86400
        transformed_x_future = transformed_x_future/86400

        trend_bias_data = trend.calculate_future_trend_bias(statistics = ["mean"], 
                                                            trend_type = 'additive',
                                                    raw_validate = x, raw_future = x_future,
                                                            metrics = pr_metrics,
                                                    QM = [QM_debiased, QM_debiased_future],
                                                    LOCA2 = [loca, loca_future],
                                                            delCLIMD_BA = [transformed_x, transformed_x_future])

        trend_plot = trend.plot_future_trend_bias_boxplot(variable ='pr', 
                                                        bias_df = trend_bias_data, 
                                                        remove_outliers = True,
                                                                outlier_threshold = 500)
        
        trend_plot.savefig(f'{future_save_path}/ibicus_fig2.png')

    if logging:
        writer.add_text(
            "Evaluation Metrics",
            f"""
            Average distribution improvement: {avg_improvement:.4f}\n
            Quantile RMSE between Model and Target: {quantile_rmse_model}\n 
            Quantile RMSE between Corrected and Target: {quantile_rmse_bs}\n
            Quantile RMSE Improvement: {quantile_rmse_model - quantile_rmse_bs}\n            
            """,
            0
            )       
    
        writer.add_figure("Figure 1", pr_marginal_bias_plot, global_step=epochs)


        writer.add_figure("Figure 2", spatiotemporal_fig, global_step=epochs)

        if trend_analysis:
            writer.add_figure("Figure 3", trend_plot, global_step=epochs)

        writer.close()


