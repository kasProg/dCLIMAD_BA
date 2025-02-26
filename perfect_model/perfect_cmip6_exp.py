
import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import xarray as xr
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import numpy as np
from model.model import QuantileMappingModel
from model.loss import rainy_day_loss, distributional_loss_interpolated, compare_distributions, rmse, kl_divergence_loss, wasserstein_distance_loss, trend_loss
import data.process as process
from sklearn.preprocessing import StandardScaler
from ibicus.evaluate import assumptions, correlation, marginal, multivariate, trend
from ibicus.evaluate.metrics import *
from model.benchmark import BiasCorrectionBenchmark
import data.valid_crd as valid_crd
from torch.utils.data import DataLoader, TensorDataset

###-----The code is currently accustomed to CMIP6-Livneh Data format ----###

torch.manual_seed(42)
cuda_device = 0
device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
logging = True
cmip6_dir = '/pscratch/sd/k/kas7897/cmip6'
# ref_path = '/pscratch/sd/k/kas7897/cmip6/'

clim = 'access_cm2'
ref = 'gfdl_esm4'

input_x = ['precipitation']
clim_var = 'pr'
target_y = ['precipitation']
input_attrs = ['elevation']

scenario = 'historical'
# trend_future_period = [2075, 2099]

train_period = [1950, 1980]
# test_period = [1991, 2014]
test_period = [2075, 2099]
epochs = 200
testepoch = 40
benchmarking = True
train = True

# model params
model_type = 'ANN' #[SST, Poly2]
# resume = False
degree = 1 # degree of transformation
layers = 4 #number of layers to ANN
emph_quantile = 0.5

## loss params
w1 = 1
w2 = 0
# ny = 4 # number of params

##number of coordinates; if all then set to 'all'
num = 'all'
# slice = 0 #for spatial test; set 0 otherwise
batch_size = 50

seriesLst = input_x
attrLst =input_attrs


###-------- Developer section here -----------###

if logging:
    exp = f'conus/{clim}-{ref}/{model_type}_{layers}Layers_{degree}degree_quantile{emph_quantile}'
    writer = SummaryWriter(f"runs/{exp}")

if train:
    period = train_period
else:
    period = test_period

save_path = f'jobs/{clim}-{ref}/QM_{model_type}_layers{layers}_degree{degree}_quantile{emph_quantile}/{num}/{train_period[0]}_{train_period[1]}/'
model_save_path = save_path
if not train:
    if scenario == 'historical':
        save_path =  save_path + f'{test_period[0]}_{test_period[1]}/'
    else:
        save_path = save_path + f'{scenario}_{test_period[0]}_{test_period[1]}/'
    test_save_path = save_path + f'ep{testepoch}'
    os.makedirs(test_save_path, exist_ok=True)

os.makedirs(save_path, exist_ok=True)

######------------------Data Loading------------------#######

clim_data = xr.open_dataset(f'{cmip6_dir}/{clim}/{scenario}/{input_x[0]}/clipped_US.nc')
ref_data = xr.open_dataset(f'{cmip6_dir}/{ref}/{scenario}/{input_x[0]}/{clim}/clipped_US.nc')
valid_coords = valid_crd.valid_lat_lon(clim_data, clim_var)

x_data = []
x_var = clim_data[clim_var].sel(
                    lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                    lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                    method='nearest'
                ).sel(time=slice(f"{period[0]}", f"{period[1]}"))
time_x = x_var.time.values
x_var = x_var.values*86400
# managing units
# x_var = unit_identifier.convert(x_var, , units[matched_var]) 
x_var = torch.tensor(x_var).to(device)
x_data.append(x_var.unsqueeze(-1))
x_data = torch.cat(x_data, dim=-1).to(torch.float32)
torch.save(x_data, f'{save_path}/x.pt')
torch.save(time_x, f'{save_path}/time.pt')


y_data = []
y_var = ref_data[clim_var].sel(
                    lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                    lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                    method='nearest'
                ).sel(time=slice(f"{period[0]}", f"{period[1]}"))
time_y = y_var.time.values
y_var = y_var.values*86400
y_var = torch.tensor(y_var).to(device)
y_data.append(y_var.unsqueeze(-1))
y_data = torch.cat(y_data, dim=-1).to(x_data.dtype)
torch.save(y_data, f'{save_path}/y.pt')
torch.save(time_y, f'{save_path}/time_y.pt')

elev = xr.open_dataset(f'{cmip6_dir}/{clim}/elev.nc')
attrs_data = elev['elevation'].sel(
                    lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                    lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                    method='nearest'
                ).values
attrs_tensor = torch.tensor(attrs_data).unsqueeze(-1).to(device).to(x_data.dtype)

print("Normalizing data...")
if train:
    statDict = process.getStatDic(flow_regime=0, seriesLst=input_x, seriesdata=x_data, 
                                attrLst=input_attrs, attrdata=attrs_tensor)
    process.save_dict(statDict, f'{model_save_path}/statDict.json')
else:
    statDict = process.load_dict(f'{model_save_path}/statDict.json')

attr_norm = process.transNormbyDic(attrs_tensor, input_attrs, statDict, toNorm=True, flow_regime=0)
attr_norm[torch.isnan(attr_norm)] = 0.0
series_norm = process.transNormbyDic(x_data, input_x, statDict, toNorm=True, flow_regime=0)
series_norm[torch.isnan(series_norm)] = 0.0

attr_norm_tensor = attr_norm.unsqueeze(0).expand(series_norm.shape[0], -1, -1)
input_norm_tensor = torch.cat((series_norm, attr_norm_tensor), dim=2).permute(1, 0, 2)

x = x_data.squeeze().T
y = y_data.squeeze().T

dataset = TensorDataset(input_norm_tensor, x, y)
dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=False)

num_series = valid_coords.shape[0]
nx = len(input_x)+ len(input_attrs)


#######------------------Model Training/Testing------------------#######

if model_type == 'ANN':
    model = QuantileMappingModel(nx=nx, degree=degree, num_series=num_series, hidden_dim=64, num_layers=layers, modelType='ANN').to(device)
else:
    model = QuantileMappingModel(nx=nx, degree=degree, num_series=num_series, hidden_dim=64, num_layers=layers, modelType=model_type).to(device)
    

if train:
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    balance_loss = 0  # Adjust this weight to balance between distributional and rainy day losses

    # Training loop
    num_epochs = epochs
    loss_list = []
    for epoch in range(num_epochs+1):
        model.train()
        epoch_loss = 0
        
        loss1 = 0
        loss2 = 0
        loss3 = 0

        for batch_input_norm, batch_x, batch_y in dataloader:
            # Move batch to device
            batch_input_norm = batch_input_norm.to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            transformed_x = model(batch_x, batch_input_norm)

            # Compute the loss
            # kl_loss = 0.699*kl_divergence_loss(transformed_x.T, batch_y.T, num_bins=1000)

            dist_loss = w1*distributional_loss_interpolated(transformed_x.T, batch_y.T, device=device, num_quantiles=1000,  emph_quantile=emph_quantile)
            rainy_loss = w2*rainy_day_loss(transformed_x.T, batch_y.T)
            # ws_dist = 0.5*wasserstein_distance_loss(transformed_x.T, batch_y.T)
            # trendloss = trend_loss(transformed_x.T, batch_x.T, device)
            loss = dist_loss + rainy_loss 
            # loss = dist_loss + kl_loss + ws_dist + balance_loss * rainy_loss

            # loss = dist_loss + balance_loss * rainy_loss
            # loss = dist_loss + ws_dist + balance_loss * rainy_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loss1 += dist_loss.item()
            loss2 += rainy_loss.item()
            # loss3 += trendloss.item()


        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_loss1 = loss1 / len(dataloader)
        avg_epoch_loss2 = loss2 / len(dataloader)
        avg_epoch_loss3 = loss3 / len(dataloader)


        if logging:
            writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        loss_list.append(avg_epoch_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}, Average Loss1: {avg_epoch_loss1:.4f}, Average Loss2: {avg_epoch_loss2:.4f}, Average Loss3: {avg_epoch_loss3:.4f}')
            torch.save(model.state_dict(), f'{save_path}/model_{epoch}.pth')

    plt.plot(loss_list)
    plt.title('Loss Curve')
    plt.show()

else:
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
            predictions = model(batch_x, batch_input_norm)

            # Store predictions
            transformed_x.append(predictions.cpu())

            y.append(batch_y.cpu())
            x.append(batch_x.cpu())
            

    ## no batch exp
    # transformed_x = model(x, input_norm_tensor).cpu().detach().numpy()
    # x = x.cpu().detach().numpy()
    # y = y.cpu().detach().numpy()
   

    transformed_x = torch.cat(transformed_x, dim=0).numpy().T
    # transformed_x_nc = valid_crd.reconstruct_nc(transformed_x/86400, valid_coords, time_x, input_x['precipitation'][0])
    # transformed_x_nc.to_netcdf(f'{test_save_path}/xt.nc')
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
        loca = xr.open_dataset(f'{cmip6_dir}/{clim}/{scenario}/precipitation/loca/coarse_USclip.nc')
        loca = loca[clim_var].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                                            lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                                            method='nearest')
        loca = loca.sel(time =slice(f'{period[0]}', f'{period[1]}')).values


        
        QM_bench = f'benchmark/QuantileMapping/conus/{clim}-{ref}/{train_period}_{scenario}_{test_period}.pt'
        if os.path.exists(QM_bench):
            QM_debiased = torch.load(QM_bench, weights_only=False)
        else:
            bench = BiasCorrectionBenchmark(clim = clim,
                                            ref = ref,
                                            hist_period = train_period, 
                                            test_period = test_period, 
                                            scenario = scenario, 
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
                                                                diffDownscale = transformed_x)

        spatiotemporal_dry = dry_days.calculate_spatiotemporal_clusters(obs = y,
                                                                raw = x, 
                                                                QM = QM_debiased,
                                                                LOCA2 = loca,
                                                                diffDownscale = transformed_x)

        spatial_dry = dry_days.calculate_spatial_extent(obs = y,
                                                        raw = x, 
                                                        QM = QM_debiased,
                                                        LOCA2 = loca,
                                                        diffDownscale = transformed_x)

        spatiotemporal_fig = marginal.plot_spatiotemporal(data = [spelllength_dry, spatiotemporal_dry, spatial_dry])

        spatiotemporal_fig.savefig(f'{test_save_path}/ibicus_fig1.png')


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

            writer.close()


