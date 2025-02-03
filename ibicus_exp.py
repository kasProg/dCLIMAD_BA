import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import xarray as xr
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import numpy as np
import pickle
import valid_crd
import tqdm
from model import QuantileMappingModel_, QuantileMappingModel, QuantileMappingModel_Poly2
from loss import rainy_day_loss, distributional_loss_interpolated, compare_distributions, rmse, kl_divergence_loss, wasserstein_distance_loss, trend_loss
from data import process_data, getStatDic
from torch.utils.data import DataLoader, TensorDataset
import data
from sklearn.preprocessing import StandardScaler
from ibicus.evaluate import assumptions, correlation, marginal, multivariate, trend
from ibicus.evaluate.metrics import *

###-----The code is currently accustomed to Livneh Data format ----###

torch.manual_seed(42)
device = torch.device('cuda:0')
logging = False

# clim = 'ipsl_cm6a_lr'
clim = 'mpi_esm1_2_lr'
dataset = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/obs_hist.nc'
clim_model = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/raw_hist.nc'
QM_debiased = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/QM_val.nc'
QM_debiased_future = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/QM_fut.nc'

elev_path = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/elev.nc'
##if running synthetic case
# noise_type = 'R5noisy001d'
noise_type = clim
ref = 'era5'
# noise_type = 'upscale_1by4_bci'
# train_period = [1959, 1989]
train_period = [1980, 1990]
# test_period = [1980, 1990]
test_period = [1991, 1995]
# test_period = [1990, 2005]

train = 0 # training = 1; else test
seriesLst = ['noisy_prcp']
# seriesLst = ['noisy_prcp']
attrLst = ['elev']
epochs = 200
testepoch = 20
testmod = 6

if train!=1:
    dataset = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/obs_validate.nc'
    clim_model = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/raw_val.nc'
x_future = f'/data/kas7897/data/ibicus_GMD_submission/data/pr/{clim}/raw_fut.nc'
# model params
model_type = 'SST' #[SST, Poly2]
resume = False
degree = 1 # degree of transformation
layers = 4 #number of layers to ANN
emph_quantile = 0.5
w1 = 1
w2 = 0
# ny = 4 # number of params

##number of coordinates; if all then set to 'all'
num = 'all'
slice = 0 #for spatial test; set 0 otherwise
batch_size = 100

if logging:
    exp = f'{clim}/{model_type}_{layers}Layers_{degree}degree_mod9'
    writer = SummaryWriter(f"runs/{exp}")

###-------- Developer section here -----------###

save_path = f'models/{clim}-{ref}/QM_{model_type}_layers{layers}_degree{degree}_mod9/{num}/{train_period[0]}_{train_period[1]}/{noise_type}/'
os.makedirs(save_path, exist_ok=True)

#extracting valid lat-lon pairs with non-nan prcp
ds_sample = xr.open_dataset(f"{dataset}")
valid_coords = valid_crd.valid_lat_lon(ds_sample, var_name='__xarray_dataarray_variable__')


#processing elevation data
x = xr.open_dataset(clim_model)
x_future = xr.open_dataset(x_future)
y = xr.open_dataset(dataset)
elev = xr.open_dataset(elev_path)
if num == 'all':
    elev_data = elev['elevation'].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                                      lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                                      method='nearest').values
    x = x['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[:, 0], dims='points'),
                                      longitude=xr.DataArray(valid_coords[:, 1], dims='points'),
                                      method='nearest').values*86400 #converting mm/day
    y = y['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[:, 0], dims='points'),
                                      longitude=xr.DataArray(valid_coords[:, 1], dims='points'),
                                      method='nearest').values*86400 #converting mm/day
    x_future = x_future['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[:, 0], dims='points'),
                                      longitude=xr.DataArray(valid_coords[:, 1], dims='points'),
                                      method='nearest').values*86400 #converting mm/day
else:
    elev_data = elev['elevation'].sel(lat=xr.DataArray(valid_coords[slice:num, 0], dims='points'),
                                 lon=xr.DataArray(valid_coords[slice:num, 1], dims='points'),
                                 method='nearest').values*86400 #converting mm/day
    x = x['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[slice:num, 0], dims='points'),
                                      longitude=xr.DataArray(valid_coords[slice:num, 1], dims='points'),
                                      method='nearest').values*86400 #converting mm/day
    y = y['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[slice:num, 0], dims='points'),
                                      longitude=xr.DataArray(valid_coords[slice:num, 1], dims='points'),
                                      method='nearest').values*86400 #converting mm/day
    x_future = x_future['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[slice:num, 0], dims='points'),
                                      longitude=xr.DataArray(valid_coords[slice:num, 1], dims='points'),
                                      method='nearest').values*86400 #converting mm/day

x = torch.tensor(x).to(device)
y = torch.tensor(y).to(device)
x_future = torch.tensor(x_future).to(device)
pathx = clim_model + noise_type

if train==1:
    period = train_period
else:
    period = test_period


# if os.path.exists(f'{clim_model}QM_input/x{period}{num}_{noise_type}.pt'):
#     print('loading x...')
#     x = torch.load(f'{clim_model}QM_input/x{period}{num}_{noise_type}.pt', weights_only=False).to(device)
# else:
#     print("processing x data...")
#     x = process_data(pathx, period, valid_coords, num, device, var='prec')
#     torch.save(x, f'{clim_model}QM_input/x{period}{num}_{noise_type}.pt')

# if os.path.exists(f'{dataset}QM_input/y{period}{num}.pt'):
#     print('loading y...')
#     y = torch.load(f'{dataset}QM_input/y{period}{num}.pt', weights_only=False).to(device)
# else:
#     print("processing y data...")
#     y = process_data(dataset, period, valid_coords, num, device, var='prec')
#     torch.save(y, f'{dataset}QM_input/y{period}{num}.pt')

# x = x[:,slice:num]
# y = y[:,slice:num]

x_in = x.unsqueeze(-1)
x_in_future = x_future.unsqueeze(-1)

#addigng wind_data
if 'wind' in seriesLst:
    if os.path.exists(f'{dataset}/wind/QM_input/wind{period}{num}.pt'):
        print('loading wind....')
        wind = torch.load(f'{dataset}/wind/QM_input/wind{period}{num}.pt', weights_only=False).to(device)
    else:
        print('processing wind...')
        wind = process_data(f'{dataset}/wind', period, valid_coords, num, device, var='wind')
        torch.save(wind, f'{dataset}/wind/QM_input/wind{period}{num}.pt')
        wind = torch.tensor(wind)

    wind = wind[:, slice:num]
    wind_tensor = wind.to(x.dtype).to(device)
    wind_tensor = wind_tensor.unsqueeze(-1)
    x_in = torch.cat((x_in, wind_tensor), dim=2)

elev_tensor = torch.tensor(elev_data).to(x.dtype).to(device)
attr_tensor = elev_tensor.unsqueeze(-1)


if logging:
    ## For tensorboard
    writer.add_text(
        "Hyperparameters",
        f"""
        Noise Type: {noise_type}\n
        Training Period: {train_period}\n
        Testing Period: {test_period}\n
        Num Coordinates: {num}\n
        Model Type: {model_type}
        """,
        0
    )
    ##

if train == 1:
    statDict = getStatDic(flow_regime = 0, seriesLst = seriesLst, seriesdata = x_in, attrLst = attrLst, attrdata = attr_tensor)
    data.save_dict(statDict, f'{save_path}/statDict.json')
    # save statDict
else:
    statDict = data.load_dict(f'{save_path}/statDict.json')
    # load StatDict

attr_norm = data.transNormbyDic(attr_tensor, attrLst, statDict, toNorm=True, flow_regime=0)
attr_norm[torch.isnan(attr_norm)] = 0.0
series_norm = data.transNormbyDic(
    x_in, seriesLst, statDict, toNorm=True, flow_regime= 0
)

series_norm_future = data.transNormbyDic(
    x_in_future, seriesLst, statDict, toNorm=True, flow_regime= 0
)

series_norm[torch.isnan(series_norm)] = 0.0
series_norm_future[torch.isnan(series_norm_future)] = 0.0


attr_norm_tensor = attr_norm.unsqueeze(0).expand(series_norm.shape[0], -1, -1)
attr_norm_tensor_future = attr_norm.unsqueeze(0).expand(series_norm_future.shape[0], -1, -1)
input_norm_tensor = torch.cat((series_norm, attr_norm_tensor), dim=2)
input_norm_tensor_future = torch.cat((series_norm_future, attr_norm_tensor_future), dim=2)


torch.set_default_dtype(x.dtype)

#for batches
x = x.T
y = y.T
x_future = x_future.T
input_norm_tensor = input_norm_tensor.permute(1, 0, 2)
input_norm_tensor_future = input_norm_tensor_future.permute(1, 0, 2)

dataset = TensorDataset(input_norm_tensor, x, y)
dataset_future = TensorDataset(input_norm_tensor_future, x_future, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
dataloader_future = DataLoader(dataset_future, batch_size=batch_size, shuffle=False)


num_series = x.shape[0]
nx = x_in.shape[-1] + attr_tensor.shape[-1]


## Choose model

if model_type == 'SST':
    model = QuantileMappingModel(nx=nx, degree=degree, num_series=num_series, hidden_dim=64, num_layers=layers, modelType='ANN').to(device)
else:
    model = QuantileMappingModel(nx=nx, degree=degree, num_series=num_series, hidden_dim=64, num_layers=layers, modelType=model_type).to(device)


if resume:
    state_dict = torch.load(f'{save_path}model_{testepoch}.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    

if train == 1:
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    balance_loss = 0  # Adjust this weight to balance between distributional and rainy day losses

    # Training loop
    num_epochs = epochs
    loss_list = []
    for epoch in range(testepoch ,num_epochs+1):
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
    model.load_state_dict(torch.load(f'{save_path}/model_{testepoch}.pth', weights_only=True))
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

        for batch in dataloader_future:
            batch_input_norm, batch_x, batch_y = [b.to(device) for b in batch]

            # Forward pass
            predictions = model(batch_x, batch_input_norm)

            # Store predictions
            transformed_x_future.append(predictions.cpu())

            # y.append(batch_y.cpu())
            x_future.append(batch_x.cpu())


    ## no batch exp
    # transformed_x = model(x, input_norm_tensor).cpu().detach().numpy()
    # x = x.cpu().detach().numpy()
    # y = y.cpu().detach().numpy()

    transformed_x = torch.cat(transformed_x, dim=0).numpy().T
    transformed_x_future = torch.cat(transformed_x_future, dim=0).numpy().T
    x = torch.cat(x, dim=0).numpy().T
    x_future = torch.cat(x_future, dim=0).numpy().T
    y = torch.cat(y, dim=0).numpy().T


    test_save_path = f'{save_path}/{testepoch}'
    os.makedirs(test_save_path, exist_ok=True)

    torch.save(transformed_x, f'{test_save_path}/xt.pt')
    torch.save(x, f'{save_path}/x.pt')
    torch.save(y, f'{save_path}/y.pt')
    avg_improvement, individual_improvements = compare_distributions(transformed_x, x, y)

    quantile_rmse_model = torch.sqrt(distributional_loss_interpolated(torch.tensor(x), torch.tensor(y), device='cpu', num_quantiles=1000, emph_quantile=None))
    quantile_rmse_bs = torch.sqrt(distributional_loss_interpolated(torch.tensor(transformed_x), torch.tensor(y), device='cpu', num_quantiles=1000, emph_quantile=None))
    print(f"Average distribution improvement: {avg_improvement:.4f}")

    print(f"Quantile RMSE between Model and Target: {quantile_rmse_model}")
    print(f"Quantile RMSE between Corrected and Target: {quantile_rmse_bs}")
    print(f"Quantile RMSE Improvement: {quantile_rmse_model - quantile_rmse_bs}")
    # print(f'RMSE between Model and Target: {np.median(rmse(x, y))}')
    # print(f'RMSE between Transformed and Target: {np.median(rmse(transformed_x, y))}')

    QM_debiased = xr.open_dataset(QM_debiased)
    QM_debiased = QM_debiased['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[:, 0], dims='points'),
                                    longitude=xr.DataArray(valid_coords[:, 1], dims='points'),
                                    method='nearest').values 
    QM_debiased = np.expand_dims(QM_debiased, axis=-1)

    QM_debiased_future = xr.open_dataset(QM_debiased_future)
    QM_debiased_future = QM_debiased_future['__xarray_dataarray_variable__'].sel(latitude=xr.DataArray(valid_coords[:, 0], dims='points'),
                                    longitude=xr.DataArray(valid_coords[:, 1], dims='points'),
                                    method='nearest').values
    QM_debiased_future = np.expand_dims(QM_debiased_future, axis=-1)

    x = np.expand_dims(x, axis=-1)
    x_future = np.expand_dims(x_future, axis=-1)

    y = np.expand_dims(y, axis=-1)
    # ANN_debiased = np.expand_dims(ANN_debiased, axis=-1)
    transformed_x = np.expand_dims(transformed_x, axis=-1)
    transformed_x_future = np.expand_dims(transformed_x_future, axis=-1)

    #ibicus plots
    pr_metrics = [dry_days, wet_days, R10mm]

    x = x/86400
    x_future = x_future/86400
    y = y/86400
    transformed_x = transformed_x/86400
    transformed_x_future = transformed_x_future/86400

    # ANN_debiased = ANN_debiased/86400

    pr_marginal_bias_data = marginal.calculate_marginal_bias(metrics = pr_metrics, 
                                                            statistics = ['mean', 0.95],
                                                            percentage_or_absolute = 'percentage',
                                                            obs = y,
                                                            raw = x, 
                                                            QM = QM_debiased,
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
                                                            diffDownscale = transformed_x)

    spatiotemporal_dry = dry_days.calculate_spatiotemporal_clusters(obs = y,
                                                            raw = x, 
                                                            QM = QM_debiased,
                                                            diffDownscale = transformed_x)

    spatial_dry = dry_days.calculate_spatial_extent(obs = y,
                                                    raw = x, 
                                                    QM = QM_debiased,
                                                    diffDownscale = transformed_x)

    spatiotemporal_fig = marginal.plot_spatiotemporal(data = [spelllength_dry, spatiotemporal_dry, spatial_dry])

    spatiotemporal_fig.savefig(f'{test_save_path}/ibicus_fig1.png')

    trend_bias_data = trend.calculate_future_trend_bias(statistics = ["mean"], 
                                                        trend_type = 'additive',
                                                  raw_validate = x, raw_future = x_future,
                                                        metrics = pr_metrics,
                                                  QM = [QM_debiased, QM_debiased_future],
                                                        diffDownscale = [transformed_x, transformed_x_future])

    trend_plot = trend.plot_future_trend_bias_boxplot(variable ='pr', 
                                                      bias_df = trend_bias_data, 
                                                      remove_outliers = True,
                                                            outlier_threshold = 500)
    
    trend_plot.savefig(f'{test_save_path}/ibicus_fig2.png')

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
        
        
        # y[y==0] = np.nan
        # x[x==0] = np.nan
        # transformed_x[transformed_x==0] = np.nan
        
        # if 'W' in noise_type:
        #     n = x - y
        #     ln = x - transformed_x
        # elif 'R' in noise_type:
        #     n = x/y
        #     ln = x/transformed_x
        # else:
        #     n = x - y
        #     ln = x - transformed_x

        
        # # Simple plot
        # elev_broadcast = np.repeat(elev_data[np.newaxis, :], x.shape[0], axis=0)
        # data.log_plots_to_tensorboard(writer, y, x, transformed_x, elev_broadcast, n, ln, exp, epoch=0)
        # writer.close()

        ## comparing with benchmark
        # QM_debiased = torch.load(f'benchmark/QM_ibicus/{test_period}{num}.pt', weights_only=False)
        # ANN_debiased = torch.load(f'models/{clim}-{ref}/QM_SST_layers{layers}_degree{degree}_mod{testmod}/{num}/1980_1990/{noise_type}/xt.pt', weights_only=False)
        
        writer.add_figure("Figure 1", pr_marginal_bias_plot, global_step=epochs)


        writer.add_figure("Figure 2", spatiotemporal_fig, global_step=epochs)


        # tas_trend_bias_data = trend.calculate_future_trend_bias(statistics = ["mean", 0.05, 0.95], 
        #                                                 trend_type = 'additive',
        #                                           raw_validate = tas_cm_validate, raw_future = tas_cm_future,
        #                                                 metrics = tas_metrics,
        #                                           QM = [tas_val_debiased_QM, tas_fut_debiased_QM],
        #                                                 QDM = [tas_val_debiased_QM, tas_fut_debiased_QDM],
        #                                           ISIMIP = [tas_val_debiased_ISIMIP, tas_fut_debiased_ISIMIP],
        #                                           CDFT = [tas_val_debiased_CDFT, tas_fut_debiased_CDFT])

        # tas_trend_plot = trend.plot_future_trend_bias_boxplot(variable ='tas', bias_df = tas_trend_bias_data,                                                         remove_outliers = True,
        #                                                         outlier_threshold = 500)
        writer.close()



    # best_ind, best_improv = max(enumerate(individual_improvements), key=lambda x: x[1])

    # # Step 6: Plotting the original and transformed distributions
    # plt.figure(figsize=(12, 6))
    # plt.suptitle(f'WS Distance Improvement:{best_improv}')
    # plt.subplot(1, 2, 1)
    # plt.hist(x[:, best_ind], bins=30, alpha=0.6, label="Noisy")
    # plt.hist(y[:, best_ind], bins=30, alpha=0.6, label="Target Y", color='orange')
    # plt.title("Noisy x and Target Distributions")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.hist(transformed_x[:, best_ind], bins=30, alpha=0.6, label="Transformed x", color='green')
    # plt.hist(y[:, best_ind], bins=30, alpha=0.6, label="Target y", color='orange')
    # plt.title("Transformed x vs Target y")
    # plt.legend()

    # plt.show(

# best_ind, best_improv = min(enumerate(individual_improvements), key=lambda x: x[1])

# # Step 6: Plotting the original and transformed distributions
# plt.figure(figsize=(12, 6))
# # plt.suptitle(f'WS Distance Improvement:{best_improv}')
# plt.subplot(1, 2, 1)
# plt.hist(x[:, 500], bins=30, alpha=0.6, label="Noisy")
# plt.hist(y[:, 500], bins=30, alpha=0.6, label="Target Y", color='orange')
# plt.title("Noisy x and Target Distributions")
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.hist(transformed_x[:, 500], bins=30, alpha=0.6, label="Transformed x", color='green')
# plt.hist(y[:, 500], bins=30, alpha=0.6, label="Target y", color='orange')
# plt.title("Transformed x vs Target y")
# plt.legend()
#
# plt.show()
#
# # Step 6: Plotting the original and transformed distributions
# plt.figure(figsize=(12, 6))
# # plt.suptitle(f'WS Distance Improvement:{best_improv}')
# plt.subplot(1, 2, 1)
# plt.hist(x[:, 100], bins=30, alpha=0.6, label="Noisy")
# plt.hist(y[:, 100], bins=30, alpha=0.6, label="Target Y", color='orange')
# plt.title("Noisy x and Target Distributions")
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.hist(transformed_x[:, 500], bins=30, alpha=0.6, label="Transformed x", color='green')
# plt.hist(y[:, 100], bins=30, alpha=0.6, label="Target y", color='orange')
# plt.title("Transformed x vs Target y")
# plt.legend()
#
# plt.show()
#
# # Step 6: Plotting the original and transformed distributions
# plt.figure(figsize=(12, 6))
# # plt.suptitle(f'WS Distance Improvement:{best_improv}')
# plt.subplot(1, 2, 1)
# plt.hist(x[:, 1000], bins=30, alpha=0.6, label="Noisy")
# plt.hist(y[:, 1000], bins=30, alpha=0.6, label="Target Y", color='orange')
# plt.title("Noisy x and Target Distributions")
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.hist(transformed_x[:, 1000], bins=30, alpha=0.6, label="Transformed x", color='green')
# plt.hist(y[:,1000], bins=30, alpha=0.6, label="Target y", color='orange')
# plt.title("Transformed x vs Target y")
# plt.legend()
#
# plt.show()
#
# # Step 6: Plotting the original and transformed distributions
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.hist(x[:, 1500], bins=30, alpha=0.6, label="Noisy")
# plt.hist(y[:, 1500], bins=30, alpha=0.6, label="Target Y", color='orange')
# plt.title("Noisy x and Target Distributions")
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.hist(transformed_x[:, 1500], bins=30, alpha=0.6, label="Transformed x", color='green')
# plt.hist(y[:, 1500], bins=30, alpha=0.6, label="Target y", color='orange')
# plt.title("Transformed x vs Target y")
# plt.legend()
#
# plt.show()

# Print skewness to verify positive skew
# print("Skewness of original x: ", skew(x.cpu().detach().numpy()))
# print("Skewness of target y: ", skew(y.cpu().detach().numpy()))
# print("Skewness of transformed x: ", skew(transformed_x))