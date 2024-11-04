import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import skew
import xarray as xr
import os
import pandas as pd
import numpy as np
import pickle
import valid_crd
import tqdm
from model import QuantileMappingModel_, QuantileMappingModel_Poly2
from loss import rainy_day_loss, distributional_loss_interpolated, compare_distributions, rmse, kl_divergence_loss
from data import process_data, getStatDic
from torch.utils.data import DataLoader, TensorDataset
import data
from sklearn.preprocessing import StandardScaler

###-----The code is currently accustomed to Livneh Data format ----###

torch.manual_seed(42)
device = torch.device('cuda:6')

dataset = '/data/kas7897/Livneh/'
clim_model = '/data/kas7897/Livneh/'

clim = 'livneh'
ref = 'livneh'

##if running synthetic case
noise_type = 'R5noisy01d'
# noise_type = 'livneh_bci'
# noise_type = 'bci_Wnoisy001d'
train_period = [1980, 1990]
# test_period = [1980, 1990]
test_period = [1991, 1995]
train = 0 # training = 1; else test
# seriesLst = ['noisy_prcp', 'wind']
seriesLst = ['noisy_prcp']
attrLst = ['elev']
epochs = 300

model_type = 'SST' #[SST/model, Poly2]
degree = 1 # only if model_type = Poly

##number of coordinates; if all then set to 'all'
num = 2000
batch_size = 100

save_path = f'models/{clim}-{ref}/QM_{model_type}/{num}/{train_period[0]}_{train_period[1]}/{noise_type}/'
os.makedirs(save_path, exist_ok=True)

#extracting valid lat-lon pairs with non-nan prcp
ds_sample = xr.open_dataset(f"{dataset}prec.1980.nc")
valid_coords = valid_crd.valid_lat_lon(ds_sample)


#processing elevation data
elev = xr.open_dataset('/data/kas7897/diffDownscale/elev_Livneh.nc')
if num == 'all':
    elev_data = elev['elevation'].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'),
                                      lon=xr.DataArray(valid_coords[:, 1], dims='points'),
                                      method='nearest').values
else:
    elev_data = elev['elevation'].sel(lat=xr.DataArray(valid_coords[:num, 0], dims='points'),
                                 lon=xr.DataArray(valid_coords[:num, 1], dims='points'),
                                 method='nearest').values

pathx = clim_model + noise_type

if train==1:
    period = train_period
else:
    period = test_period


if os.path.exists(f'{clim_model}QM_input/x{period}{num}_{noise_type}.pt'):
    print('loading x...')
    x = torch.load(f'{clim_model}QM_input/x{period}{num}_{noise_type}.pt', weights_only=False).to(device)
else:
    print("processing x data...")
    x = process_data(pathx, period, valid_coords, num, device, var='prec')
    torch.save(x, f'{clim_model}QM_input/x{period}{num}_{noise_type}.pt')

if os.path.exists(f'{dataset}QM_input/y{period}{num}.pt'):
    print('loading y...')
    y = torch.load(f'{dataset}QM_input/y{period}{num}.pt', weights_only=False).to(device)
else:
    print("processing y data...")
    y = process_data(dataset, period, valid_coords, num, device, var='prec')
    torch.save(y, f'{dataset}QM_input/y{period}{num}.pt')


x_in = x.unsqueeze(-1)

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

    wind_tensor = wind.to(x.dtype).to(device)
    wind_tensor = wind_tensor.unsqueeze(-1)
    x_in = torch.cat((x_in, wind_tensor), dim=2)

elev_tensor = torch.tensor(elev_data).to(x.dtype).to(device)
attr_tensor = elev_tensor.unsqueeze(-1)

# for inp in inputs:
# elev_in_tensor = torch.broadcast_to(elev_tensor, x.shape).unsqueeze(-1)

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
series_norm[torch.isnan(series_norm)] = 0.0

attr_norm_tensor = attr_norm.unsqueeze(0).expand(series_norm.shape[0], -1, -1)
input_norm_tensor = torch.cat((series_norm, attr_norm_tensor), dim=2)


torch.set_default_dtype(x.dtype)

#for batches
x = x.T
y = y.T
input_norm_tensor = input_norm_tensor.permute(1, 0, 2)
dataset = TensorDataset(input_norm_tensor, x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


num_series = x.shape[1]
nx = x_in.shape[-1] + attr_tensor.shape[-1]
ny = 3 # 3-params

## Choose model
if model_type == 'SST':
    model = QuantileMappingModel_(nx=nx, ny=ny, num_series=num_series, hidden_dim=64).to(device)
elif model_type == 'Poly2':
    model = QuantileMappingModel_Poly2(num_series=num_series, degree=degree, hidden_dim=64).to(device)

if train == 1:
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    balance_loss = 0.01  # Adjust this weight to balance between distributional and rainy day losses

    # Training loop
    num_epochs = epochs
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_input_norm, batch_x, batch_y in dataloader:
            # Move batch to device
            batch_input_norm = batch_input_norm.to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            transformed_x = model(batch_x, batch_input_norm)

            # Compute the loss
            dist_loss = distributional_loss_interpolated(transformed_x.T, batch_y.T, device=device, num_quantiles=100)
            kl_loss = kl_divergence_loss(transformed_x.T, batch_y.T, num_bins=100)
            rainy_loss = rainy_day_loss(transformed_x.T, batch_y.T)

            loss = dist_loss + kl_loss + balance_loss * rainy_loss
            # loss = dist_loss + balance_loss * rainy_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_list.append(avg_epoch_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}')


    torch.save(model.state_dict(), f'{save_path}/model.pth')

    plt.plot(loss_list)
    plt.title('Loss Curve')
    plt.show()

else:
    model.load_state_dict(torch.load(f'{save_path}/model.pth', weights_only=True))
    model.eval()
    transformed_x = []
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


    # # transformed_x = model(x, elev_tensor).cpu().detach().numpy()
    # transformed_x = model(x, input_norm_tensor).cpu().detach().numpy()
    # x = x.cpu().detach().numpy()
    # y = y.cpu().detach().numpy()
    transformed_x = torch.cat(transformed_x, dim=0).numpy().T
    x = torch.cat(x, dim=0).numpy().T
    y = torch.cat(y, dim=0).numpy().T
    avg_improvement, individual_improvements = compare_distributions(transformed_x, x, y)
    print(f"Average distribution improvement: {avg_improvement:.4f}")

    print(f"RMSE between Noise and Target: {np.median(rmse(x, y))}")
    print(f"RMSE between Corrected and Target: {np.median(rmse(transformed_x, y))}")

    best_ind, best_improv = max(enumerate(individual_improvements), key=lambda x: x[1])

    # Step 6: Plotting the original and transformed distributions
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'WS Distance Improvement:{best_improv}')
    plt.subplot(1, 2, 1)
    plt.hist(x[:, best_ind], bins=30, alpha=0.6, label="Noisy")
    plt.hist(y[:, best_ind], bins=30, alpha=0.6, label="Target Y", color='orange')
    plt.title("Noisy x and Target Distributions")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(transformed_x[:, best_ind], bins=30, alpha=0.6, label="Transformed x", color='green')
    plt.hist(y[:, best_ind], bins=30, alpha=0.6, label="Target y", color='orange')
    plt.title("Transformed x vs Target y")
    plt.legend()

    plt.show()


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