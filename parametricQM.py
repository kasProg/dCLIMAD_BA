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
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import wasserstein_distance
from model import QuantileMappingModel, QuantileMappingModel_Poly2
from loss import rainy_day_loss, distributional_loss_interpolated
from sklearn.preprocessing import StandardScaler

train = 0

# Step 1: Generate a set of Gaussian random numbers (x)
torch.manual_seed(42)
device = torch.device('cuda:5')

#extracting valid lat-lon pairs with non-nan prcp
ds_sample = xr.open_dataset(f"/data/kas7897/Livneh/prec.1980.nc")
valid_coords = valid_crd.valid_lat_lon(ds_sample)

elev = xr.open_dataset('/data/kas7897/diffDownscale/elev_Livneh.nc')
test_period = [1991, 1995]
num = 1000


def load_and_process_year(year, valid_coords):
    x_year = xr.open_dataset(f'/data/kas7897/Livneh/upscale_1by4_bci_noisy01d/prec_{year}.nc')
    y_year = xr.open_dataset(f'/data/kas7897/Livneh/prec.{year}.nc')

    x_data = x_year['prec'].sel(lat=xr.DataArray(valid_coords[:num, 0], dims='points'),
                                lon=xr.DataArray(valid_coords[:num, 1], dims='points'),
                                method='nearest').values
    y_data = y_year['prec'].sel(lat=xr.DataArray(valid_coords[:num, 0], dims='points'),
                                lon=xr.DataArray(valid_coords[:num, 1], dims='points'),
                                method='nearest').values


    return x_data, y_data
    # return x_data

elev_data = elev['elevation'].sel(lat=xr.DataArray(valid_coords[:num, 0], dims='points'),
                             lon=xr.DataArray(valid_coords[:num, 1], dims='points'),
                             method='nearest').values

# elevation_scaler = StandardScaler()
# elevation_scaler.fit(elev_data.reshape(-1, 1))

def process_data(train_period, valid_coords, device):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_and_process_year,
                                    range(train_period[0], train_period[1]),
                                    [valid_coords] * (train_period[1] - train_period[0])))

    x_list, y_list = zip(*results)
    # x_list = results

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)


    return torch.tensor(x).to(device), torch.tensor(y).to(device)
    # return torch.tensor(x).to(device)

train_period = [1980, 1990]

# x, y = process_data(test_period, valid_coords, device)
# x = process_data(train_period, valid_coords, device)
# torch.save(x,f'/data/kas7897/Livneh/QM_input/x{test_period}{num}_bci_noisy01d.pt')
# torch.save(y,f'/data/kas7897/Livneh/QM_input/y{test_period}{num}.pt')
x = torch.load(f'/data/kas7897/Livneh/QM_input/x{num}_bci_noisy01d.pt', weights_only=False)
y = torch.load(f'/data/kas7897/Livneh/QM_input/y{num}.pt', weights_only=False)
elev_tensor = torch.tensor(elev_data).to(x.dtype).to(device)
torch.set_default_dtype(x.dtype)

num_series = x.shape[1]

if train == 1:
    # # Step 5: Instantiate the model, define the optimizer and the loss function
    degree = 3
    # model = QuantileMappingModel_Poly2(num_series=num_series, degree=3, hidden_dim=64).to(device)
    model = QuantileMappingModel(num_series=num_series, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    balance_loss = 0.1  # Adjust this weight to balance between distributional and rainy day losses

    # Training loop
    num_epochs = 1000
    # progress_bar = tqdm(range(num_epochs), desc='Epochs')
    for epoch in range(num_epochs):
        # for epoch in range(1):
        model.train()

        # Forward pass
        transformed_x = model(x, elev_tensor)

        # Compute the loss
        dist_loss = distributional_loss_interpolated(transformed_x, y, device=device, num_quantiles=100)
        rainy_loss = rainy_day_loss(transformed_x, y)

        loss = dist_loss + balance_loss*rainy_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'QM_model_weights.pth')

else:
    model = QuantileMappingModel(num_series=num_series, hidden_dim=64).to(device)
    model.load_state_dict(torch.load('QM_model_weights.pth', weights_only=True))
    model.eval()

# Print skewness to verify positive skew
# print("Skewness of original x: ", skew(x.cpu().detach().numpy()))
# print("Skewness of target y: ", skew(y.cpu().detach().numpy()))
# print("Skewness of transformed x: ", skew(transformed_x))

transformed_x = model(x, elev_tensor).cpu().detach().numpy()

def compare_distributions(transformed_x, x, y):
    # Assuming transformed_x, x, and y are 2D tensors of shape (num_time_series, time_steps)
    num_series = transformed_x.shape[1]
    wasserstein_distances = []

    for i in range(num_series):
        # Calculate Wasserstein distance between transformed_x and y
        dist_transformed = wasserstein_distance(transformed_x[:,i], y[:,i].cpu().numpy())

        # Calculate Wasserstein distance between x and y
        dist_original = wasserstein_distance(x[:, i].cpu().numpy(), y[:,i].cpu().numpy())

        # Calculate improvement ratio
        improvement_ratio = (dist_original - dist_transformed) / dist_original

        wasserstein_distances.append(improvement_ratio)

    # Average improvement across all series
    avg_improvement = sum(wasserstein_distances) / num_series

    return avg_improvement, wasserstein_distances

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2, axis=0))

# Usage in your training loop
avg_improvement, individual_improvements = compare_distributions(transformed_x, x, y)
print(f"Average distribution improvement: {avg_improvement:.4f}")

print(f"RMSE between Noise and Target: {np.median(rmse(x.cpu().detach().numpy(), y.cpu().detach().numpy()))}")
print(f"RMSE between Corrected and Target: {np.median(rmse(transformed_x, y.cpu().detach().numpy()))}")

best_improv = max(enumerate(individual_improvements), key=lambda x: x[1])[0]

# Step 6: Plotting the original and transformed distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(x[:,300].cpu().detach().numpy(), bins=30, alpha=0.6, label="Noisy")
plt.hist(y[:,300].cpu().detach().numpy(), bins=30, alpha=0.6, label="Target Y", color='orange')
plt.title("Noisy x and Target Distributions")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(transformed_x[:, 300], bins=30, alpha=0.6, label="Transformed x", color='green')
plt.hist(y[:, 300].cpu().detach().numpy(), bins=30, alpha=0.6, label="Target y", color='orange')
plt.title("Transformed x vs Target y")
plt.legend()

plt.show()