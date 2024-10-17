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
from model import QuantileMappingModel, QuantileMappingModel_Poly2
from loss import rainy_day_loss, distributional_loss_interpolated, compare_distributions, rmse
from data import process_data
from sklearn.preprocessing import StandardScaler

###-----The code is currently accustomed to Livneh Data format ----###

torch.manual_seed(42)
device = torch.device('cuda:5')

dataset = '/data/kas7897/Livneh/'

##if running synthetic case
noise_type = 'upscale_1by4_Wnoisy1_bci'
# noise_type = 'bci_Wnoisy001d'
train_period = [1980, 1990]
test_period = [1980, 1990]
train = 0#training = 1; else test

model_type = 'SST' #[SST/model, Poly2]
degree = 3 # only if model_type = Poly

##number of coordinates; if all then set to 'all'
num = 1000

#extracting valid lat-lon pairs with non-nan prcp
ds_sample = xr.open_dataset(f"/data/kas7897/Livneh/prec.1980.nc")
valid_coords = valid_crd.valid_lat_lon(ds_sample)

#processing elevation data
elev = xr.open_dataset('/data/kas7897/diffDownscale/elev_Livneh.nc')
elev_data = elev['elevation'].sel(lat=xr.DataArray(valid_coords[:num, 0], dims='points'),
                             lon=xr.DataArray(valid_coords[:num, 1], dims='points'),
                             method='nearest').values

pathx = dataset + noise_type

if train==1:
    period = train_period
else:
    period = test_period


if os.path.exists(f'{dataset}QM_input/x{period}{num}_{noise_type}.pt'):
    print('loading x...')
    x = torch.load(f'{dataset}/QM_input/x{period}{num}_{noise_type}.pt', weights_only=False)
else:
    print("processing x data...")
    x = process_data(pathx, period, valid_coords, num, device)
    torch.save(x, f'{dataset}QM_input/x{period}{num}_{noise_type}.pt')

if os.path.exists(f'{dataset}QM_input/y{period}{num}.pt'):
    print('loading y...')
    y = torch.load(f'{dataset}QM_input/y{period}{num}.pt', weights_only=False)
else:
    print("processing y data...")
    y = process_data(dataset, period, valid_coords, num, device)
    torch.save(y, f'{dataset}QM_input/y{period}{num}.pt')


elev_tensor = torch.tensor(elev_data).to(x.dtype).to(device)
torch.set_default_dtype(x.dtype)

num_series = x.shape[1]

## Choose model
# model = QuantileMappingModel_Poly2(num_series=num_series, degree=3, hidden_dim=64).to(device)
if model_type == 'SST':
    model = QuantileMappingModel(num_series=num_series, hidden_dim=64).to(device)
elif model_type == 'Poly2':
    model = QuantileMappingModel_Poly2(num_series=num_series, degree=3, hidden_dim=64).to(device)

if train == 1:
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    balance_loss = 0.01  # Adjust this weight to balance between distributional and rainy day losses

    # Training loop
    num_epochs = 2000
    for epoch in range(num_epochs):
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

    torch.save(model.state_dict(), f'models/QM_{model_type}_{num}{train_period}_{noise_type}.pth')

else:
    model.load_state_dict(torch.load(f'models/QM_{model_type}_{num}{train_period}_{noise_type}.pth', weights_only=True))
    model.eval()


transformed_x = model(x, elev_tensor).cpu().detach().numpy()

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

# Print skewness to verify positive skew
# print("Skewness of original x: ", skew(x.cpu().detach().numpy()))
# print("Skewness of target y: ", skew(y.cpu().detach().numpy()))
# print("Skewness of transformed x: ", skew(transformed_x))