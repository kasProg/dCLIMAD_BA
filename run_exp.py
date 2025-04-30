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
validation = args.validation


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

if train:
    period = train_period
else:
    period = test_period

save_path = f'jobs/{clim}-{ref}/QM_{model_type}_layers{layers}_degree{degree}_quantile{emph_quantile}_scale{time_scale}/{num}/{train_period[0]}_{train_period[1]}/'
model_save_path = save_path
if validation:
    val_save_path =  save_path + f'{test_period[0]}_{test_period[1]}/'

    # test_save_path = val_save_path + f'ep{testepoch}'
    os.makedirs(val_save_path, exist_ok=True)
    
    args_dict = vars(args)  # Converts argparse Namespace into a dictionary
    with open(os.path.join(val_save_path, "test_config.yaml"), "w") as f:
        yaml.dump(args_dict, f)


os.makedirs(save_path, exist_ok=True)
# Save current arguments into config.yaml inside save_path
if train:
    args_dict = vars(args)  # Converts argparse Namespace into a dictionary
    with open(os.path.join(save_path, "train_config.yaml"), "w") as f:
        yaml.dump(args_dict, f)


data_loader = DataLoaderWrapper(
    clim=clim, scenario='historical', ref=ref, period=period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, target_y=target_y, save_path=save_path, stat_save_path = model_save_path,
    crd='all', batch_size=batch_size, train=train, device=device)

dataloader = data_loader.get_dataloader()

if validation:
    data_loader_val = DataLoaderWrapper(
    clim=clim, scenario='historical', ref=ref, period=test_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, target_y=target_y, save_path=val_save_path, stat_save_path = model_save_path,
    crd='all', batch_size=batch_size, train=train, device=device)

    dataloader_val = data_loader_val.get_dataloader()


if not train and trend_analysis:
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
    if validation:
        _, time_x_val = data_loader_val.load_dynamic_inputs()
        time_labels_val = helper.extract_time_labels(time_x_val, label_type=time_scale)
else:
    time_labels = 'daily'
    time_labels_val = 'daily'


model = QuantileMappingModel(nx=nx, degree=degree, hidden_dim=64, num_layers=layers, modelType=model_type).to(device)


# if resume:
#     state_dict = torch.load(f'{save_path}model_{testepoch}.pth', map_location=device, weights_only=True)
#     model.load_state_dict(state_dict)
    
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
        transformed_x = model(batch_x, batch_input_norm, time_scale=time_labels)

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
        
        # ====== 🔥 VALIDATION SECTION 🔥 ======
        if validation:
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                for batch_input_norm, batch_x, batch_y in dataloader_val:
                    batch_input_norm = batch_input_norm.to(device)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    transformed_x = model(batch_x, batch_input_norm, time_labels_val)

                    val_dist_loss = w1 * distributional_loss_interpolated(transformed_x.T, batch_y.T, device=device, num_quantiles=1000, emph_quantile=emph_quantile)
                    val_rainy_loss = w2 * rainy_day_loss(transformed_x.T, batch_y.T)
                    val_loss = val_dist_loss + val_rainy_loss

                    val_epoch_loss += val_loss.item()

            avg_val_loss = val_epoch_loss / len(dataloader_val)

            if logging:
                writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            
                print(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")

