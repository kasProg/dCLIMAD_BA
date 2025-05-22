import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from model.model import QuantileMappingModel_, QuantileMappingModel, QuantileMappingModel_Poly2
from model.loss import rainy_day_loss, distributional_loss_interpolated, compare_distributions, rmse, kl_divergence_loss, wasserstein_distance_loss, trend_loss
from ibicus.evaluate.metrics import *
from data.loader import DataLoaderWrapper
from data.helper import generate_run_id
import argparse
import yaml
import data.helper as helper
import datetime
import os
from eval.metrics import *

###-----The code is currently accustomed to CMIP6-Livneh Data format ----###

parser = argparse.ArgumentParser(description='Quantile Mapping Model Configuration')

parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--logging', action='store_true')
parser.add_argument('--clim', type=str, default='miroc6')
parser.add_argument('--ref', type=str, default='livneh')
parser.add_argument('--cmip_dir', type=str)
parser.add_argument('--ref_dir', type=str)
parser.add_argument('--ref_var', type=str)
parser.add_argument('--input_attrs', type=str, help="Semicolon-separated list of input attributes")
parser.add_argument('--train', action='store_true')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--model_type', type=str, default='ANN')
parser.add_argument('--degree', type=int, default=1)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--time_scale', type=str, default='seasonal')
parser.add_argument('--emph_quantile', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--testepoch', type=int, default=50)
parser.add_argument('--spatial_extent', nargs='+', default=None)
parser.add_argument('--spatial_extent_val', nargs='+', default=None)
parser.add_argument('--shapefile_filter_path', type=str, default=None)
parser.add_argument('--spatial_test', action='store_true')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--trend_analysis', action='store_true')
parser.add_argument('--benchmarking', action='store_true')
parser.add_argument('--train_start', type=int, default=1950)
parser.add_argument('--train_end', type=int, default=1980)
parser.add_argument('--val_start', type=int, default=1981)
parser.add_argument('--val_end', type=int, default=1995)
parser.add_argument('--scenario', type=str, default='ssp5_8_5')
parser.add_argument('--trend_start', type=int, default=2075)
parser.add_argument('--trend_end', type=int, default=2099)


args = parser.parse_args()

# Generate unique run ID
args_dict = vars(args)
run_id = generate_run_id(args_dict)

torch.manual_seed(42)
cuda_device = args.cuda_device

logging = args.logging


cmip6_dir = args.cmip_dir
ref_path = args.ref_dir
clim = args.clim
ref = args.ref
ref_var = args.ref_var

train = args.train
validation = args.validation

train_period = [args.train_start, args.train_end]
val_period = [args.val_start, args.val_end]
epochs = args.epochs
testepoch = args.testepoch
benchmarking = args.benchmarking

# model params
model_type = args.model_type
batch_size = args.batch_size
degree = args.degree
layers = args.layers
time_scale = args.time_scale
emph_quantile = args.emph_quantile


## For Spatial Test
spatial_test = args.spatial_test
spatial_extent =  None if not spatial_test  else args.spatial_extent
spatial_extent_val =  None if not spatial_test  else args.spatial_extent_val
shapefile_filter_path =  None if not spatial_test  else args.shapefile_filter_path


## INPUTS
input_attrs = args.input_attrs.split(';')


####------------FIXED INPUTS------------####


input_x = {'precipitation': ['pr', 'prec', 'prcp' 'PRCP', 'precipitation']}
clim_var = 'pr'

## loss params
w1 = 1
w2 = 0
# ny = 4 # number of params



###------------ Developer section here --------------###
if cuda_device == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
    else:
        raise RuntimeError(f"CUDA device {cuda_device} requested but CUDA is not available.")

if logging:
    exp = f'conus/{clim}-{ref}/{model_type}_{layers}Layers_{degree}degree_quantile{emph_quantile}_scale{time_scale}/{run_id}_{train_period[0]}_{train_period[1]}_{val_period[0]}_{val_period[1]}'
    writer = SummaryWriter(f"runs/{exp}")


save_path = f'jobs/{clim}-{ref}/QM_{model_type}_layers{layers}_degree{degree}_quantile{emph_quantile}_scale{time_scale}/{run_id}/{train_period[0]}_{train_period[1]}/'
model_save_path = save_path
if validation:
    val_save_path =  save_path + f'{val_period[0]}_{val_period[1]}/'
    # test_save_path = val_save_path + f'ep{testepoch}'
    os.makedirs(val_save_path, exist_ok=True)

os.makedirs(save_path, exist_ok=True)

# Save current arguments into config.yaml inside save_path
with open(os.path.join(save_path, "train_config.yaml"), "w") as f:
    yaml.dump(args_dict, f)

data_loader = DataLoaderWrapper(
    clim=clim, scenario='historical', ref=ref, period=train_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, ref_var=ref_var, save_path=save_path, stat_save_path = model_save_path,
    crd=spatial_extent, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train, device=device)

dataloader = data_loader.get_dataloader()

if validation:
    data_loader_val = DataLoaderWrapper(
    clim=clim, scenario='historical', ref=ref, period=val_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, ref_var=ref_var, save_path=val_save_path, stat_save_path = model_save_path,
    crd=spatial_extent_val, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train, device=device)

    dataloader_val = data_loader_val.get_dataloader()


if time_scale == 'daily':
    time_labels = time_labels_val = 'daily'
else:
    time_labels = helper.extract_time_labels(data_loader.load_dynamic_inputs()[1], label_type=time_scale)
    time_labels_val = helper.extract_time_labels(data_loader_val.load_dynamic_inputs()[1], label_type=time_scale) if validation else None


nx = len(input_x)+ len(input_attrs)
model = QuantileMappingModel(nx=nx, degree=degree, hidden_dim=64, num_layers=layers, modelType=model_type).to(device)

    
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
        
        # ====== VALIDATION SECTION ====== #
        if validation:
            model.eval()
            val_epoch_loss = 0
            xt_val = []
            x_val = []
            y_val = []
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
                    # Store predictions
                    xt_val.append(transformed_x.cpu())

                    y_val.append(batch_y.cpu())
                    x_val.append(batch_x.cpu())

            avg_val_loss = val_epoch_loss / len(dataloader_val)

            xt_val = torch.cat(xt_val, dim=0).numpy().T
            x_val = torch.cat(x_val, dim=0).numpy().T
            y_val = torch.cat(y_val, dim=0).numpy().T
            x_val_time = torch.load(f'{val_save_path}/time.pt', weights_only = False)

            ## to manage time
            x_val_time_np = np.array([pd.Timestamp(str(t)) for t in x_val_time])
            x_val_time_np = np.array([pd.Timestamp(t).replace(hour=0, minute=0, second=0) for t in x_val_time_np], dtype='datetime64[D]')
            # Generate a daily time array following the standard Gregorian calendar
            y_val_time = pd.date_range(start=f"{val_period[0]}-01-01", end=f"{val_period[1]}-12-31", freq="D")
            # Convert to NumPy array for indexing and comparison
            y_val_time_np = y_val_time.to_numpy()
            # Find indices where observed time matches model time
            matched_indices = np.where(np.isin(y_val_time_np, x_val_time_np))[0]
            y_val = y_val[matched_indices,:]

            # Initialize climate indices manager
            climate_indices = ClimateIndices()          

            # day_bias_percentages = get_day_bias_percentages(x_val, y_val, xt_val, climate_indices)
            mean_bias_percentages = get_mean_bias_percentages(x_val, y_val, xt_val, x_val_time_np, climate_indices)
            day_bias_percentages = get_day_bias_percentages(x_val, y_val, xt_val, climate_indices)

            keys_mean = ['SDII (Monthly)','CDD (Yearly)', 'CWD (Yearly)', "Rx1day", "Rx5day", "R10mm",  "R20mm", "R95pTOT", "R99pTOT"]
            mean_bias_percentages = dict(filter(lambda item: item[0] in keys_mean , mean_bias_percentages.items()))


            if logging:
                writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            
                print(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")

                # Extract and log median(corrected) per metric
                for name, values in mean_bias_percentages.items():
                    corrected = values[1]  # extract corrected values
                    median_corrected = float(np.median(corrected))
                    writer.add_scalar(f'median_adjusted/{name}', median_corrected, epoch)

                # Extract and log median(corrected) per metric
                for name, values in day_bias_percentages.items():
                    corrected = values[1]  # extract corrected values
                    median_corrected = float(np.median(corrected))
                    writer.add_scalar(f'median_adjusted/{name}', median_corrected, epoch)
                
                

# Save finished.txt to mark successful completion
finished_file = os.path.join(model_save_path, "finished.txt")
with open(finished_file, "w") as f:
    f.write(f"Finished successfully at {datetime.datetime.now()}\n")