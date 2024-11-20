 #%%
import torch
import numpy as np
from scipy import stats
from loss import compare_distributions, rmse, distributional_loss_interpolated
from scipy.stats import norm, gamma, erlang, expon, percentileofscore
import os
import matplotlib.pyplot as plt


dataset = '/data/kas7897/Livneh/'
clim_model = '/data/kas7897/GFDL-ESM4/'
# clim_model = '/data/kas7897/Livneh/'
noise_type = 'livneh_bci'
hist_period = [1980, 1990]
future_period = [1991, 1995]

num = 2000
QM = True
save_path = f'benchmark/QM_replace/'
os.makedirs(save_path, exist_ok=True)

# Load the .pt files
reference = torch.load(f'{dataset}QM_input/y{hist_period}{num}.pt', weights_only=False)
target_reference = torch.load(f'{dataset}QM_input/y{future_period}{num}.pt', weights_only=False)

hist_model = torch.load(f'{clim_model}QM_input/x{hist_period}{num}_{noise_type}.pt', weights_only=False)
future_model = torch.load(f'{clim_model}QM_input/x{future_period}{num}_{noise_type}.pt', weights_only=False)
# model = torch.load(f'{dataset}/QM_input/m{period}{num}_{noise_type}.pt', weights_only=False)

# Ensure the tensors are on the CPU and convert to numpy arrays
ref_prcp = reference.cpu().numpy()
future_ref = target_reference.cpu().numpy()
hist_prcp = hist_model.cpu().numpy()
future_prcp = future_model.cpu().numpy()
# model_prcp = model.cpu().numpy()

def eQM_replace(ref_dataset, model_present, model_future):
        """
        For each model_future value, get its percentile on the CDF of model_present,
        then ust it to get a value from the model_present.
        returns: downscaled model_present and model_future        
        """
        model_present_corrected = np.zeros(model_present.size)  
        model_future_corrected = np.zeros(model_future.size)

        for ival, model_value in enumerate(model_present):
            percentile = percentileofscore(model_present, model_value)
            model_present_corrected[ival] = np.percentile(ref_dataset, percentile)

        for ival, model_value in enumerate(model_future):
            percentile = percentileofscore(model_present, model_value)
            model_future_corrected[ival] = np.percentile(ref_dataset, percentile)
            
        return model_present_corrected, model_future_corrected

def eQM_delta(ref_dataset, model_present, model_future):
        """
        Remove the biases for each quantile value taking the difference between 
        ref_dataset and model_present at each percentile as a kind of systematic bias (delta)
        and add them to model_future at the same percentile.

        returns: downscaled model_present and model_future        
        """

        model_present_corrected = np.zeros(model_present.size)  
        model_future_corrected = np.zeros(model_future.size)

        for ival, model_value in enumerate(model_present):
            percentile = percentileofscore(model_present, model_value)
            model_present_corrected[ival] = np.percentile(ref_dataset, percentile)

        for ival, model_value in enumerate(model_future):
            percentile = percentileofscore(model_future, model_value)
            model_future_corrected[ival] = model_value + np.percentile(
                ref_dataset, percentile) - np.percentile(model_present, percentile)

        return model_present_corrected, model_future_corrected


if QM:
    # Get the shape of the data
    time, num_coordinates = ref_prcp.shape

    # Initialize an array to store the corrected data
    corrected_hist_prcp = np.zeros_like(hist_prcp)
    corrected_future_prcp = np.zeros_like(future_prcp)


    # Apply quantile mapping for each coordinate
    for coord in range(num_coordinates):
        # corrected_hist_prcp[:, coord], corrected_future_prcp[:, coord] = eQM_delta(ref_prcp[:, coord], hist_prcp[:, coord], future_prcp[:, coord])
        corrected_hist_prcp[:, coord], corrected_future_prcp[:, coord] = eQM_replace(ref_prcp[:, coord], hist_prcp[:, coord], future_prcp[:, coord])

    
    torch.save(corrected_hist_prcp, f'{save_path}/hist{hist_period}{num}.pt')
    torch.save(corrected_future_prcp, f'{save_path}/future{future_period}{num}.pt')

else:
     
    corrected_future_prcp = torch.load(f'{save_path}/future{future_period}{num}.pt', weights_only=False)
    corrected_hist_prcp = torch.load(f'{save_path}/hist{hist_period}{num}.pt', weights_only=False)






print('Historical Metric:\n')
avg_improvement, individual_improvements = compare_distributions(corrected_hist_prcp, hist_prcp, ref_prcp)
print(f"Average distribution improvement: {avg_improvement:.4f}")

print(f"RMSE between Noise and Target: {np.median(rmse(hist_prcp, ref_prcp))}")
print(f"RMSE between Corrected and Target: {np.median(rmse(corrected_hist_prcp, ref_prcp))}")

print(f"RMSE Diff: {np.median(rmse(hist_prcp, ref_prcp)) - np.median(rmse(corrected_hist_prcp, ref_prcp))}")


quantile_rmse_model = torch.sqrt(distributional_loss_interpolated(torch.tensor(future_prcp), torch.tensor(future_ref), device='cpu', num_quantiles=100))
quantile_rmse_bs = torch.sqrt(distributional_loss_interpolated(torch.tensor(corrected_future_prcp), torch.tensor(future_ref), device='cpu', num_quantiles=100))

print('Future Metric:\n')
avg_improvement, individual_improvements = compare_distributions(corrected_future_prcp, future_prcp, future_ref)
print(f"Average distribution improvement: {avg_improvement:.4f}")

print(f"Quantile RMSE between Model and Target: {quantile_rmse_model}")
print(f"Quantile RMSE between Corrected and Target: {quantile_rmse_bs}")

print(f"RMSE Diff: {quantile_rmse_model - quantile_rmse_bs}")


best_ind, best_improv = max(enumerate(individual_improvements), key=lambda x: x[1])

# Step 6: Plotting the original and transformed distributions
plt.figure(figsize=(12, 6))
plt.suptitle(f'Quantile Mapping (Delta) \n WS Distance Improvement:{best_improv}')
plt.subplot(1, 2, 1)
plt.hist(future_prcp[:, best_ind], bins=30, alpha=0.6, label="Model x")
plt.hist(future_ref[:, best_ind], bins=30, alpha=0.6, label="Target Y", color='orange')
plt.title("Modeled x and Target Distributions")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(corrected_future_prcp[:, best_ind], bins=30, alpha=0.6, label="Transformed x", color='green')
plt.hist(future_ref[:, best_ind], bins=30, alpha=0.6, label="Target y", color='orange')
plt.title("Transformed x vs Target y")
plt.legend()

plt.savefig('fig.png')