import torch
import numpy as np
from scipy import stats
from loss import compare_distributions, rmse

dataset = '/data/kas7897/Livneh/'
# clim_model = '/data/kas7897/GFDL-ESM4/'
clim_model = '/data/kas7897/Livneh/'
noise_type = 'upscale_1by4_LOGnoisy001d_bci'
period = [1991, 1995]
num = 2000

# Load the .pt files
reference = torch.load(f'{dataset}QM_input/y{period}{num}.pt', weights_only=False)
noise = torch.load(f'{clim_model}QM_input/x{period}{num}_{noise_type}.pt', weights_only=False)
# model = torch.load(f'{dataset}/QM_input/m{period}{num}_{noise_type}.pt', weights_only=False)

# Ensure the tensors are on the CPU and convert to numpy arrays
ref_prcp = reference.cpu().numpy()
noise_prcp = noise.cpu().numpy()
# model_prcp = model.cpu().numpy()


def quantile_mapping(obs, model):
    # Remove zero precipitation values
    obs_nonzero = obs[obs > 0]
    model_nonzero = model[model > 0]

    # Calculate empirical CDFs
    obs_cdf = stats.rankdata(obs_nonzero) / len(obs_nonzero)
    model_cdf = stats.rankdata(model_nonzero) / len(model_nonzero)

    # Interpolate to map model values to observed distribution
    corrected = np.interp(model_cdf, obs_cdf, np.sort(obs_nonzero))

    # Preserve zero precipitation values
    corrected_full = np.zeros_like(model)
    corrected_full[model > 0] = corrected

    return corrected_full

# Ensure the datasets have the same shape
if ref_prcp.shape != noise_prcp.shape:
    raise ValueError("Datasets must have the same shape")

# Get the shape of the data
time, num_coordinates = ref_prcp.shape

# Initialize an array to store the corrected data
corrected_prcp = np.zeros_like(noise_prcp)

# Apply quantile mapping for each coordinate
for coord in range(num_coordinates):
    corrected_prcp[:, coord] = quantile_mapping(ref_prcp[:, coord], noise_prcp[:, coord])

# # Apply quantile mapping
# corrected_prcp = np.apply_along_axis(
#     lambda x: quantile_mapping(ref_prcp.flatten(), x),
#     axis=0,
#     arr=noise_prcp
# )

# Reshape the corrected data to match the original shape
# transformed_x = corrected_prcp.reshape(noise_prcp.shape)

# # Convert the corrected numpy array back to a PyTorch tensor
# corrected_tensor = torch.from_numpy(corrected_prcp)

avg_improvement, individual_improvements = compare_distributions(corrected_prcp, noise_prcp, ref_prcp)
print(f"Average distribution improvement: {avg_improvement:.4f}")

print(f"RMSE between Noise and Target: {np.median(rmse(noise_prcp, ref_prcp))}")
print(f"RMSE between Corrected and Target: {np.median(rmse(corrected_prcp, ref_prcp))}")

print(f"RMSE Diff: {np.median(rmse(noise_prcp, ref_prcp)) - np.median(rmse(corrected_prcp, ref_prcp))}")

