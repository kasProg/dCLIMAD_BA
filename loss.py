import torch
from scipy.stats import wasserstein_distance
import numpy as np

def interpolate_quantiles(data, device, num_quantiles=200):
    """
    Interpolates the quantiles of a given dataset to match the desired number of quantile levels.
    """
    sorted_data = torch.sort(data, dim=0)[0]
    quantile_levels = torch.linspace(0.0, 1.0, num_quantiles).to(sorted_data.dtype).to(device)  # Evenly spaced quantiles between 0 and 1
    data_quantiles = torch.quantile(sorted_data, quantile_levels, dim=0)  # Interpolated quantiles
    return data_quantiles


def distributional_loss_interpolated(transformed_x, target_y, device, num_quantiles=100):
    """
    Calculates the distributional loss by interpolating the quantiles of both distributions,
    even if the lengths of the two datasets are different.
    """
    # Interpolate the quantiles for both distributions
    quantiles_x = interpolate_quantiles(transformed_x, device, num_quantiles)
    quantiles_y = interpolate_quantiles(target_y, device, num_quantiles)

    # Calculate the mean squared error (MSE) between quantiles
    loss = torch.mean((quantiles_x - quantiles_y) ** 2)
    return loss

def rainy_day_loss(transformed_x, target_y):
    """
    Calculates the loss that targets preserving the number of rainy days in the transformed data.
    """
    # Calculate the number of rainy days in transformed and target datasets
    rainy_days_transformed = (transformed_x == 0).sum(dim=0)
    rainy_days_target = (target_y == 0).sum(dim=0)

    # Calculate the mean absolute difference in the number of rainy days across all series
    rainy_days_loss = torch.mean(torch.abs(rainy_days_transformed - rainy_days_target).to(transformed_x.dtype))
    return rainy_days_loss

def compare_distributions(transformed_x, x, y):
    # Assuming transformed_x, x, and y are 2D tensors of shape (num_time_series, time_steps)
    num_series = transformed_x.shape[1]
    wasserstein_distances = []

    for i in range(num_series):
        # Calculate Wasserstein distance between transformed_x and y
        dist_transformed = wasserstein_distance(transformed_x[:,i], y[:,i])

        # Calculate Wasserstein distance between x and y
        dist_original = wasserstein_distance(x[:, i], y[:,i])

        # Calculate improvement ratio
        improvement_ratio = (dist_original - dist_transformed) / dist_original

        wasserstein_distances.append(improvement_ratio)

    # Average improvement across all series
    avg_improvement = sum(wasserstein_distances) / num_series

    return avg_improvement, wasserstein_distances

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2, axis=0))