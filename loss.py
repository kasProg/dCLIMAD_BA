import torch
from scipy.stats import wasserstein_distance
import numpy as np
import torch.nn.functional as F

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

# def rainy_day_loss(transformed_x, target_y):
#     """
#     Calculates the loss that targets preserving the number of rainy days in the transformed data.
#     """
#     # Calculate the number of rainy days in transformed and target datasets
#     rainy_days_transformed = (transformed_x == 0).sum(dim=0)
#     rainy_days_target = (target_y == 0).sum(dim=0)
#
#     # Calculate the mean absolute difference in the number of rainy days across all series
#     rainy_days_loss = torch.mean(torch.abs(rainy_days_transformed - rainy_days_target).to(transformed_x.dtype))
#     return rainy_days_loss


def rainy_day_loss(transformed_x, target_y, threshold=0.1):
    """
    Calculates a differentiable loss that targets preserving the number of rainy days in the transformed data.
    """
    # Use a sigmoid approximation for the "rainy days" indicator
    sigmoid_transformed_x = torch.sigmoid(-transformed_x / threshold)
    sigmoid_target_y = torch.sigmoid(-target_y / threshold)

    # Sum over the approximated rainy days in both transformed and target data
    rainy_days_transformed = sigmoid_transformed_x.sum(dim=0)
    rainy_days_target = sigmoid_target_y.sum(dim=0)

    # Calculate the mean absolute difference in the number of rainy days across all series
    rainy_days_loss = torch.mean(torch.abs(rainy_days_transformed - rainy_days_target))
    return rainy_days_loss

def compare_distributions(transformed_x, x, y):
    # Assuming transformed_x, x, and y are 2D tensors of shape (time_steps, num_time_series)
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

def kl_divergence_loss(transformed_x, target_y, num_bins=50):
    # Initialize histograms for transformed and target data
    # transformed_hist = torch.stack(
    #     [torch.histc(transformed_x[:, i], bins=num_bins, min=0, max=8) for i in range(transformed_x.shape[1])])
    # target_hist = torch.stack(
    #     [torch.histc(target_y[:, i], bins=num_bins, min=0, max=8) for i in range(target_y.shape[1])])
    #
    # # Normalize to get probability distributions
    # transformed_prob = transformed_hist / (transformed_hist.sum(dim=1, keepdim=True) + 1e-10)
    # target_prob = target_hist / (target_hist.sum(dim=1, keepdim=True) + 1e-10)
    #
    #
    # # Add a small value to avoid log(0)
    # epsilon = 1e-10
    # transformed_prob = transformed_prob + epsilon
    # target_prob = target_prob + epsilon

    prob_log_x = F.log_softmax(transformed_x, dim=1)
    prob_y = F.softmax(target_y, dim=1)

    # Calculate KL divergence for each coordinate and take the mean
    kl_loss = F.kl_div(prob_log_x, prob_y, reduction='batchmean')

    # Return the average KL divergence across all coordinates
    return kl_loss


