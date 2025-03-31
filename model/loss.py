import torch
from scipy.stats import wasserstein_distance
import numpy as np
import torch.nn.functional as F

def interpolate_quantiles(data, device, quantile_levels, num_quantiles=200):
    """
    Interpolates the quantiles of a given dataset to match the desired number of quantile levels.
    """
    sorted_data = torch.sort(data, dim=0)[0]
    data_quantiles = torch.quantile(sorted_data, quantile_levels, dim=0)  # Interpolated quantiles
    return data_quantiles


def distributional_loss_interpolated(transformed_x, target_y, device, num_quantiles=100, emph_quantile=0.5):
    """
    Calculates the distributional loss by interpolating the quantiles of both distributions,
    even if the lengths of the two datasets are different.
    """    
    quantile_levels = torch.linspace(0.0, 1.0, num_quantiles).to(target_y.dtype).to(device)

    # Interpolate the quantiles for both distributions
    quantiles_x = interpolate_quantiles(transformed_x, device, quantile_levels, num_quantiles)
    quantiles_y = interpolate_quantiles(target_y, device, quantile_levels, num_quantiles)
    
    if isinstance(emph_quantile, (float, int)):
        # Single quantile emphasis
        weights = torch.exp(-torch.abs(quantile_levels - emph_quantile)).unsqueeze(-1)
    elif isinstance(emph_quantile, (list, tuple)) and all(isinstance(q, (float, int)) for q in emph_quantile):
        # Multiple quantiles emphasis
        weights = sum(torch.exp(-torch.abs(quantile_levels - q)) for q in emph_quantile).unsqueeze(-1)
    else:
        # No emphasis (uniform weighting)
        weights = 1


    # Calculate the mean squared error (MSE) between quantiles
    loss = torch.mean(weights*(quantiles_x - quantiles_y) ** 2)
    return loss

def trend_loss(transformed_x, original_x, device):
    time_index = torch.arange(original_x.size(0), dtype=original_x.dtype).unsqueeze(1).to(device)
    original_trend = torch.linalg.lstsq(time_index, original_x).solution.squeeze()
    transformed_trend = torch.linalg.lstsq(time_index, transformed_x).solution.squeeze()

    trend_loss = torch.mean((transformed_trend - original_trend) ** 2)

    return trend_loss


def rainy_day_loss(transformed_x, target_y, threshold=1):
    """
    Calculates a differentiable loss that targets preserving the number of rainy days in the transformed data.
    """
    # # Use a sigmoid approximation for the "rainy days" indicator
    rainy_indicator_transformed = torch.sigmoid(transformed_x - threshold)
    rainy_indicator_target = torch.sigmoid(target_y - threshold)

    # Binary "rainy day" approximation using ReLU
    # rainy_indicator_transformed = torch.relu((transformed_x - threshold) / threshold)
    # rainy_indicator_target = torch.relu((target_y - threshold) / threshold)

    # Sum over the approximated rainy days in both transformed and target data
    rainy_days_transformed = rainy_indicator_transformed.sum(dim=0)
    rainy_days_target = rainy_indicator_target.sum(dim=0)

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

    prob_log_x = F.log_softmax(transformed_x, dim=1)
    prob_y = F.softmax(target_y, dim=1)

    # Calculate KL divergence for each coordinate and take the mean
    kl_loss = F.kl_div(prob_log_x, prob_y, reduction='batchmean')

    # Return the average KL divergence across all coordinates
    return kl_loss



def wasserstein_distance_loss(predicted, target, dim=1):
    """
    Compute the Wasserstein distance loss between two distributions.

    Args:
        predicted (torch.Tensor): Predicted distribution (logits or probabilities).
        target (torch.Tensor): Target distribution (logits or probabilities).
        dim (int): Dimension along which to compute the distributions.
        
    Returns:
        torch.Tensor: Wasserstein distance loss.
    """
    # Normalize logits to probabilities using softmax if needed
    prob_predicted = torch.softmax(predicted, dim=dim)
    prob_target = torch.softmax(target, dim=dim)

    # Compute the CDFs
    cdf_predicted = torch.cumsum(prob_predicted, dim=dim)
    cdf_target = torch.cumsum(prob_target, dim=dim)

    # Wasserstein distance is the sum of absolute differences between CDFs
    wasserstein_loss = torch.sum(torch.abs(cdf_predicted - cdf_target), dim=dim)
    
    # Average over the batch
    return wasserstein_loss.mean()
