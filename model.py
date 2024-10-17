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
from sklearn.preprocessing import StandardScaler

class QuantileMappingModel(nn.Module):
    def __init__(self, num_series=100, hidden_dim=64):
        super(QuantileMappingModel, self).__init__()
        self.num_series = num_series

        # Neural network to generate transformation parameters
        self.transform_generator = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Output: [scale1, scale2, shift, threshold]
        )

        # self.transform_generator = nn.Sequential(
        #     nn.Linear(1, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 4)  # Output: [scale, shift, threshold]
        # )

    def normalize_elevation(self, elevation):
        mean = torch.mean(elevation)
        std_dev = torch.std(elevation)
        normalized_elevation = (elevation - mean) / std_dev
        return normalized_elevation.unsqueeze(1)

    def forward(self, x, elevation):
        # x shape: (Len(time series), num_series)
        # elevation shape: (num_series,)

        normalized_elevation = self.normalize_elevation(elevation)

        # Generate transformation parameters
        params = self.transform_generator(normalized_elevation)
        scale1 = torch.exp(params[:, 0]).unsqueeze(0)  # Ensure positive scaling
        # scale2 = torch.exp(params[:, 1]).unsqueeze(0)  # Ensure positive scaling
        shift = params[:, 1].unsqueeze(0)
        threshold = torch.sigmoid(params[:, 2]).unsqueeze(0)*0.1 # Between 0 and 1

        # Apply transformation
        # transformed_x = (x * scale1) + ((x**2) * scale2) + shift
        transformed_x = (x * scale1) + shift
        # torch.save(scale1, 'scale1.pt')
        # torch.save(shift, 'shift.pt')

        # Apply threshold-based zero handling
        zero_mask = x <= threshold
        transformed_x = torch.where(zero_mask, torch.zeros_like(transformed_x), transformed_x)

        # Ensure non-negative values using softplus
        transformed_x = torch.relu(transformed_x)

        return transformed_x

class QuantileMappingModel_Poly2(nn.Module):
    def __init__(self, num_series=100, degree=3, hidden_dim=64):
        super(QuantileMappingModel_Poly2, self).__init__()
        self.num_series = num_series
        self.degree = degree

        # Neural network to generate coefficients based on elevation
        self.coeff_generator = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, degree + 1)
        )

        # Scaler for elevation data
        # self.elevation_scaler = elevation_scaler

    def normalize_elevation(self, elevation):
        # Calculate mean and standard deviation directly
        mean = torch.mean(elevation)
        std_dev = torch.std(elevation)

        # Normalize the elevation data
        normalized_elevation = (elevation - mean) / std_dev
        return normalized_elevation.unsqueeze(1)


    def forward(self, x, elevation):
        # x shape: (Len(time series), num_series)
        # elevation shape: (num_series,)

        # Normalize elevation data
        normalized_elevation = self.normalize_elevation(elevation)

        # Generate coefficients for each series based on normalized elevation
        coeffs = self.coeff_generator(normalized_elevation) # Shape: (num_series, degree + 1)

        rainy_mask = x>0

        # Create a tensor of powers of x
        x_powers = x.unsqueeze(-1).pow(torch.arange(self.degree + 1, device=x.device))
        # x_powers shape: (Len(time series), num_series, degree + 1)

        # Multiply coefficients with x_powers and sum along the last dimension, only for rainy days
        transformed_rainy = torch.sum(coeffs.unsqueeze(0) * x_powers, dim=-1)

        # Apply transformation only to rainy days, keep original values for non-rainy days
        transformed_x = torch.where(rainy_mask, transformed_rainy, x)

        # Ensure non-negative values using ReLU instead of clamp
        transformed_x = torch.relu(transformed_x)

        # # Multiply coefficients with x_powers and sum along the last dimension
        # transformed_x = torch.sum(coeffs.unsqueeze(0) * x_powers, dim=-1)
        # # transformed_x shape: (Len(time series), num_series)
        #
        # # Ensure non-negative values
        # transformed_x = torch.clamp(transformed_x, min=0)

        return transformed_x


class QuantileMappingModel_Poly1(nn.Module):
    def __init__(self, num_series=100, degree=3):
        super(QuantileMappingModel_Poly1, self).__init__()
        self.num_series = num_series
        self.degree = degree
        # Initialize trainable coefficients for all time series
        self.coeffs = nn.Parameter(torch.randn(num_series, degree + 1))

    def forward(self, x):
        # x shape: (Len(time series), num_series)
        # Create a tensor of powers of x
        x_powers = x.unsqueeze(-1).pow(torch.arange(self.degree + 1, device=x.device))
        # x_powers shape: (Len(time series), num_series, degree + 1)

        # Multiply coefficients with x_powers and sum along the last dimension
        transformed_x = torch.sum(self.coeffs.unsqueeze(0) * x_powers, dim=-1)
        # transformed_x shape: (Len(time series), num_series)

        # Ensure non-negative values
        transformed_x = torch.clamp(transformed_x, min=0)

        return transformed_x