import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Parameter
import math
import torch.nn.functional as F
# from lstm import CudnnLstmModel
# from fno import FNO2d, FNO1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class QuantileMappingModel(nn.Module):
    def __init__(self, nx=1, hidden_dim=64, num_layers=2, modelType='ANN', degree=2):
        super(QuantileMappingModel, self).__init__()
        self.degree = degree

        # Automatically calculate ny: degree scales + 1 shift + 1 threshold
        ny = degree + 1 
        self.model_type = modelType 

        if modelType == 'ANN':
            self.transform_generator = self.build_transform_generator(nx, hidden_dim, ny, num_layers)
        elif modelType == 'FNO2d':
            self.transform_generator = FNO2d(16, 16, hidden_dim)
        elif modelType == 'FNO1d':
            self.transform_generator = FNO1d(modes=16, width=hidden_dim, input_dim=nx, output_dim=ny)
        elif modelType == 'LSTM':
            self.lstm = nn.LSTM(input_size=nx, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, ny) 

    def build_transform_generator(self, nx, hidden_dim, ny, num_layers):
        layers = []
        
        # Input layer
        layers.append(nn.Linear(nx, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))

        # Hidden layers
        for _ in range(num_layers - 2):  # -2 because we have input and output layers separately
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        # Output layer
        layers.append(nn.Linear(hidden_dim, ny))

        return nn.Sequential(*layers)

    def forward(self, x, input_tensor, time_scale):
        # Generate transformation parameters
        if self.model_type == "LSTM":
            # LSTM forward pass
            lstm_out, _ = self.lstm(input_tensor)  # Output shape: [batch_size, seq_len, hidden_dim]
            # Generate transformation parameters
            params = self.fc(lstm_out)  # Output shape: [batch_size, seq_len, ny]
        else:
            params = self.transform_generator(input_tensor)

        # Extract scale and shift parameters dynamically based on degree
        if str(time_scale) != 'daily':
            label_dummies = pd.get_dummies(time_scale)
            weights_np = label_dummies.div(label_dummies.sum(axis=0), axis=1).values.astype(np.float32)
            weights = torch.tensor(weights_np, device=params.device)  # shape (time, n_months)
            label_avg = torch.einsum('stp,tm->smp', params, weights)  # (sites, months, params)
            params = torch.einsum('tm,smp->stp', weights, label_avg)  # shape: (sites, time, params)

        scales = [torch.exp(params[:, :, i]) for i in range(self.degree)]  # Ensure positive scaling
        shift = params[:, :, self.degree]
        # threshold = torch.sigmoid(params[:, :, self.degree + 1])  # Between 0 and 1

        # Apply polynomial transformation
        transformed_x = sum((x ** (i + 1)) * scales[i] for i in range(self.degree)) + shift

        # Apply thresholding and activation
        # zero_mask = x <= threshold
        # transformed_x = torch.where(zero_mask, torch.zeros_like(transformed_x), transformed_x)
        transformed_x = torch.relu(transformed_x)

        return transformed_x


class QuantileMappingModel_(nn.Module):
    def __init__(self, nx=1, ny=3, num_series=100, hidden_dim=64):
        super(QuantileMappingModel_, self).__init__()
        self.num_series = num_series

        # Neural network to generate transformation parameters
        # self.transform_generator = nn.Sequential(
        #     nn.Linear(nx, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),  # Add dropout for regularization
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, ny)  # Output: [scale1, scale2, shift, threshold]
        # )

        # Neural network to generate transformation parameters
        # self.transform_generator = nn.Sequential(
        #     nn.Linear(nx, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),  # Add dropout for regularization
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        self.transform_generator = nn.Sequential(
            nn.Linear(nx, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
        )

        # self.lstminv = CudnnLstmModel(nx=nx, ny=ny, hiddenSize=hidden_dim, dr=0.5)

    def forward(self, x, input_tensor):

        params = self.transform_generator(input_tensor)
        # params = self.lstminv(input_tensor)
        scale1 = torch.exp(params[:, :, 0])  # Ensure positive scaling
        # scale1 = F.softplus(params[:, :, 0])  # Ensure positive scaling

        # scale1 = 1 + 0.2*(torch.clamp(params[:, :, 0], min=-1, max=1))  # Ensure positive scaling
        # shift1 = params[:, :, 1]
        # threshold = torch.sigmoid(params[:, :, 1])*0.1
        threshold = torch.sigmoid(params[:, :, 1]) # Between 0 and 1
        # power = torch.sigmoid(params[:, :, 3])* 3   # Range: [0.5, 1]
        power = 1.0

        # scale2 = torch.exp(params[:, :, 3]) # Ensure positive scaling
        # shift2 = params[:, 4].unsqueeze(0)

        # Apply transformation
        # transformed_x = ((x**power)*scale1) + shift1
        transformed_x = ((x**power)*scale1)

        # transformed_x = (scale2 * (x**2)) + (x * scale1) + shift1
        # transformed_x = transformed_x * scale2 + shift2
        # transformed_x = transformed_x * scale3 + shift3

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
        # transformed_x = torch.sum(coeffs.unsqueeze(0) * x_powers, dim=-1)
        transformed_x = torch.where(rainy_mask, transformed_rainy, x)

        # Ensure non-negative values using ReLU instead of clamp
        transformed_x = torch.relu(transformed_x)

        # # Multiply coefficients with x_powers and sum along the last dimension
        #
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