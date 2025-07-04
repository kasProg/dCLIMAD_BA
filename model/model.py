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

        if modelType == 'MLP':
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
            weights = torch.tensor(weights_np, device=params.device)  # shape (time, scale)
            label_avg = torch.einsum('stp,tm->smp', params, weights)  # (sites, scale, params)
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


class QuantileMappingModel1(nn.Module):
    def __init__(self, nx=1, hidden_dim=64, num_layers=2, modelType='ANN', max_degree=5):
        super(QuantileMappingModel1, self).__init__()
        self.max_degree = max_degree
        self.model_type = modelType

        ny = max_degree + 1  # D_max scale params + 1 shift

        if modelType == 'MLP':
            self.transform_generator = self.build_transform_generator(nx, hidden_dim, ny, num_layers)
        elif modelType == 'FNO2d':
            self.transform_generator = FNO2d(16, 16, hidden_dim)
        elif modelType == 'FNO1d':
            self.transform_generator = FNO1d(modes=16, width=hidden_dim, input_dim=nx, output_dim=ny)
        elif modelType == 'LSTM':
            self.lstm = nn.LSTM(input_size=nx, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, ny)

    def build_transform_generator(self, nx, hidden_dim, ny, num_layers):
        layers = [
            nn.Linear(nx, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        layers.append(nn.Linear(hidden_dim, ny))
        return nn.Sequential(*layers)

    def forward(self, x, input_tensor, time_scale):
        if self.model_type == "LSTM":
            lstm_out, _ = self.lstm(input_tensor)
            params = self.fc(lstm_out)
        else:
            params = self.transform_generator(input_tensor)

        if str(time_scale) != 'daily':
            label_dummies = pd.get_dummies(time_scale)
            weights_np = label_dummies.div(label_dummies.sum(axis=0), axis=1).values.astype(np.float32)
            weights = torch.tensor(weights_np, device=params.device)  # shape (time, scale)
            label_avg = torch.einsum('stp,tm->smp', params, weights)  # (sites, scale, params)
            params = torch.einsum('tm,smp->stp', weights, label_avg)  # shape: (sites, time, params)

        poly_weights = params[:, :, :self.max_degree]  # shape: [sites, time, D_max]
        shift = params[:, :, -1]

        # Apply learned polynomial transformation
        powers = [x ** (i + 1) for i in range(self.max_degree)]
        transformed_x = sum(w * p for w, p in zip(torch.unbind(poly_weights, dim=-1), powers)) + shift
        transformed_x = torch.relu(transformed_x)

        self.latest_poly_weights = poly_weights  # Store for regularization
        
        return transformed_x

    def get_weighted_l1_penalty(self, lambda_l1=1e-4):
        """
        Returns the weighted L1 regularization loss for polynomial weights.
        """
        if not hasattr(self, 'latest_poly_weights'):
            raise RuntimeError("Run a forward pass before calling L1 penalty.")

        degree_weights = torch.arange(
            1, self.max_degree + 1, dtype=self.latest_poly_weights.dtype, device=self.latest_poly_weights.device
        )
        weighted_abs = torch.abs(self.latest_poly_weights) * degree_weights
        return lambda_l1 * torch.sum(weighted_abs)
