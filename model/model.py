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
import torch.fft as fft

class QuantileMappingModel(nn.Module):
    def __init__(self, nx=1, hidden_dim=64, num_layers=2, modelType='ANN', degree=2, pca_mode=False):
        super(QuantileMappingModel, self).__init__()
        self.degree = degree

        # Automatically calculate ny: degree scales + 1 shift + 1 threshold
        ny = degree + 1 
        self.model_type = modelType 
        self.pca_mode = pca_mode

        # if modelType == 'MLP':
        if num_layers==0:
            self.transform_generator = self.build_transform_generator(nx, hidden_dim, ny, 4)
        else:
            self.transform_generator = self.build_transform_generator(nx, hidden_dim, ny, num_layers)
        # elif modelType == 'FNO2d':
        #     self.transform_generator = FNO2d(16, 16, hidden_dim)
        # elif modelType == 'FNO1d':
        #     self.transform_generator = FNO1d(modes=16, width=hidden_dim, input_dim=nx, output_dim=ny)
        # elif modelType == 'LSTM':


        if num_layers == 0:
            self.lstm = nn.LSTM(input_size=nx, hidden_size=hidden_dim, num_layers=3, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=nx, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, ny) 
        # elif modelType == 'CNN1d':
        self.cnn = TemporalCNN(
            nx=nx, ny=ny, hidden=hidden_dim, num_blocks=num_layers,
            kernel_size=3, base_dilation=2,
            dropout=0.1, causal=False
        )

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
            params = [self.fc(lstm_out)]  # Output shape: [batch_size, seq_len, ny]
        elif self.model_type == "CNN1d":
            params = [self.cnn(input_tensor)]
        elif self.model_type == 'MLP_LSTM':
            params0 = self.transform_generator(input_tensor)
            lstm_out, _ = self.lstm(input_tensor)
            params1 = self.fc(lstm_out)
            params = [params0, params1]
        else:
            params = [self.transform_generator(input_tensor)]

        # Store all transformed outputs
        transformed_outputs = []

        for param in params:
            # Extract scale and shift parameters dynamically based on degree
            if str(time_scale) != 'daily':
                label_dummies = pd.get_dummies(time_scale)
                weights_np = label_dummies.div(label_dummies.sum(axis=0), axis=1).values.astype(np.float32)
                weights = torch.tensor(weights_np, device=param.device)  # shape (time, scale)
                label_avg = torch.einsum('stp,tm->smp', param, weights)  # (sites, scale, params)
                param = torch.einsum('tm,smp->stp', weights, label_avg)  # shape: (sites, time, params)

            if self.pca_mode:
                x_input, x_residual = self.extract_high_pca_modes(x)
            else:
                x_input = x

            wet_day_threshold = 0.1  # Set your threshold for wet days
            wet_mask = x_input > wet_day_threshold

            scales = [torch.exp(param[:, :, i]) for i in range(self.degree)]  # Ensure positive scaling
            shift = param[:, :, self.degree]

            # Apply polynomial transformation
            # transformed_x = sum((x_input ** (i + 1)) * scales[i] for i in range(self.degree)) + shift

            # Apply polynomial transformation only to wet days
            transformed_x = torch.zeros_like(x_input)
            transformed_x[wet_mask] = sum((x_input[wet_mask] ** (i + 1)) * scales[i][wet_mask] for i in range(self.degree)) + shift[wet_mask]

            # For dry days, you can keep the original value or set to zero
            transformed_x[~wet_mask] = x_input[~wet_mask]  # or 0

            # Apply thresholding and activation
            transformed_x = torch.relu(transformed_x)

            if self.pca_mode:
                transformed_x = transformed_x + x_residual

            transformed_outputs.append(transformed_x)

        # Average over all transformed outputs
        if len(transformed_outputs) > 1:
            final_output = torch.stack(transformed_outputs, dim=0).mean(dim=0)
        else:
            final_output = transformed_outputs[0]

        return final_output
    
    def extract_high_pca_modes(self, X: torch.Tensor, min_variance: float = 0.90) -> tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Center the data (remove spatial mean)
        X_mean = X.mean(dim=1, keepdim=True)
        X_centered = X - X_mean  # shape: (coords, time)

        # Step 2: Perform SVD
        # X = U @ S @ Vh
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)  # shapes: (coords, time), (min), (time, time)

        # Step 3: Compute cumulative variance explained
        S_squared = S ** 2
        explained_variance = torch.cumsum(S_squared, dim=0) / S_squared.sum()
        
        # Step 4: Choose minimum k such that at least min_variance is explained
        k = int((explained_variance >= min_variance).nonzero(as_tuple=True)[0][0].item()) + 1

        # Step 3: Keep top-k modes
        U_k = U[:, :k]           # (coords, k)
        S_k = S[:k]
        Vh_k = Vh[:k, :]         # (k, time)

        X_top = U_k @ torch.diag(S_k) @ Vh_k  # shape: (coords, time)
        X_top  = X_mean + X_top

        X_residual = X - X_top

        return X_top, X_residual
    

    def extract_high_fourier_mode(self, series: torch.Tensor, high_mode_cutoff: int = 100) -> torch.Tensor:
        """
        Extracts the high-frequency Fourier modes from a time series.

        Args:
            series (torch.Tensor): Input tensor of shape (batch, time)
            high_mode_cutoff (int): Frequencies above this are considered "high".

        Returns:
            torch.Tensor: Tensor containing only the high-frequency components.
        """
        freq_series = fft.fft(series, dim=-1)
        high_only_freq = torch.zeros_like(freq_series)
        high_only_freq[:, high_mode_cutoff:] = freq_series[:, high_mode_cutoff:]
        # Inverse FFT to real domain of just the high modes
        # high_only_series = fft.ifft(high_only_freq, dim=-1).real
        return high_only_freq
    
    def reassemble_high_fourier_modes(self, transformed_high: torch.Tensor, original_series: torch.Tensor, high_mode_cutoff: int = 100) -> torch.Tensor:
        """
        Reassembles the time series by combining the original low-frequency modes with the transformed high-frequency modes.

        Args:
            transformed_high (torch.Tensor): Transformed high-frequency components.
            original_series (torch.Tensor): Original time series to retain low-frequency components.
            high_mode_cutoff (int): Frequencies above this are considered "high".

        Returns:
            torch.Tensor: Reconstructed time series with transformed high frequencies.
        """
        freq_series = fft.fft(original_series, dim=-1)
        freq_series[:, high_mode_cutoff:] = transformed_high[:, high_mode_cutoff:]
        # Inverse FFT to get back to the time domain
        reconstructed_series = fft.ifft(freq_series, dim=-1).real
        return reconstructed_series
    




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







def selective_mode_polynomial_transform(
    series: torch.Tensor,
    poly_transform_fn,
    low_mode_cutoff: int = 10,
    high_mode_cutoff: int = 100
):
    """
    Applies a selective transformation to different Fourier modes of a time series using a polynomial transformation.
    
    Args:
        series (torch.Tensor): Input tensor of shape (batch, time)
        poly_transform_fn (callable): A function that takes a real-valued series and returns a transformed series.
        low_mode_cutoff (int): Frequencies below this are considered "low" and left untransformed.
        high_mode_cutoff (int): Frequencies above this are considered "high" and transformed.
    
    Returns:
        torch.Tensor: Reconstructed series after transforming selected modes.
    """
    # FFT (complex)
    freq_series = fft.fft(series, dim=-1)

    # Clone for modification
    transformed_freq = freq_series.clone()

    # High-frequency modes
    high_only_freq = torch.zeros_like(freq_series)
    high_only_freq[:, high_mode_cutoff:] = freq_series[:, high_mode_cutoff:]

    # Inverse FFT to real domain of just the high modes
    high_only_series = fft.ifft(high_only_freq, dim=-1).real

    # Apply polynomial transformation in time domain
    transformed_high = poly_transform_fn(high_only_series)

    # FFT of transformed high part
    transformed_high_freq = fft.fft(transformed_high, dim=-1)

    # Replace high-frequency components
    transformed_freq[:, high_mode_cutoff:] = transformed_high_freq[:, high_mode_cutoff:]

    # Inverse FFT to return to time domain
    reconstructed = fft.ifft(transformed_freq, dim=-1).real

    return reconstructed



### class of diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        pad = (kernel_size - 1) * dilation
        left_pad = pad if causal else pad // 2

        self.pad1 = (left_pad, 0) if causal else (pad // 2, pad - pad // 2)
        self.pad2 = (left_pad, 0) if causal else (pad // 2, pad - pad // 2)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        y = F.pad(x, self.pad1) if self.causal else F.pad(x, self.pad1, mode='constant', value=0)
        y = self.conv1(y)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)

        y = F.pad(y, self.pad2) if self.causal else F.pad(y, self.pad2, mode='constant', value=0)
        y = self.conv2(y)
        y = self.norm2(y)

        return F.gelu(x + self.dropout(y))  # residual

class TemporalCNN(nn.Module):
    """
    Temporal 1D CNN over rho (sequence length).
    Accepts [B, T, nx] and returns [B, T, ny].
    """
    def __init__(
        self,
        nx,
        ny,
        hidden=64,
        num_blocks=4,
        kernel_size=3,
        base_dilation=1,
        dropout=0.1,
        causal=False
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(nx, hidden, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            dilation = (base_dilation ** i) if base_dilation > 1 else (2 ** i)
            blocks.append(DilatedResBlock(hidden, kernel_size, dilation, dropout, causal))
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Conv1d(hidden, ny, kernel_size=1)

    def forward(self, x_b_t_nx):
        # x_b_t_nx: [B, T, nx]
        x = x_b_t_nx.transpose(1, 2)        # -> [B, nx, T]
        x = self.input_proj(x)              # -> [B, H, T]
        x = self.blocks(x)                  # -> [B, H, T]
        y = self.output_proj(x)             # -> [B, ny, T]
        return y.transpose(1, 2)