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



class MonotoneMap1D(nn.Module):
    """
    Monotone map:
      f(x) = alpha*x + sum_k w_k * softplus(s_k*(x - b_k)) + c
    with alpha, w_k, s_k >= 0 (via softplus), b_k, c free.
    """
    def __init__(self, n_bumps: int = 8, eps: float = 1e-6):
        super().__init__()
        self.n_bumps = n_bumps
        self.eps = eps

    def forward(self, x, packed_params):
        """
        x: (...,)
        packed_params: (..., P) where P = 2 + 3K   (alpha, c, [w_k, s_k, b_k]_k)
        Layout we expect:
          [ alpha, c, w_1..w_K, s_1..s_K, b_1..b_K ]
        """
        K = (packed_params.shape[-1] - 2) // 3
        assert K > 0

        alpha_raw = packed_params[..., 0]
        c         = packed_params[..., 1]
        w_raw     = packed_params[..., 2:2+K]
        s_raw     = packed_params[..., 2+K:2+2*K]
        b         = packed_params[..., 2+2*K:2+3*K]

        # constrain to >= 0
        alpha = F.softplus(alpha_raw) + self.eps
        w     = F.softplus(w_raw)     + self.eps
        s     = F.softplus(s_raw)     + self.eps

        # broadcast over K
        z = s * (x[..., None] - b)     # (..., K)
        bumps = F.softplus(z)
        y = alpha * x + (w * bumps).sum(dim=-1) + c
        return y

# Build the generator(s)
def build_transform_generator(nx, hidden_dim, ny, num_layers):
    layers = []
    layers.append(nn.Linear(nx, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.2))
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(hidden_dim, ny))
    return nn.Sequential(*layers)


class SpatioTemporalQM(nn.Module):
    def __init__(self, f_in, f_model=64, heads=4, t_blocks=3, st_layers=2, degree=8, dropout=0.1, transform_type='monotone', temp_enc='Conv1d'):
        super().__init__()
        self.embed = nn.Linear(f_in, f_model)
        self.stacks = nn.ModuleList([STBlock(f_model, heads=heads, t_hidden=2*f_model,
                                             t_blocks=t_blocks, dropout=dropout, tempModel=temp_enc) for _ in range(st_layers)])
        self.transform_type = transform_type
        if self.transform_type == 'monotone':
            self.to_params = nn.Linear(f_model, 2 + 3 * degree)  # alpha, c, (w_k, s_k, b_k) for k=1..K
        else:
            self.to_params = nn.Linear(f_model, degree + 1)
        self.monotone = MonotoneMap1D(n_bumps=degree)

    def forward(self, inps, patches_latlon, x_target):       # inps: (B,P,T,F_in), x_target: (B,P,T)
        h = self.embed(inps)                                 # (B,P,T,Fm)
        for blk in self.stacks:
            h = blk(h, patches_latlon)                       # spatio-temporal mixing
        params = self.to_params(h)                           # (B,P,T,ny)
        if self.transform_type == 'monotone':
            yhat = self.monotone(x_target, params)           # (B,P,T)
        else:
            # polynomial transform
            scales = [torch.exp(params[..., i]) for i in range(params.shape[-1]-1)]
            shift  = params[..., -1]
            yhat = sum((x_target ** (i + 1)) * scales[i] for i in range(len(scales))) + shift
        yhat = F.relu(yhat)
        return yhat, params


class STBlock(nn.Module):
    """
    Interleaved Spatio-Temporal Block:
      x -> TemporalConv1d -> SpatialAttention -> (residuals + norm)
    """
    def __init__(self, dim, heads=4, t_hidden=128, t_blocks=3, dropout=0.1, tempModel='Conv1d'):
        super().__init__()

        self.tempModel = tempModel
        if self.tempModel == 'Conv1d':
            self.tenc = TemporalConv1d(dim, hidden=t_hidden, n_blocks=t_blocks, dropout=dropout)
        elif self.tempModel == 'LSTM':
            self.tenc = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=t_blocks, dropout=dropout, batch_first=True)
        elif self.tempModel == 'MLP':
            self.tenc = build_transform_generator(dim, t_hidden, dim, t_blocks)
        elif self.tempModel == 'MLP+LSTM':
            self.tenc_mlp = build_transform_generator(dim, t_hidden, dim, t_blocks)
            self.tenc_lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=t_blocks, dropout=dropout, batch_first=True)
        elif self.tempModel == 'Transformer':
            self.tenc = TemporalSelfAttention(dim, heads=heads, ff_mult=2, dropout=dropout, causal=False)
        elif self.tempModel == 'Conv1d+MLP':
            self.tenc_conv = TemporalConv1d(dim, hidden=t_hidden, n_blocks=t_blocks, dropout=dropout)
            self.tenc_mlp = build_transform_generator(dim, t_hidden, dim, t_blocks)
        else:
            raise ValueError(f"Unknown tempModel type: {self.tempModel}")
        
        self.sattn = PatchSpatialAttention(dim, n_heads=heads, ff_mult=2, dropout=dropout)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x, pos):            # x: (B,P,T,F); pos: (B,P,2)
        B, P, T, F_ = x.size()
        if self.tempModel in ['LSTM', 'MLP', 'MLP+LSTM', 'Transformer', 'Conv1d+MLP']:
            x_ = x.view(B * P, T, F_) # (BP,T,F)
            if self.tempModel == 'LSTM':
                y_ = self.tenc(x_)[0] + x_  # (BP,T,F)
            elif self.tempModel == 'MLP':
                y_ = self.tenc(self.n1(x_)) + x_ # (BP,T,F)
            elif self.tempModel == 'MLP+LSTM':
                y_ = self.tenc_mlp(self.n1(x_)) + x_
                y_ = self.tenc_lstm(y_)[0] + y_
            elif self.tempModel == 'Transformer':
                y_ = self.tenc(self.n1(x_)) + x_
            elif self.tempModel == 'Conv1d+MLP':
                y_ = self.tenc_conv(self.n1(x_.view(B, P, T, F_))).view(B*P, T, F_) + x_
                y_ = self.tenc_mlp(self.n1(y_)) + y_
            y = y_.view(B, P, T, F_)  # (B,P,T,F)
        else:
            y = self.tenc(self.n1(x)) + x     # temporal residual
       
        z = self.sattn(self.n2(y), pos)   # spatial attn (already residual inside)
        return z

class QuantileMappingModel(nn.Module):
    def __init__(self, nx=1, hidden_dim=64, num_layers=2,
                 modelType='ANN', degree=2, pca_mode=False,
                 monotone=True):
        super(QuantileMappingModel, self).__init__()
        self.model_type = modelType
        self.pca_mode = pca_mode
        self.monotone = monotone
        self.degree = degree

        # Decide how many params the head must produce
        if monotone:
            # alpha, c, and (w_k, s_k, b_k) for k=1..K
            self.ny = 2 + 3 * degree
        else:
            # your original: degree scales + 1 shift
            self.ny = degree + 1

        

        # MLP
        if num_layers == 0:
            self.transform_generator = build_transform_generator(nx, hidden_dim, self.ny, 4)
        else:
            self.transform_generator = build_transform_generator(nx, hidden_dim, self.ny, num_layers)

        # LSTM
        lstm_layers = 3 if num_layers == 0 else num_layers
        self.lstm = nn.LSTM(input_size=nx, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.ny)

        # CNN
        self.cnn = TemporalCNN(
            nx=nx, ny=self.ny, hidden=hidden_dim, num_blocks=num_layers,
            kernel_size=3, base_dilation=2, dropout=0.1, causal=False
        )

        # Monotone map
        if monotone:
            self.monotone_map = MonotoneMap1D(n_bumps=degree)

    def _time_scale_pool(self, param, time_scale):
        # identical to your logic, just factored out
        if str(time_scale) == 'daily':
            return param
        label_dummies = pd.get_dummies(time_scale)
        weights_np = label_dummies.div(label_dummies.sum(axis=0), axis=1).values.astype(np.float32)
        weights = torch.tensor(weights_np, device=param.device)
        label_avg = torch.einsum('stp,tm->smp', param, weights)  # (sites, scale, params)
        param = torch.einsum('tm,smp->stp', weights, label_avg)  # (sites, time, params)
        return param

    def forward(self, x, input_tensor, time_scale):
        # Produce param tensors from the selected backbone(s)
        if self.model_type == "LSTM":
            lstm_out, _ = self.lstm(input_tensor)
            params = [ self.fc(lstm_out) ]                 # [B, T, ny]
        elif self.model_type == "CNN1d":
            params = [ self.cnn(input_tensor) ]            # [B, T, ny]
        elif self.model_type == "MLP_LSTM":
            p0 = self.transform_generator(input_tensor)    # [B, T, ny]
            lstm_out, _ = self.lstm(input_tensor)
            p1 = self.fc(lstm_out)                         # [B, T, ny]
            params = [p0, p1]
        else:  # "ANN"/default
            params = [ self.transform_generator(input_tensor) ]  # [B, T, ny]

        transformed_outputs = []
        pooled_params = []

        for param in params:
            # (Optional) pool across time_scale like your original code
            param = self._time_scale_pool(param, time_scale)

            # pca mode
            if self.pca_mode:
                x_input, x_residual = self.extract_high_pca_modes(x)
            else:
                x_input = x

            if self.monotone:
                # Monotone intensity transform (then gate at the end)
                transformed_x = self.monotone_map(x_input, param)
                # Final non-negativity gate (dry days => 0)
                transformed_x = F.relu(transformed_x)
            else:
                # Original polynomial + ReLU
                scales = [torch.exp(param[:, :, i]) for i in range(self.degree)]
                shift  = param[:, :, self.degree]
                transformed_x = sum((x_input ** (i + 1)) * scales[i] for i in range(self.degree)) + shift
                transformed_x = F.relu(transformed_x)

            if self.pca_mode:
                transformed_x = transformed_x + x_residual

            transformed_outputs.append(transformed_x)
            pooled_params.append(param)

        # Average if multiple heads (MLP_LSTM)
        if len(transformed_outputs) > 1:
            final_output = torch.stack(transformed_outputs, dim=0).mean(dim=0)
            params_out   = torch.stack(pooled_params, dim=0).mean(dim=0)
        else:
            final_output = transformed_outputs[0]
            params_out   = pooled_params[0]

        return final_output, params_out
    
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
    



class TemporalConv1d(nn.Module):
    def __init__(self, dim, hidden=128, kernel_size=3, base_dilation=1, n_blocks=3, dropout=0.1):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            dil = base_dilation * (2**i)
            blocks += [
                nn.Conv1d(dim, dim, kernel_size, padding=dil*(kernel_size-1)//2, dilation=dil, groups=dim),
                nn.GELU(),
                nn.GroupNorm(1, dim),            # <- channel norm on (N,C,T)
                nn.Conv1d(dim, hidden, 1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, dim, 1),
                nn.GroupNorm(1, dim),            # <- channel norm
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):  # x: (B,P,T,F)
        B, P, T, F = x.shape
        y = x.reshape(B*P, T, F).transpose(1, 2)  # (BP, F, T)
        k = 0
        for _ in range(len(self.blocks)//8):
            residual = y
            y = self.blocks[k+0](y); y = self.blocks[k+1](y); y = self.blocks[k+2](y)
            y = self.blocks[k+3](y); y = self.blocks[k+4](y); y = self.blocks[k+5](y)
            y = self.blocks[k+6](y); y = self.blocks[k+7](y)
            y = y + residual
            k += 8
        y = y.transpose(1, 2).reshape(B, P, T, F)  # back to (B,P,T,F)
        return y



def pairwise_relpos(latlon):  # latlon: (B, P, 2) [lat, lon] in degrees
    # returns rel: (B, P, P, 4): [dx, dy, great_circle_dist_km, bearing_sin]
    lat = torch.deg2rad(latlon[..., 0])
    lon = torch.deg2rad(latlon[..., 1])

    dlat = lat[:, :, None] - lat[:, None, :]
    dlon = lon[:, :, None] - lon[:, None, :]

    dx = dlon * torch.cos((lat[:, :, None] + lat[:, None, :]) / 2.0)
    dy = dlat

    # haversine distance (km)
    a = torch.sin(dlat/2)**2 + torch.cos(lat[:, :, None]) * torch.cos(lat[:, None, :]) * torch.sin(dlon/2)**2
    dist = 2 * 6371.0 * torch.arcsin(torch.clamp(torch.sqrt(a), 0, 1-1e-7))

    bearing = torch.atan2(
        torch.sin(dlon) * torch.cos(lat[:, None, :]),
        torch.cos(lat[:, :, None]) * torch.sin(lat[:, None, :]) - torch.sin(lat[:, :, None]) * torch.cos(lat[:, None, :]) * torch.cos(dlon)
    )
    rel = torch.stack([dx, dy, dist/500.0, torch.sin(bearing)], dim=-1)  # mild scaling for dist
    return rel  # (B, P, P, 4)

class PatchSpatialAttention(nn.Module):
    """
    Spatial self-attention over the K+1 nodes in each patch, per time step.
    Input:  x  (B, P, T, F)
            pos (B, P, 2) with [lat, lon] for each node in the patch (deg)
    Output: (B, P, T, F)
    """
    def __init__(self, dim, n_heads=4, ff_mult=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.h = n_heads
        self.dk = dim // n_heads
        assert dim % n_heads == 0

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult*dim),
            nn.GELU(),
            nn.Linear(ff_mult*dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        # Rel-pos -> bias per head
        self.relproj = nn.Linear(4, n_heads, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x, pos):
        """
        x:   (B, P, T, F)
        pos: (B, P, 2) lat/lon degrees for nodes in each patch
        """
        B, P, T, F_ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, P, F)
        x_ = self.norm1(x)

        # QKV along spatial nodes for each time slice
        qkv = self.qkv(x_)  # (B, T, P, 3F)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape for heads
        def split_heads(t):
            return t.view(B, T, P, self.h, self.dk).permute(0,1,3,2,4)  # (B,T,H,P,dk)
        q, k, v = map(split_heads, (q, k, v))

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,T,H,P,P)

        # add relative positional bias per head
        rel = pairwise_relpos(pos)             # (B,P,P,4)
        rel_h = self.relproj(rel).permute(0,3,1,2)  # (B,H,P,P)
        attn = attn + rel_h[:, None, ...]      # broadcast to (B,T,H,P,P)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)            # (B,T,H,P,dk)

        # merge heads
        out = out.permute(0,1,3,2,4).contiguous().view(B, T, P, F_)  # (B,T,P,F)
        out = self.out(out)
        x = x + self.dropout(out)              # residual
        y = self.norm2(x)
        y = y + self.dropout(self.ff(y))       # feed-forward + residual
        y = y.permute(0,2,1,3).contiguous()    # (B,P,T,F)
        return y


class TemporalSelfAttention(nn.Module):
    """
    Temporal Transformer block (per patch):
      Input:  (B*P, T, F)
      Output: (B*P, T, F)
    """
    def __init__(self, dim, heads=4, ff_mult=2, dropout=0.1, causal=False, max_T=6000):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.causal = causal

        # Learned positional embeddings over time steps
        # self.pos_emb = nn.Embedding(max_T, dim)

        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def sinusoidal_pos_emb(T, dim, device='cpu'):
        """
        Returns sinusoidal positional encodings (T, dim)
        for positions [0, T-1].
        """
        pos = torch.arange(T, device=device).unsqueeze(1)           # (T, 1)
        i = torch.arange(dim // 2, device=device).unsqueeze(0)      # (1, dim/2)
        denom = torch.pow(10000, (2 * i) / dim)
        angles = pos / denom                                        # (T, dim/2)
        pe = torch.zeros(T, dim, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def _causal_mask(self, T, device):
        # [T, T] upper-triangular mask: True = mask out
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x):  # x: (B*P, T, F)
        BP, T, F = x.shape
        # pos_ids = torch.arange(T, device=x.device)
        # x = x + self.pos_emb(pos_ids)[None, :, :]  # broadcast over batch
        x = x + self.sinusoidal_pos_emb(T, F, x.device)

        # Pre-norm
        h = self.ln1(x)

        attn_mask = self._causal_mask(T, x.device) if self.causal else None
        # Self-attention
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        y = y + x  # residual

        # FFN
        z = self.ln2(y)
        z = self.ff(z)
        z = z + y  # residual
        return z




# class QuantileMappingModel(nn.Module):
#     def __init__(self, nx=1, hidden_dim=64, num_layers=2, modelType='ANN', degree=2, pca_mode=False):
#         super(QuantileMappingModel, self).__init__()
#         self.degree = degree

#         # Automatically calculate ny: degree scales + 1 shift + 1 threshold
#         ny = degree + 1 
#         self.model_type = modelType 
#         self.pca_mode = pca_mode

#         # if modelType == 'MLP':
#         if num_layers==0:
#             self.transform_generator = self.build_transform_generator(nx, hidden_dim, ny, 4)
#         else:
#             self.transform_generator = self.build_transform_generator(nx, hidden_dim, ny, num_layers)
#         # elif modelType == 'FNO2d':
#         #     self.transform_generator = FNO2d(16, 16, hidden_dim)
#         # elif modelType == 'FNO1d':
#         #     self.transform_generator = FNO1d(modes=16, width=hidden_dim, input_dim=nx, output_dim=ny)
#         # elif modelType == 'LSTM':


#         if num_layers == 0:
#             self.lstm = nn.LSTM(input_size=nx, hidden_size=hidden_dim, num_layers=3, batch_first=True)
#         else:
#             self.lstm = nn.LSTM(input_size=nx, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            
#         self.fc = nn.Linear(hidden_dim, ny) 
#         # elif modelType == 'CNN1d':
#         self.cnn = TemporalCNN(
#             nx=nx, ny=ny, hidden=hidden_dim, num_blocks=num_layers,
#             kernel_size=3, base_dilation=2,
#             dropout=0.1, causal=False
#         )

#     def build_transform_generator(self, nx, hidden_dim, ny, num_layers):
#         layers = []
        
#         # Input layer
#         layers.append(nn.Linear(nx, hidden_dim))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(0.2))

#         # Hidden layers
#         for _ in range(num_layers - 2):  # -2 because we have input and output layers separately
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.2))

#         # Output layer
#         layers.append(nn.Linear(hidden_dim, ny))

#         return nn.Sequential(*layers)

#     def forward(self, x, input_tensor, time_scale):
#         # Generate transformation parameters
#         if self.model_type == "LSTM":
#             # LSTM forward pass
#             lstm_out, _ = self.lstm(input_tensor)  # Output shape: [batch_size, seq_len, hidden_dim]
#             # Generate transformation parameters
#             params = [self.fc(lstm_out)]  # Output shape: [batch_size, seq_len, ny]
#         elif self.model_type == "CNN1d":
#             params = [self.cnn(input_tensor)]
#         elif self.model_type == 'MLP_LSTM':
#             params0 = self.transform_generator(input_tensor)
#             lstm_out, _ = self.lstm(input_tensor)
#             params1 = self.fc(lstm_out)
#             params = [params0, params1]
#         else:
#             params = [self.transform_generator(input_tensor)]

#         # Store all transformed outputs
#         transformed_outputs = []

#         for param in params:
#             # Extract scale and shift parameters dynamically based on degree
#             if str(time_scale) != 'daily':
#                 label_dummies = pd.get_dummies(time_scale)
#                 weights_np = label_dummies.div(label_dummies.sum(axis=0), axis=1).values.astype(np.float32)
#                 weights = torch.tensor(weights_np, device=param.device)  # shape (time, scale)
#                 label_avg = torch.einsum('stp,tm->smp', param, weights)  # (sites, scale, params)
#                 param = torch.einsum('tm,smp->stp', weights, label_avg)  # shape: (sites, time, params)

#             if self.pca_mode:
#                 x_input, x_residual = self.extract_high_pca_modes(x)
#             else:
#                 x_input = x

#             # wet_day_threshold = 0.1  # Set your threshold for wet days
#             # wet_mask = x_input > wet_day_threshold

#             scales = [torch.exp(param[:, :, i]) for i in range(self.degree)]  # Ensure positive scaling
#             shift = param[:, :, self.degree]

#             # # Apply polynomial transformation
#             transformed_x = sum((x_input ** (i + 1)) * scales[i] for i in range(self.degree)) + shift

#             # # # Apply polynomial transformation only to wet days
#             # transformed_x = torch.zeros_like(x_input)
#             # transformed_x[wet_mask] = sum((x_input[wet_mask] ** (i + 1)) * scales[i][wet_mask] for i in range(self.degree)) + shift[wet_mask]

#             # # # # For dry days, you can keep the original value or set to zero
#             # transformed_x[~wet_mask] = x_input[~wet_mask]  # or 0

#             # Apply thresholding and activation
#             transformed_x = torch.relu(transformed_x)

#             if self.pca_mode:
#                 transformed_x = transformed_x + x_residual

#             transformed_outputs.append(transformed_x)

#         # Average over all transformed outputs
#         if len(transformed_outputs) > 1:
#             final_output = torch.stack(transformed_outputs, dim=0).mean(dim=0)
#             params = torch.stack(params, dim=0).mean(dim=0)
#         else:
#             final_output = transformed_outputs[0]
#             params = params[0]

#         return final_output, params
    
#     def extract_high_pca_modes(self, X: torch.Tensor, min_variance: float = 0.90) -> tuple[torch.Tensor, torch.Tensor]:
#         # Step 1: Center the data (remove spatial mean)
#         X_mean = X.mean(dim=1, keepdim=True)
#         X_centered = X - X_mean  # shape: (coords, time)

#         # Step 2: Perform SVD
#         # X = U @ S @ Vh
#         U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)  # shapes: (coords, time), (min), (time, time)

#         # Step 3: Compute cumulative variance explained
#         S_squared = S ** 2
#         explained_variance = torch.cumsum(S_squared, dim=0) / S_squared.sum()
        
#         # Step 4: Choose minimum k such that at least min_variance is explained
#         k = int((explained_variance >= min_variance).nonzero(as_tuple=True)[0][0].item()) + 1

#         # Step 3: Keep top-k modes
#         U_k = U[:, :k]           # (coords, k)
#         S_k = S[:k]
#         Vh_k = Vh[:k, :]         # (k, time)

#         X_top = U_k @ torch.diag(S_k) @ Vh_k  # shape: (coords, time)
#         X_top  = X_mean + X_top

#         X_residual = X - X_top

#         return X_top, X_residual
    

#     def extract_high_fourier_mode(self, series: torch.Tensor, high_mode_cutoff: int = 100) -> torch.Tensor:
#         """
#         Extracts the high-frequency Fourier modes from a time series.

#         Args:
#             series (torch.Tensor): Input tensor of shape (batch, time)
#             high_mode_cutoff (int): Frequencies above this are considered "high".

#         Returns:
#             torch.Tensor: Tensor containing only the high-frequency components.
#         """
#         freq_series = fft.fft(series, dim=-1)
#         high_only_freq = torch.zeros_like(freq_series)
#         high_only_freq[:, high_mode_cutoff:] = freq_series[:, high_mode_cutoff:]
#         # Inverse FFT to real domain of just the high modes
#         # high_only_series = fft.ifft(high_only_freq, dim=-1).real
#         return high_only_freq
    
#     def reassemble_high_fourier_modes(self, transformed_high: torch.Tensor, original_series: torch.Tensor, high_mode_cutoff: int = 100) -> torch.Tensor:
#         """
#         Reassembles the time series by combining the original low-frequency modes with the transformed high-frequency modes.

#         Args:
#             transformed_high (torch.Tensor): Transformed high-frequency components.
#             original_series (torch.Tensor): Original time series to retain low-frequency components.
#             high_mode_cutoff (int): Frequencies above this are considered "high".

#         Returns:
#             torch.Tensor: Reconstructed time series with transformed high frequencies.
#         """
#         freq_series = fft.fft(original_series, dim=-1)
#         freq_series[:, high_mode_cutoff:] = transformed_high[:, high_mode_cutoff:]
#         # Inverse FFT to get back to the time domain
#         reconstructed_series = fft.ifft(freq_series, dim=-1).real
#         return reconstructed_series