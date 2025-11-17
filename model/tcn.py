import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    Trim off the extra padding on the right so the conv is causal.
    Input:  (N, C, L + pad)
    Output: (N, C, L)
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    Standard TCN residual block:
    Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout + Residual.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation   # causal padding (only left)

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # If channels differ, use 1x1 conv for residual
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs else
            None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):  # x: (N, C_in, L)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)



class TemporalTCN(nn.Module):
    """
    TCN over time with residual blocks and exponentially increasing dilation.

    Input:  x: (B, P, T, F)
    Output: (B, P, T, F)
    """
    def __init__(
        self,
        dim,               # F (features per node)
        hidden=128,
        kernel_size=3,
        base_dilation=1,
        n_blocks=3,
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        in_channels = dim
        for i in range(n_blocks):
            dilation = base_dilation * (2 ** i)
            out_channels = hidden if i < n_blocks - 1 else dim  # last back to dim

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, P, T, F)
        B, P, T, F = x.shape

        # (B, P, T, F) -> (BP, F, T)
        y = x.reshape(B * P, T, F).transpose(1, 2)

        # TCN over time
        y = self.network(y)  # (BP, F, T)

        # back to (B, P, T, F)
        y = y.transpose(1, 2).reshape(B, P, T, F)
        return y
