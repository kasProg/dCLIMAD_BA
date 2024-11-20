import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(SpectralConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

#     def forward(self, x):
#         batchsize, _, size1, size2 = x.shape
#         # Compute Fourier coefficients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfft2(x)

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels, size1, size2//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2] = \
#             self.weights1 * x_ft[:, :, :self.modes1, :self.modes2].unsqueeze(1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#             self.weights2 * x_ft[:, :, -self.modes1:, :self.modes2].unsqueeze(1)

#         # Return to physical space
#         x = torch.fft.irfft2(out_ft, s=(size1, size2))
#         return x

# class FNO2d(nn.Module):
#     def __init__(self, modes1, modes2, width):
#         super(FNO2d, self).__init__()
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.width = width
#         self.padding = 2  # pad the domain if input is non-periodic

#         self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

#         self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.w0 = nn.Conv2d(self.width, self.width, 1)
#         self.w1 = nn.Conv2d(self.width, self.width, 1)
#         self.w2 = nn.Conv2d(self.width, self.width, 1)
#         self.w3 = nn.Conv2d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 3)

#     def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)
        
#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2

#         x = x.permute(0, 2, 3, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize, _, size1, size2 = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, size1, size2//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.weights1 * x_ft[:, :, :self.modes1, :self.modes2].unsqueeze(1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.weights2 * x_ft[:, :, -self.modes1:, :self.modes2].unsqueeze(1)

        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.conv0 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(width, width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc0 = nn.Linear(1, self.width)  # Changed input channel to 1
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.fc0(x.unsqueeze(-1))
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    

# class SpectralConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1):
#         super(SpectralConv1d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

#     def forward(self, x):
#         batchsize, size1, _ = x.shape
#         # Compute Fourier coefficients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfft(x)

#         # Multiply relevant Fourier modes
#         # out_ft = torch.zeros(batchsize, self.out_channels, size1 // 2 + 1, device=x.device, dtype=torch.cfloat)
#         # out_ft[:, :, :self.modes1] = self.weights1 * x_ft[:, :, :self.modes1].unsqueeze(1)
#         out_ft = torch.einsum("bi,io->bo", x_ft, self.weights1)

#         # Return to physical space
#         x = torch.fft.irfft(out_ft, n=size1)
#         return x

# class FNO1d(nn.Module):
#     def __init__(self, modes, width, input_dim):
#         super(FNO1d, self).__init__()
#         self.modes = modes
#         self.width = width

#         self.fc0 = nn.Linear(input_dim, self.width)

#         self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
#         self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
#         self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
#         self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 3)  # Output 3 parameters for scale, shift, and threshold

#     def forward(self, x):
#         # x shape: (batch, time_steps, inputs)
#         x = self.fc0(x)
#         # x shape after fc0: (batch, time_steps, width)
#         x = x.permute(0, 2, 1)
#         # x shape after permute: (batch, width, time_steps)

#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2

#         # x shape before permute: (batch, width, time_steps)
#         x = x.permute(0, 2, 1)
#         # x shape after permute: (batch, time_steps, width)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        1D Spectral convolution layer.

        Args:
        - in_channels (int): Number of input channels
        - out_channels (int): Number of output channels
        - modes1 (int): Number of Fourier modes to multiply (not directly used in this implementation)
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        # Complex-valued weights for the Fourier space transformation
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, dtype=torch.cfloat))

    def forward(self, x):
        """
        Forward pass of the spectral convolution.

        Args:
        - x (Tensor): Input tensor of shape [batch_size, in_channels]

        Returns:
        - Tensor: Output tensor of shape [batch_size, out_channels]
        """
        # Compute Fourier coefficients
        x_ft = torch.fft.fft(x, dim=1)
        # Initialize output Fourier coefficients
        # out_ft = torch.zeros(x.shape[0], self.out_channels, dtype=torch.cfloat, device=x.device)
        # Multiply relevant Fourier modes
        out_ft = torch.einsum("bti,io->bto", x_ft, self.weights.to(x_ft.dtype))
        # Return to physical space
        x = torch.fft.ifft(out_ft).real
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, input_dim, output_dim):
        """
        1D Fourier Neural Operator.

        Args:
        - modes (int): Number of Fourier modes to multiply
        - width (int): Number of channels in the convolutional layers
        """
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width

        # Initial fully connected layer to project input to 'width' dimensions
        self.fc0 = nn.Linear(input_dim, self.width)

        # Four spectral convolution layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
# 
        # Four linear layers for the residual connections
        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)

        # Final fully connected layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass of the FNO.

        Args:
        - x (Tensor): Input tensor of shape [batch_size, 3] representing (x, y, t) coordinates

        Returns:
        - Tensor: Output tensor of shape [batch_size] representing the predicted values
        """
        # Initial projection
        x = self.fc0(x)

        # Four layers of the integral operator
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = torch.tanh(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = torch.tanh(x)


        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = torch.tanh(x)


        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Final fully connected layers
        x = self.fc1(x)
        # x = F.gelu(x)
        x = torch.tanh(x)
        x = self.fc2(x)

        return x