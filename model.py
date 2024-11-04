import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Parameter
import math
import torch.nn.functional as F

class DropMask(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, mask, train=False, inplace=False):
        ctx.train = train
        ctx.inplace = inplace
        ctx.mask = mask

        if not ctx.train:
            return input
        else:
            if ctx.inplace:
                ctx.mark_dirty(input)
                output = input
            else:
                output = input.clone()
            output.mul_(ctx.mask)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * ctx.mask, None, None, None
        else:
            return grad_output, None, None, None

def createMask(x, dr):
    mask = x.new().resize_as_(x).bernoulli_(1 - dr).div_(1 - dr).detach_()
    # print('droprate='+str(dr))
    return mask

class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 1, False, 0, self.training, False, (), None)
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 0, 1, False, 0, self.training, False, (), None)
        return output, (hy, cy)

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)
        return out


class QuantileMappingModel_(nn.Module):
    def __init__(self, nx=1, ny=3, num_series=100, hidden_dim=64):
        super(QuantileMappingModel_, self).__init__()
        self.num_series = num_series

        # Neural network to generate transformation parameters
        self.transform_generator = nn.Sequential(
            nn.Linear(nx, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ny)  # Output: [scale1, scale2, shift, threshold]
        )

        # self.lstminv = CudnnLstmModel(nx=nx, ny=ny, hiddenSize=hidden_dim, dr=0.5)

    def normalize_elevation(self, elevation):
        mean = torch.mean(elevation)
        std_dev = torch.std(elevation)
        normalized_elevation = (elevation - mean) / std_dev
        return normalized_elevation.unsqueeze(1)

    def forward(self, x, input_tensor):

        params = self.transform_generator(input_tensor)
        # params = self.lstminv(input_tensor)
        scale1 = torch.exp(params[:, :, 0])  # Ensure positive scaling
        # scale1 = 1 + 0.2*(torch.clamp(params[:, :, 0], min=-1, max=1))  # Ensure positive scaling
        shift1 = params[:, :, 1]
        threshold = torch.sigmoid(params[:, :, 2])*0.1 # Between 0 and 1
        # power = torch.sigmoid(params[:, :, 3])* 3   # Range: [0.5, 1]
        power = 1.0

        # scale2 = torch.exp(params[:, :, 3]) # Ensure positive scaling
        # shift2 = params[:, 4].unsqueeze(0)

        # Apply transformation
        # transformed_x = torch.log1p(x) * scale1 + shift1
        transformed_x = ((x**power)*scale1) + shift1
        # transformed_x = (scale2 * (x**2)) + (x * scale1) + shift1
        # transformed_x = transformed_x * scale2 + shift2
        # transformed_x = transformed_x * scale3 + shift3
        # torch.save(scale1, 'scale1.pt')
        # torch.save(shift, 'shift.pt')

        # Apply threshold-based zero handling
        zero_mask = x <= threshold
        transformed_x = torch.where(zero_mask, torch.zeros_like(transformed_x), transformed_x)

        # Ensure non-negative values using softplus
        transformed_x = torch.relu(transformed_x)

        # e = transformed_x - x
        # torch.save(e, 'e.pt')
        return transformed_x


class QuantileMappingModel(nn.Module):
    def __init__(self, nx=2, ny=4, num_series=100, hidden_dim=128):
        super(QuantileMappingModel, self).__init__()
        self.num_series = num_series

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(nx, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Dropout(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, ny)
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

    def forward(self, x, input_tensor):
        # Encoder
        encoded = self.encoder(input_tensor)

        # Apply attention
        attn_output, _ = self.attention(encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0))
        attn_output = attn_output.squeeze(0)

        # Decode
        params = self.decoder(attn_output)

        # Extract transformation parameters
        scale = torch.sigmoid(params[:, 0]).unsqueeze(1) * 0.4 + 0.8  # Range: [0.8, 1.2]
        shift = params[:, 1].unsqueeze(1)
        threshold = torch.sigmoid(params[:, 2]).unsqueeze(1) * 0.1  # Range: [0, 0.1]
        power = torch.sigmoid(params[:, 3]).unsqueeze(1) * 0.5 + 0.5  # Range: [0.5, 1]

        # Apply non-linear transformation
        transformed_x = (x ** power) * scale + shift

        # Apply threshold-based zero handling
        zero_mask = x <= threshold
        transformed_x = torch.where(zero_mask, torch.zeros_like(transformed_x), transformed_x)

        # Ensure non-negative values
        transformed_x = F.softplus(transformed_x)

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