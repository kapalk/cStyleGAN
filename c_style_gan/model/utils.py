from torch import nn

import torch
import torch.nn.functional as F
import numpy as np

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'lrelu': nn.LeakyReLU(negative_slope=0.2)
}


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNormLayer, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        axis = (2, 3)

        if x.dim() == 3:
            axis = (1, 2)

        x = x - torch.mean(x, axis, True)
        return x * torch.rsqrt(torch.mean(x**2, axis, True) + self.epsilon)


class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, gain=np.sqrt(2), use_wscale=False, lr_mul=1):
        super(EqualizedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gain = gain
        self.use_wscale = use_wscale
        self.lr_mul = lr_mul

        he_std = self.gain * self.in_features ** (-0.5)  # init He

        if self.use_wscale:
            init_std = 1.0 / self.lr_mul
            self.weight_mul = he_std * self.lr_mul
        else:
            init_std = he_std / self.lr_mul
            self.weight_mul = lr_mul

        self.bias_mul = lr_mul
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features) * init_std)
        self.bias = nn.Parameter(torch.zeros(self.out_features))

    def forward(self, x):
        return F.linear(x, self.weight * self.weight_mul, self.bias * self.bias_mul)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, gain={self.gain},' \
            f' use_wscale={self.use_wscale}, lr_mul={self.lr_mul}'


class Upscale2D(nn.Module):

    def __init__(self, factor=2, gain=1):
        super(Upscale2D, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x *= self.gain

        if self.factor != 1:
            shape = x.shape

            if x.dim() == 4:
                x = x.view(
                    shape[0],
                    shape[1],
                    shape[2],
                    1,
                    shape[3],
                    1
                ).expand(-1, -1, -1, self.factor, -1, self.factor)
                x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
            elif x.dim() == 3:
                x = x.view(shape[0], shape[1], shape[2], 1).expand(-1, -1, -1, self.factor)
                x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2])
            else:
                raise Exception("Input dimensions {0} not supported".format(shape))

        return x

    def extra_repr(self):
        return f'factor={self.factor}, gain={self.gain}'


class BlurLayer(nn.Module):

    def __init__(self, dimension, kernel=(1, 2, 1), normalize=True, stride=1):
        super(BlurLayer, self).__init__()
        self.normalize = normalize
        self.dimension = dimension
        kernel = torch.tensor(kernel, dtype=torch.float32)

        if self.dimension == 2:
            kernel = kernel[:, None] * kernel[None, :]
            kernel = kernel[None, None]
        elif self.dimension == 1:
            kernel = kernel * kernel
            kernel = kernel[None]

        if self.normalize:
            kernel = kernel / kernel.sum()

        self.kernel = kernel
        self.stride = stride

    def forward(self, x):
        if x.dim() == 4:
            kernel = self.kernel.expand(x.size(1), -1, -1, -1).to(x.device)
            return F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1))
        elif x.dim() == 3:
            kernel = self.kernel.expand(x.size(1), -1, -1).to(x.device)
            return F.conv1d(x, kernel, stride=self.stride, padding=int((self.kernel.size(1) - 1) / 2), groups=x.size(1))
        else:
            raise Exception("Input dimensions {0} not supported".format(x.shape))

    def extra_repr(self):
        return f'kernel={self.kernel}, stride={self.stride}, normalize={self.normalize}'


class Downscale2D(nn.Module):

    def __init__(self, dimension, factor=2, gain=1):
        super(Downscale2D, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor
        self.blur = None
        self.dimension = dimension

        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(dimension=self.dimension, kernel=f, normalize=False, stride=factor)

    def forward(self, x):
        if self.dimension == 2:
            assert x.dim() == 4
        elif self.dimension == 1:
            assert x.dim() == 3

        # 2x2, float32 => downscale using blur2d.
        if self.blur is not None and x.dtype == torch.float32:
            x = self.blur(x)
            return x

        if self.gain != 1:
            x *= self.gain

        # Large factor => downscale using F.avg_pool2d().
        if self.factor != 1:
            return F.avg_pool2d(x, self.factor)

    def extra_repr(self):
        return f'factor={self.factor}, gain={self.gain}'


class EqualizedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dimension, stride=1, padding=0, use_wscale=False,
                 gain=np.sqrt(2), lr_mul=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.gain = gain
        self.lr_mul = lr_mul
        self.use_wscale = use_wscale

        he_std = self.gain * (self.in_channels * self.kernel_size ** 2) ** (-0.5)  # init He

        if self.use_wscale:
            init_std = 1.0 / self.lr_mul
            self.weight_mul = he_std * self.lr_mul
        else:
            init_std = he_std / self.lr_mul
            self.weight_mul = self.lr_mul

        self.bias_mul = lr_mul

        if dimension == 1:
            self.weight = nn.Parameter(
                torch.randn(self.out_channels, self.in_channels, self.kernel_size) * init_std
            )
        else:
            self.weight = nn.Parameter(
                torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) * init_std
            )

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.padding = padding
        self.stride = stride
        self.dimension = dimension

    def forward(self, x):
        if self.dimension == 1:
            return F.conv1d(x, self.weight * self.weight_mul, self.bias * self.bias_mul, self.stride, self.padding)

        return F.conv2d(x, self.weight * self.weight_mul, self.bias * self.bias_mul, self.stride, self.padding)

    def extra_repr(self):
        return f'dimension={self.dimension}, in_channels={self.in_channels}, ' \
            f'out_channels={self.out_channels}, kernel_size={self.kernel_size}, ' \
            f'stride={self.stride}, use_wscale={self.use_wscale}, gain={self.gain}, lr_mul={self.lr_mul}'


class FusedUpsample(nn.Module):
    """https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py#L56"""

    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        weight = torch.randn(self.in_channel, self.out_channel, self.kernel_size, self.kernel_size)
        bias = torch.zeros(self.out_channel)

        fan_in = self.in_channel * self.kernel_size * self.kernel_size
        self.multiplier = np.sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, x):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(x, weight, self.bias, stride=2, padding=self.pad)

        return out

    def extra_repr(self):
        return f'in_channel={self.in_channel}, out_channels={self.out_channel},' \
            f' kernel_size={self.kernel_size}, padding={self.pad}'


class FusedDownsample(nn.Module):
    """https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py#L85"""

    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        weight = torch.randn(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        bias = torch.zeros(self.out_channel)

        fan_in = self.in_channel * self.kernel_size * self.kernel_size
        self.multiplier = np.sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, x):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(x, weight, self.bias, stride=2, padding=self.pad)

        return out

    def extra_repr(self):
        return f'in_channel={self.in_channel}, out_channels={self.out_channel},' \
            f' kernel_size={self.kernel_size}, padding={self.pad}'
