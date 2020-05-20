from torch import nn
from .utils import ACTIVATION_FUNCTIONS, PixelNormLayer, EqualizedLinear, EqualizedConv, Upscale2D, InstanceNorm, \
    BlurLayer, FusedUpsample
from collections import OrderedDict
import torch
import numpy as np


class NoiseLayer(nn.Module):

    def __init__(self, channels, noise=None):
        super(NoiseLayer, self).__init__()
        self.noise = noise
        self.channels = channels
        self.weight = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            if x.dim() == 4:
                noise = torch.randn((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
            else:
                noise = torch.randn((x.size(0), 1, x.size(2)), device=x.device, dtype=x.dtype)
        elif noise is None:
            noise = self.noise
        if x.dim() == 4:
            return x + self.weight.view(1, -1, 1, 1) * noise
        else:
            return x + self.weight.view(1, -1, 1) * noise

    def extra_repr(self):
        return 'channels={channels}'.format(channels=self.channels)


class StyleMod(nn.Module):

    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.layer = EqualizedLinear(
            in_features=latent_size,
            out_features=2 * channels,
            gain=1.,
            use_wscale=use_wscale
        )

    def forward(self, x, latent):
        style = self.layer(latent)
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)
        out = x * (style[:, 0] + 1.) + style[:, 1]
        return out

class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""
    def __init__(
            self,
            channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation
    ):
        super(LayerEpilogue, self).__init__()

        layers = []

        if use_noise:
            layers.append(NoiseLayer(channels))
        layers.append(activation)

        if use_pixel_norm:
            layers.append(PixelNormLayer())

        if use_instance_norm:
            layers.append(InstanceNorm())

        self.top_epilogue = nn.Sequential(*layers)

        self.style_mod = None
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale)

    def forward(self, x, dlatent_slice=None):
        x = self.top_epilogue(x)

        if self.style_mod is not None:
            x = self.style_mod(x, dlatent_slice)

        return x


class InputBlock(nn.Module):

    def __init__(
            self,
            nf,
            dlatent_size,
            kernel_size,
            const_input_layer,
            gain,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
            conv_dim
    ):
        super(InputBlock, self).__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        self.conv_dim = conv_dim

        if self.const_input_layer:
            if self.conv_dim == 1:
                self.const = nn.Parameter(torch.ones(1, nf, 4))
                self.bias = nn.Parameter(torch.ones(nf))
            else:
                self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
                self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.linear = EqualizedLinear(
                in_features=dlatent_size,
                out_features=nf * 16,
                gain=gain / 4,
                use_wscale=use_wscale
            )

        self.epi1 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer
        )

        self.conv = EqualizedConv(
            in_channels=nf,
            out_channels=nf,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            use_wscale=use_wscale,
            gain=gain,
            dimension=conv_dim
        )

        self.epi2 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer
        )

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)

        if self.const_input_layer:
            if self.conv_dim == 1:
                x = self.const.expand(batch_size, -1, -1)
                x = x + self.bias.view(1, -1, 1)
            else:
                x = self.const.expand(batch_size, -1, -1, -1)
                x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.linear(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)

        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        return self.epi2(x, dlatents_in_range[:, 1])


class SynthesisBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            blur_filter,
            dlatent_size,
            gain,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
            conv_dim
    ):
        super(SynthesisBlock, self).__init__()

        self.upscale = None
        if in_channels * 2 >= 128:
            self.conv0 = FusedUpsample(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        else:
            self.upscale = Upscale2D()

            self.conv0 = EqualizedConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                gain=gain,
                use_wscale=use_wscale,
                dimension=conv_dim

            )

        self.blur = None
        if blur_filter:
            self.blur = BlurLayer(
                dimension=conv_dim,
                kernel=blur_filter
            )

        self.epi1 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer
        )

        self.conv1 = EqualizedConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            gain=gain,
            use_wscale=use_wscale,
            dimension=conv_dim
        )

        self.epi2 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer
        )

    def forward(self, x, dlatents_in_range):
        if self.upscale:
            x = self.upscale(x)
        x = self.conv0(x)

        if self.blur is not None:
            x = self.blur(x)

        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class SynthesisNet(nn.Module):

    def __init__(self, conv_dim=2, dlatent_size=512, num_channels=3, resolution=1024, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu', use_wscale=True,
                 use_pixel_norm=False, use_instance_norm=True, blur_filter=(1, 2, 1), structure="fixed"):
        """
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param num_channels: Number of output color channels.
        :param resolution: Output resolution.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param use_styles: Enable style inputs
        :param const_input_layer: First layer is a learned constant?
        :param use_noise: Enable noise inputs
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate
        :param use_pixel_norm: Enable pixelwise feature vector normalization
        :param use_instance_norm: Enable instance normalization
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering
        :param structure: Structure of the forward pass: fixed or linear
        """
        super(SynthesisNet, self).__init__()
        self.structure = structure
        self.resolution = resolution

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        kernel_size = 3

        activation = ACTIVATION_FUNCTIONS[nonlinearity]
        gain = np.sqrt(2)
        blocks = list()
        to_rgb = list()
        last_channels = None

        for res in range(2, resolution_log2 + 1):
            s = 2 ** res
            channels = nf(res - 1)
            name = f'{s}x{s}'

            if res == 2:
                self.input_block = InputBlock(
                    channels,
                    dlatent_size,
                    kernel_size,
                    const_input_layer,
                    gain,
                    use_wscale,
                    use_noise,
                    use_pixel_norm,
                    use_instance_norm,
                    use_styles,
                    activation, conv_dim
                )
            else:
                blocks.append(
                    (name,
                     SynthesisBlock(
                         last_channels,
                         channels,
                         kernel_size,
                         blur_filter,
                         dlatent_size,
                         gain,
                         use_wscale,
                         use_noise,
                         use_pixel_norm,
                         use_instance_norm,
                         use_styles,
                         activation,
                         conv_dim
                     )
                     )
                )
            to_rgb.append(
                (f'to RGB {name}',
                 EqualizedConv(
                     channels,
                     num_channels,
                     1,
                     gain=1,
                     use_wscale=use_wscale,
                     dimension=conv_dim
                 )
                 )
            )
            last_channels = channels

        self.upscale = Upscale2D()
        self.to_rgb = nn.ModuleDict(OrderedDict(to_rgb))
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(self, dlatents_in, step=8, alpha=0.5):
        """
        :param dlatents_in: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        :param step: current state of progressive growing
        :param alpha: fade in factor
        :return: images
        """

        blocks = list(self.blocks.values())
        to_rgbs = list(self.to_rgb.values())

        if self.structure == "fixed":
            output_ix = int(np.log2(self.resolution)) - 2

            for i, block in enumerate(blocks[:output_ix]):
                if i == 0:
                    out = self.input_block(dlatents_in[:, 0:2])

                out = block(out, dlatents_in[:, 2 * i:2 * i + 2])

            images_out = to_rgbs[-1](out)
            return images_out
        elif self.structure == "pro":
            y = self.input_block(dlatents_in[:, 0:2])

            if step > 0:
                for i, block in enumerate(blocks[:step]):

                    if i < step - 1:
                        y = block(y, dlatents_in[:, 2 * i:2 * i + 2])
                    else:
                        upsample = self.upscale(y)
                        residual = to_rgbs[step-1](upsample)
                        straight = to_rgbs[step](blocks[step - 1](y, dlatents_in[:, 2 * i:2 * i + 2]))
                        out = alpha * straight + (1 - alpha) * residual
            else:
                out = to_rgbs[0](y)
            return out
        else:
            raise Exception("Network structure {} not supported".format(self.structure))
