from torch import nn
from .utils import ACTIVATION_FUNCTIONS, BlurLayer, Downscale2D, EqualizedLinear, EqualizedConv, FusedDownsample
from collections import OrderedDict

import numpy as np
import torch

class MinibatchStdDevLayer(nn.Module):
    """Minibatch standard deviation"""
    def __init__(self, batch_size=4, num_new_features=1):
        super(MinibatchStdDevLayer, self).__init__()
        self.batch_size = batch_size
        self.num_new_features = num_new_features

    def forward(self, x):
        group_size = min(self.batch_size, x.shape[0])

        def stddev(z):
            z = z - z.mean(0, keepdim=True)
            z = (z ** 2).mean(0, keepdim=True)
            z = (z + 1e-8) ** 0.5
            return z

        if x.dim() == 3:
            batch_size, channels, length = x.shape
            y = x.reshape([group_size, -1, self.num_new_features, channels // self.num_new_features, length])
            y = stddev(y)
            y = y.mean([3, 4], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
            y = y.expand(group_size, -1, -1,length).clone().reshape(batch_size, self.num_new_features, length)
        else:
            batch_size, channels, height, width = x.shape
            y = x.reshape([group_size, -1, self.num_new_features, channels // self.num_new_features, height, width])
            y = stddev(y)
            y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
            y = y.expand(group_size, -1, -1, height, width).clone().reshape(
                batch_size, self.num_new_features, height, width
            )
        return torch.cat([x, y], dim=1)


class DiscriminatorBlock(nn.Module):
    """The core layers of discriminator"""
    def __init__(
            self,
            input_channels,
            output_channels1,
            output_channels2,
            kernel_size,
            gain,
            use_wscale,
            blur_filter,
            activation,
            conv_dim,
            use_labels
    ):
        super(DiscriminatorBlock, self).__init__()
        layers = []
        input_channels_1 = input_channels

        if use_labels:
            input_channels_1 = input_channels + 1

        layers.append(
            EqualizedConv(
                in_channels=input_channels_1,
                out_channels=output_channels1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                gain=gain,
                use_wscale=use_wscale,
                dimension=conv_dim
            )
        )

        layers.append(activation)

        if blur_filter:
            layers.append(BlurLayer(dimension=conv_dim, kernel=blur_filter))

        if input_channels >= 128:
            layers.append(
                FusedDownsample(
                    in_channel=input_channels,
                    out_channel=output_channels2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
        else:
            layers.append(
                EqualizedConv(
                    in_channels=input_channels,
                    out_channels=output_channels2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    gain=gain,
                    use_wscale=use_wscale,
                    dimension=conv_dim
                )
            )

            layers.append(Downscale2D(dimension=conv_dim))

        layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FromRGB(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            activation,
            conv_dim,
            kernel_size=1,
            gain=np.sqrt(2),
            use_wscale=True
    ):
        super(FromRGB, self).__init__()
        self.conv = EqualizedConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            gain=gain,
            use_wscale=use_wscale,
            dimension=conv_dim
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class FinalBlock(nn.Module):
    """final layers of discriminator"""
    def __init__(
            self,
            input_channels,
            intermediate_channels,
            kernel_size,
            gain,
            use_wscale,
            minibatch_size,
            minibatch_std_num_features,
            activation,
            conv_dim,
            resolution=4,
            input_channels_2=None,
            output_features=1
    ):
        super(FinalBlock, self).__init__()
        layers = []
        in_channels = input_channels

        if minibatch_size > 1:
            layers.append(
                MinibatchStdDevLayer(
                    batch_size=minibatch_size,
                    num_new_features=minibatch_std_num_features
                )
            )
            in_channels = input_channels + minibatch_std_num_features

        if input_channels_2 is None:
            input_channels_2 = input_channels

        layers.append(
            EqualizedConv(
                in_channels=in_channels,
                out_channels=input_channels_2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                gain=gain,
                use_wscale=use_wscale,
                dimension=conv_dim
            )
        )

        layers.append(activation)
        layers.append(View(-1))
        in_features = input_channels_2 * resolution ** 2

        if conv_dim == 1:
            in_features = (input_channels_2 * resolution * resolution)

        layers.append(
            EqualizedLinear(
                in_features=in_features,
                out_features=intermediate_channels,
                gain=gain,
                use_wscale=use_wscale
            )
        )

        layers.append(activation)
        layers.append(
            EqualizedLinear(
                in_features=intermediate_channels,
                out_features=output_features,
                gain=gain,
                use_wscale=use_wscale
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for ix, mod in enumerate(self.layers):
            x = mod(x)
        return x


class Discriminator(nn.Module):
    """ Discriminator of Style GAN """
    def __init__(
            self,
            num_channels=3,
            resolution=1024,
            fmap_base=8192,
            fmap_decay=1.,
            fmap_max=512,
            activation_func='lrelu',
            use_wscale=True,
            mbstd_group_size=4,
            mbstd_num_features=1,
            blur_filter=(1, 2, 1),
            conv_dim=2,
            structure="fixed",
            embedding_size=10,
            use_labels=True,
            output_features=1
    ):
        """
        :param labels_in: Second input: Labels [minibatch, label_size].
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param activation_func: Activation function: 'relu', 'lrelu',
        :param use_wscale: Enable equalized learning rate
        :param mbstd_group_size: Group size for the minibatch standard deviation layer, 0 = disable.
        :param mdstd_num_features: Number of features for the minibatch standard deviation layer.
        :param fused_scale: True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param conv_dim: Dimensionality of convolutional layers: 1 or 2.
        :param structure: Structure of forward pass: fixed or pro
        """
        super(Discriminator, self).__init__()

        self.structure = structure
        self.conv_dim = conv_dim
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        kernel_size = 3

        self.activation, gain = ACTIVATION_FUNCTIONS[activation_func], np.sqrt(2)
        blocks = []
        from_rgb = list()
        embedding_scaling_layers = list()

        self.embedding = None
        if use_labels:
            self.embedding = nn.Embedding(10, embedding_dim=embedding_size)

        for res in range(2, self.resolution_log2 + 1):
            s = 2 ** res
            name = f'{s}x{s}'

            if res > 2:
                blocks.append(
                    (name,
                     DiscriminatorBlock(
                         input_channels=nf(res - 1),
                         output_channels1=nf(res - 1),
                         output_channels2=nf(res - 2),
                         kernel_size=kernel_size,
                         gain=gain,
                         use_wscale=use_wscale,
                         blur_filter=blur_filter,
                         activation=self.activation,
                         conv_dim=self.conv_dim,
                         use_labels=use_labels
                     )
                     )
                )

            embedding_scaling_layers.append((name, nn.Linear(embedding_size, s ** 2)))

            from_rgb.append(
                (
                    f'from RGB {name}',
                    FromRGB(
                        num_channels, nf(res - 1),
                        activation=self.activation,
                        kernel_size=1,
                        gain=gain,
                        use_wscale=use_wscale,
                        conv_dim=self.conv_dim
                    )
                )
            )

        if use_labels:
            self.embedding_scaling_layers = nn.ModuleDict(OrderedDict(embedding_scaling_layers))

        self.downscale = Downscale2D(dimension=self.conv_dim)
        self.from_rgb = nn.ModuleDict(OrderedDict(from_rgb))
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

        input_res = 4

        self.final_block = FinalBlock(
            input_channels=nf(2),
            intermediate_channels=nf(2),
            kernel_size=kernel_size,
            gain=gain,
            use_wscale=use_wscale,
            minibatch_size=mbstd_group_size,
            minibatch_std_num_features=mbstd_num_features,
            resolution=input_res,
            activation=self.activation,
            conv_dim=self.conv_dim,
            output_features=output_features
        )

    def forward(self, images, labels, step, alpha, classify=False):
        """
        :param images: input: Images [B, C, H, W].
        :param images: labels of Images [B, 1]
        :param step: current state of progressive growing
        :param alpha: fade in factor
        :return: prediction values
        """

        blocks = list(self.blocks.values())
        from_rbgs = list(self.from_rgb.values())

        if self.embedding is not None:
            embedding_layers = list(self.embedding_scaling_layers.values())
            embeddings_og = self.embedding(labels)

        if self.structure == "fixed":
            output_ix = int(np.log2(self.resolution)) - 2
            out = from_rbgs[-1](images)

            for i, block in enumerate(reversed(blocks[:output_ix])):
                if self.embedding is not None:
                    e_ix = output_ix - i
                    embeddings = embedding_layers[e_ix](embeddings_og).view(
                        embeddings_og.size(0), 1,  out.size(2), out.size(3)
                    )
                    out = torch.cat((out, embeddings), dim=1)

                out = block(out)

            out = self.final_block(out)
            return out

        elif self.structure == "pro":

            if step > 0:
                residual = from_rbgs[step - 1](self.downscale(images))
                rgb = from_rbgs[step](images)

                if self.embedding is not None:
                    embeddings = embedding_layers[step](embeddings_og).view(
                        embeddings_og.size(0), 1, rgb.size(2), rgb.size(3)
                    )

                    rgb = torch.cat((rgb, embeddings), dim=1)

                straight = blocks[step - 1](rgb)

                out = alpha * straight + (1 - alpha) * residual

                for i, block in enumerate(reversed(blocks[:step - 1])):
                    if self.embedding is not None:
                        embeddings = embedding_layers[step - 1 - i](embeddings_og).view(
                            embeddings_og.size(0), 1, out.size(2), out.size(3)
                        )
                        out = torch.cat((out, embeddings), dim=1)

                    out = block(out)

                out = self.final_block(out)
            return out
        else:
            raise Exception("Network structure: {} not supported".format(self.structure))

