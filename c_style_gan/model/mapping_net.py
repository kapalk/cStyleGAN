from torch import nn
from .utils import ACTIVATION_FUNCTIONS, PixelNormLayer, EqualizedLinear
import numpy as np
import torch


class MappingBlock(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            gain,
            use_wscale,
            lr_mul,
            activation
    ):
        """
        :param input_dim: Linear layer input dim
        :param output_dim: Linear layer output dim
        :param gain: Gain
        :param use_wscale: Use weight scaling?
        :param lr_mul: Learning rate multiplier
        :param activation: Activation function,
        """
        super(MappingBlock, self).__init__()
        self.linear = EqualizedLinear(
            in_features=input_dim,
            out_features=output_dim,
            gain=gain,
            use_wscale=use_wscale,
            lr_mul=lr_mul
        )
        self.activation = activation

    def forward(self, x, y):
        out = x
        if y is not None:
            out = torch.cat((out, y), dim=1)
        return self.activation(self.linear(out))


class MappingNet(nn.Module):

    def __init__(
            self,
            input_dim=512,
            output_dim=512,
            activation_func='lrelu',
            use_wscale=True,
            depth=8,
            lr_mul=0.01,
            normalize_input=True,
            gain=np.sqrt(2),
            broadcast_output=None,
            use_labels=True
    ):
        """
        Mapping network used in the StyleGAN paper
        :param input_dim:  Latent vector (Z) dimensionality.
        :param output_dim: Disentangled latent (W) dimensionality.
        :param depth: Number of hidden layers.
        :param lr_mul: Learning rate multiplier for the mapping layers.
        :param activation_func: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_input: Normalize latent vectors (Z) before feeding them to the mapping layers?
        """
        super(MappingNet, self).__init__()
        self.depth = depth
        self.normalize_input = normalize_input
        self.broadcast_output = broadcast_output

        layers = []

        if self.normalize_input:
            self.pixel_norm = PixelNormLayer()

        if use_labels:
            output_dim_1 = output_dim // 2
        else:
            output_dim_1 = output_dim

        nonlinearity = ACTIVATION_FUNCTIONS[activation_func]
        for layer_ix in range(self.depth):

            if layer_ix < self.depth - 1:
                layers.append(
                    MappingBlock(
                        input_dim,
                        output_dim_1,
                        gain,
                        use_wscale,
                        lr_mul,
                        nonlinearity
                    )
                )

            else:
                layers.append(
                    MappingBlock(
                        input_dim,
                        output_dim,
                        gain,
                        use_wscale,
                        lr_mul,
                        nonlinearity
                    )
                )

        self.layers = nn.Sequential(*layers)

    def forward(self, latent, embedding):
        if self.normalize_input:
            latent = self.pixel_norm(latent)
        for layer in self.layers:
            latent = layer(latent, embedding)
        if self.broadcast_output:
            dlatent = latent.unsqueeze(1).expand(-1, self.broadcast_output, -1)
        else:
            dlatent = latent
        return dlatent
