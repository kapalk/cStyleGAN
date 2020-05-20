from torch import nn
from c_style_gan.model.mapping_net import MappingNet
from c_style_gan.model.synthesis_net import SynthesisNet

import torch
import numpy as np


class Truncation(nn.Module):

    def __init__(self, max_layer=8, threshold=0.7):
        super(Truncation, self).__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        # self.register_buffer('avg_latent', avg_latent)

    def forward(self, x, avg_dlatents):
        assert x.dim() == 3
        interp = torch.lerp(avg_dlatents, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)

    def extra_repr(self):
        return 'max_layer={max_layer}, threshold={threshold}'.format(
            max_layer=self.max_layer,
            threshold=self.threshold
        )


class Generator(nn.Module):

    def __init__(
            self,
            latent_size=512,
            dlatent_size=512,
            resolution=1024,
            num_channels=3,
            truncation_psi=0.7,
            truncation_cutoff=8,
            dlatent_avg_beta=0.995,
            style_mixing_prob=0.9,
            fmap_max=512,
            conv_dim=2,
            embedding_dim=10,
            lr_mul=0.01,
            structure="fixed",
            use_labels=True,
            use_noise=True
    ):
        """
        :param latent_size: latent vectors (Z) [B, latent_size]
        :param dlatent_size: dlatent vectors W [B, dlatent_size]
        :param resolution: output resolution
        :param num_channels: number of channels in generated image
        :param truncation_psi: strength multiplier for the truncation trick
        :param truncation_cutoff: Number of layers for which to apply the truncation trick
        :param dlatent_avg_beta: Decay for tracking the moving average of W during training
        :param style_mixing_prob: Probability of mixing styles during training
        :param fmap_max: max number of channels
        :param conv_dim: dimension of convolutional layres => 1 or 2
        :param embedding_dim: the dimensionality of embeddings
        :param lr_mul: learning rate multiplier
        :param structure: structure of the forward pass: fixed or progressive
        :param use_labels: concatenate labels in mapping net?
        :param use_noise: inject noise?
        """
        super(Generator, self).__init__()
        num_layers = int(np.log2(resolution)-1) * 2
        self.dlatent_avg_beta = dlatent_avg_beta
        self.register_buffer('dlatent_avg', torch.zeros(num_layers, dlatent_size))
        self.style_mixing_prob = style_mixing_prob
        self.num_layers = int(np.log2(resolution) * 2 - 2)
        self.use_labels = use_labels

        self.encoder = None
        if use_labels:
            self.encoder = nn.Sequential(
                nn.Embedding(10, embedding_dim=embedding_dim),
                nn.Linear(embedding_dim, latent_size)
            )

        self.mapping = MappingNet(
            input_dim=latent_size * 2,
            output_dim=dlatent_size,
            broadcast_output=num_layers,
            lr_mul=lr_mul,
            use_labels=use_labels
        )

        if truncation_psi and truncation_cutoff:
            self.truncation = Truncation(max_layer=truncation_cutoff, threshold=truncation_psi)
        self.synthesis = SynthesisNet(
            dlatent_size=dlatent_size,
            num_channels=num_channels,
            resolution=resolution,
            fmap_max=fmap_max,
            conv_dim=conv_dim,
            structure=structure,
            use_noise=use_noise
        )

    def forward(self, latents, labels, step=8, alpha=0.5, training=True):
        """
        :param latents: latent vectors Z  [Z, latent_size]
        :param labels: image labels [B, 1]
        :param step:  step of progressive growing
        :param alpha: fade in params for a new layer in progressive growing
        :param training: training?
        :return: images [B, C , D, D] where D is power of 2
        """
        # Encoder
        embeddings = None
        if self.encoder is not None:
            embeddings = self.encoder(labels).view(labels.size(0), -1)
        dlatents = self.mapping(latents, embeddings)

        # Update moving average of W.
        if training and self.dlatent_avg_beta:
            batch_avg = dlatents.mean(0)
            dlatent_avg = self.dlatent_avg
            self.dlatent_avg.data = batch_avg * self.dlatent_avg_beta + (1 - self.dlatent_avg_beta) * dlatent_avg

        # Style mixing regularization
        if training and self.style_mixing_prob:
            with torch.no_grad():
                latents2 = torch.randn(latents.shape).to(latents.device)
                dlatents2 = self.mapping(latents2, embeddings)
                current_layers = (step + 1) * 2
                mixing_cutoff = current_layers
                if torch.randn(size=()) < self.style_mixing_prob:
                    mixing_cutoff = torch.randint(1, current_layers, ()).item()
                for i in range(self.num_layers):
                    if i < mixing_cutoff:
                        dlatents[:, i, :] = dlatents2[:, i, :]

        # Truncation trick
        if not training:
            self.truncation(dlatents, self.dlatent_avg)

        return self.synthesis(dlatents, step, alpha)
