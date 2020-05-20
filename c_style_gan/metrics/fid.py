from scipy import linalg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from c_style_gan.dataset import ToTensor, TimePadding
from torchvision import transforms
from tqdm import tqdm
from c_style_gan.model.discriminator import Discriminator
from c_style_gan.metrics.classifier import Classifier
import logging
import torch
import os
import argparse
import numpy as np


class Melspectrograms(Dataset):

    def __init__(self, mel_dir, num_mels):
        self.mel_dir = mel_dir
        self.num_mels = num_mels
        self.files = os.listdir(mel_dir)
        self.transforms = transforms.Compose([TimePadding(length=num_mels), ToTensor(dims=3)])

    def __getitem__(self, idx):
        mel = self._load_melspecgram(self.files[idx])
        mel = self.transforms(mel)
        mel = mel.type(torch.FloatTensor)
        return mel

    def __len__(self):
        return len(self.files)

    def _load_melspecgram(self, filename):
        return np.load(os.path.join(self.mel_dir, filename))


class FID:
    def __init__(self, num_images, batch_size, classifier, reals, mean_over):
        super().__init__()
        self.num_images = num_images
        self.batch_size = batch_size
        self.classifier = classifier
        self.reals = reals
        self.mean_over = mean_over
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_classifier(self):
        state = torch.load(self.classifier, map_location=lambda storage, loc: storage)
        clf = nn.DataParallel(
            Classifier(10)
            # Discriminator(num_channels=1,
            #               resolution=128,
            #               fmap_max=512,
            #               conv_dim=2,
            #               use_labels=False,
            #               mbstd_group_size=0,
            #               output_features=10
            #               )
        ).to(self.device)
        clf.load_state_dict(state["net"])
        clf.eval()
        return clf

    def _get_dataloader(self, dataset):
        return iter(DataLoader(dataset, self.batch_size, shuffle=True, drop_last=True))

    @torch.no_grad()
    def _calc_activations(self, classifier, data_iterator):
        activations = np.empty([self.num_images, classifier.module.channels], dtype=np.float32)
        for it in range(self.num_images // self.batch_size):
            images = next(data_iterator)
            begin = it * self.batch_size
            end = min(begin + self.batch_size, self.num_images)
            # _, features = classifier(images, labels=None, step=None, alpha=None, classify=True)
            _, features = classifier(images)
            activations[begin:end] = features.cpu()
            if end == self.num_images:
                break
        return activations

    @staticmethod
    def _get_state(path):
        state = torch.load(path, map_location=lambda storage, loc: storage)
        opt = None
        if state.get("args"):
            opt = state["args"]
        return state["G"], opt

    def _load_generator(self, state, options):
        if options is not None:
            from c_style_gan.model.generator import Generator
            G = Generator(
                latent_size=options.latent_size,
                dlatent_size=options.latent_size * 2,
                resolution=options.max_resolution,
                num_channels=1,
                fmap_max=options.fmap_max,
                style_mixing_prob=options.style_mixing_prob,
                conv_dim=options.conv_dim,
                embedding_dim=10,
                lr_mul=options.lr_mul,
                structure=options.structure,
                use_labels=options.use_labels,
                use_noise=options.use_noise
            )
        else:
            from c_style_gan.dcgan_model import Generator
            G = Generator()
        G = G.to(self.device)
        G.load_state_dict(state)

        return G

    @torch.no_grad()
    def _generate(self, gen, opt, batches=1):
        if opt:
            latent_size = opt.latent_size if opt.use_labels else opt.latent_size * 2
            res = opt.max_resolution
        else:
            latent_size = 100
            res = 128

        images = []
        label_list = []
        for _ in range(batches):
            labels = torch.arange(0, 10).view(10, -1).repeat(1, 1).to(self.device)
            latent = torch.randn(10, latent_size).to(self.device)
            imgs = gen(latent, labels, int(np.log2(res)) - 2, 1, False)
            images.append(imgs)
            label_list.append(labels)
        images = torch.stack(images).view(-1, imgs.size(1), imgs.size(2), imgs.size(3))
        return images

    @staticmethod
    def _calc_stats(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    @staticmethod
    def _calc_fid(mu_fake, mu_real, sigma_fake, sigma_real):
        m = np.square(mu_fake - mu_real).sum()
        s, _ = linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + sigma_real - 2 * s)
        return np.real(dist)

    def evaluate(self, generator_path):
        classifier = self._load_classifier()


        real_dataloader = self._get_dataloader(Melspectrograms(self.reals, 128))
        real_activations = self._calc_activations(classifier, real_dataloader)
        mu_real, sigma_real = self._calc_stats(real_activations)

        G_state, options = self._get_state(generator_path)
        G = self._load_generator(G_state, options)

        scores = []
        for _ in tqdm(range(self.mean_over)):
            generated_images = self._generate(G, options, self.num_images // 10)
            fake_dataloader = self._get_dataloader(generated_images)
            fake_activations = self._calc_activations(classifier, fake_dataloader)
            mu_fake, sigma_fake = self._calc_stats(fake_activations)

            scores.append(self._calc_fid(mu_fake, mu_real, sigma_fake, sigma_real))

        return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculates FID-score')
    parser.add_argument('--num_images', type=int, help='number of images used id calculation')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--classifier', type=str, help='path to the classifier')
    parser.add_argument('--generator', type=str, help='path to the trained generator')
    parser.add_argument('--real', type=str, help='path to real images')
    parser.add_argument('--n', type=int, default=1, help='n scores average')
    args = parser.parse_args()

    if "dcgan" in args.generator:
        from c_style_gan.dcgan_model import Generator, nz
    else:
        from c_style_gan.model.generator import Generator

    fid = FID(args.num_images, args.batch_size, args.classifier, args.real, args.n)
    fid_score, fid_sd = fid.evaluate(args.generator)

    logging.info(f"FID-score:{fid_score}, SD:{fid_sd}, N:{args.n}")
