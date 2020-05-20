import torch
import os
import numpy as np
import argparse
import logging

from tqdm import tqdm
from c_style_gan.model.generator import Generator
from c_style_gan.train import G_NAME
from c_style_gan.audio import MelToAudio


def load_generator(options, gen_state, device):
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
     ).to(device)
    G.load_state_dict(gen_state)
    return G


@torch.no_grad()
def generate(gen, latent_dim, image_res, batches, device):
    images = []
    logging.info("Generating images...")
    label_list = []
    for _ in tqdm(range(batches)):
        labels = torch.arange(0, 10).view(10, -1).repeat(1, 1).to(device)
        latent = torch.randn(10, latent_dim).to(device)
        with torch.no_grad():
            imgs = gen(
                latent,
                labels,
                step=int(np.log2(image_res)) - 2,
                alpha=1,
                training=False
            )
            images.append(imgs)
            label_list.append(labels)
    images = torch.stack(images).view(-1, imgs.size(1), imgs.size(2), imgs.size(3))
    return images.squeeze(1), torch.stack(label_list).view(-1, 1)


def _get_sample_idx(generated_path):
    num_samples = len(os.listdir(generated_path))
    return num_samples + 1


def save_images(images, labels, path):
    logging.info("Saving generated images...")
    if not os.path.exists(path):
        os.mkdir(path)

    for img, label in tqdm(zip(images, labels)):
        idx = str(_get_sample_idx(path))
        np.save(os.path.join(path, ".".join([str(label.item()), idx, "npy"])), img.cpu())


def mels_to_audio(images, labels, path):
    m2a = MelToAudio(num_mels=images.size(1))
    if not os.path.exists(path):
        os.mkdir(path)

    logging.info("To audio...")
    for img, label in tqdm(zip(images, labels)):
        audio = m2a.inv_melspectrogram(img.cpu().data.numpy())
        idx = str(_get_sample_idx(path))
        m2a.save_wav(audio, os.path.join(path, ".".join([str(label.item()), idx, "wav"])))


def main():
    parser = argparse.ArgumentParser(description='Style GAN generate')
    parser.add_argument('--path', type=str, help='relative path to the model')
    parser.add_argument('--generated', type=str, help='path to save the generated samples')
    parser.add_argument('--dataset', default='sc09', type=str, help='data used in training')
    parser.add_argument('--batches', default=1, type=int)
    parser.add_argument('--mix', action="store_true", default=False, help="mix styles")

    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, args.path)
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    opt = state["args"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = load_generator(opt, state[G_NAME], device)
    G.eval()

    if opt.use_labels:
        latent_size = opt.latent_size
    else:
        latent_size = opt.latent_size * 2

    mels, labels = generate(G, latent_size, opt.max_resolution, args.batches, device)

    if args.generated:
        path = os.path.join(args.generated, str(opt.version))
        if not os.path.exists(path):
            os.mkdir(path)
        save_images(mels, labels, os.path.join(path, "mels"))

        mels_to_audio(mels, labels, os.path.join(path, "wavs"))




if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()

