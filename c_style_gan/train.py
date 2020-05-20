from torch import nn
from c_style_gan.model.discriminator import Discriminator
from c_style_gan.model.generator import Generator
from c_style_gan.dataset import ToTensor, TimePadding, NormalizeTensor, SpokenDigits, ResizeTensor
from torchvision import datasets
from torch import optim
from tqdm import tqdm
from c_style_gan.audio import MelToAudio
from torch.utils.data import DataLoader
from c_style_gan.setup_logger import logger
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.autograd import grad
from c_style_gan.metrics.fid import FID

import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse

MANUAL_SEED = 10
G_NAME = "G"
D_NAME = "D"
OPTIMIZER_G_NAME = "optimizer_G"
OPTIMIZER_D_NAME = "optimizer_D"
STEP_NAME = "step"
ALPHA_NAME = "alpha"
USED_SAMPLES_NAME = "used_samples"
ITERS = "iters"
ADVERSARIAL = "Adversarial_Loss"


class Trainer:

    @staticmethod
    def mnist_loader(path, resolution, batch_size):
        transforms.ToTensor()
        compose = [transforms.Resize(resolution), ToTensor(dims=args.conv_dim + 1)]
        if args.conv_dim == 2:
            compose.append(transforms.Normalize([0.5], [0.5]))
        elif args.conv_dim == 1:
            compose.append(NormalizeTensor(0.5, 0.5))
        transform = transforms.Compose(compose)
        dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader

    @staticmethod
    def spoken_digits_loader(mel_path, resolution, batch_size, num_mels):
        composed = transforms.Compose(
            [
                TimePadding(length=resolution),
                ToTensor(dims=3),
                ResizeTensor(resolution)
            ]
        )
        dataset = SpokenDigits(mel_dir=mel_path, num_mels=num_mels, transform=composed)
        return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    @staticmethod
    def _save_model(filename, step, alpha, used_samples, training_iters_done):
        cur_state = {
            G_NAME: G.module.state_dict(),
            D_NAME: D.module.state_dict(),
            OPTIMIZER_G_NAME: optimizer_G.state_dict(),
            OPTIMIZER_D_NAME: optimizer_D.state_dict(),
            STEP_NAME: step,
            ALPHA_NAME: alpha,
            USED_SAMPLES_NAME: used_samples,
            ITERS: training_iters_done,
            'args': args
        }
        torch.save(cur_state, filename)

    @staticmethod
    def mels2sound(mels, resolution):
        audios = []
        m2a = MelToAudio(num_mels=resolution)
        for mel in mels:
            mel = mel.squeeze(0)  # C x H x W => H x W
            audio = torch.from_numpy(m2a.inv_melspectrogram(mel.cpu().data.numpy()))
            audios.append(audio)
        return audios

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    @staticmethod
    def get_resolution(step):
        if args.structure == "fixed":
            return args.max_resolution
        return 4 * 2 ** step

    @staticmethod
    def get_batch_size(resolution):
        if args.structure == "fixed":
            return args.batch_default
        return args.batch_schedule.get(resolution, args.batch_default)

    @staticmethod
    def get_lr(resolution):
        return args.lr_schedule.get(resolution, args.lr)

    @staticmethod
    def adjust_lr(optimizer, lr):
        for group in optimizer.param_groups:
            mult = group.get('mult', 1)
            group['lr'] = lr * mult

    def train(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(cur_dir, "trained_models")
        training_dir = os.path.join(cur_dir, "training")
        checkpoint_dir = os.path.join(training_dir, 'checkpoint', str(args.version))

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(training_dir):
            os.mkdir(training_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        if args.use_labels:
            latent_size = args.latent_size
        else:
            latent_size = args.latent_size * 2

        fixed_latent = torch.randn(10, latent_size, device=device)
        fixed_labels = torch.arange(10).view(10, 1).to(device)  # digits

        training_samples = 0
        used_sample = 0
        alpha = 1
        step = 1
        step_max = int(np.log2(args.max_resolution)) - 2
        if args.structure == "fixed":
            step = step_max

        checkpoint_training_iters_done = 0
        if args.checkpoint:
            alpha = state[ALPHA_NAME]
            step = state[STEP_NAME]
            used_sample = state[USED_SAMPLES_NAME]
            checkpoint_training_iters_done = state[ITERS]

        resolution = self.get_resolution(step)
        batch_size = self.get_batch_size(resolution)
        if args.data == "mnist":
            mnist_path = "./data/mnist"
            dataset = self.mnist_loader(mnist_path, resolution, batch_size)
            sample_size = 30000
        elif args.data == "sc09":
            melspec_path = "./data/sc09/train/melspec_{num_mels}".format(num_mels=args.num_mels)
            dataset = self.spoken_digits_loader(melspec_path, resolution, batch_size, args.num_mels)
            sample_size = len(os.listdir(melspec_path))
        else:
            raise Exception("Check your dataset name")
        loader = iter(dataset)
        logger.info("Training sample size: {}".format(sample_size))

        self.adjust_lr(optimizer_G, self.get_lr(resolution))
        self.adjust_lr(optimizer_D, self.get_lr(resolution))

        progress_bar = tqdm(range(checkpoint_training_iters_done, args.training_iters))
        final_layer = False

        fid = FID(18620, 133, "./metrics/classifier.pkl", f"./data/sc09/train/melspec_{args.num_mels}", 1)
        min_fid = None

        for i in progress_bar:
            training_iters_done = i + 1

            if args.structure == "pro":
                alpha = min(1, 1 / args.phase * (used_sample + 1))

                if step == 1 or final_layer:
                    alpha = 1

                if used_sample > args.phase * 2:
                    step += 1

                    if step > step_max:
                        step = step_max
                        final_layer = True
                    else:
                        alpha = 0
                        used_sample = 0
                        resolution = self.get_resolution(step)
                        batch_size = self.get_batch_size(resolution)

                        if args.data == "mnist":
                            dataset = self.mnist_loader(mnist_path, resolution, batch_size)
                        elif args.data == "sc09":
                            dataset = self.spoken_digits_loader(melspec_path, resolution, batch_size, args.num_mels)
                        loader = iter(dataset)

                    self.adjust_lr(optimizer_G, self.get_lr(resolution))
                    self.adjust_lr(optimizer_D, self.get_lr(resolution))

            try:
                real_image, real_labels = next(loader)
            except (OSError, StopIteration):
                loader = iter(dataset)
                real_image, real_labels = next(loader)

            real_image = real_image.to(device)
            real_labels = real_labels.to(device)

            # Discriminator adversarial loss
            D.zero_grad()

            latent1, latent2 = torch.randn(2, batch_size, latent_size, device=device).chunk(2, 0)
            latent1 = latent1.squeeze(0)
            latent2 = latent2.squeeze(0)
            fake_image = G(latent1, real_labels, step=step, alpha=alpha)
            fake_predict = D(fake_image, real_labels, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                real_predict = D(real_image, real_labels, step=step, alpha=alpha)
                real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
                (-real_predict).backward()

            elif args.loss == 'logistic':
                real_image.requires_grad = True
                real_scores = D(real_image, real_labels, step=step, alpha=alpha)
                real_predict = F.softplus(-real_scores).mean()
                real_predict.backward(retain_graph=True)

                grad_real = grad(
                    outputs=real_scores.sum(), inputs=real_image, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()
                grad_penalty = 10 / 2 * grad_penalty
                grad_penalty.backward()
                grad_loss_val = grad_penalty.item()

            if args.loss == 'wgan-gp':
                fake_predict = fake_predict.mean()
                fake_predict.backward()

                eps = torch.rand(batch_size, 1, 1, 1).to(device)
                x_hat = eps * real_image.data + (1 - eps) * fake_image.data
                x_hat.requires_grad = True
                hat_predict = D(x_hat, real_labels, step=step, alpha=alpha)
                grad_x_hat = grad(
                    outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
                )[0]
                grad_penalty = (
                        (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
                ).mean()
                grad_penalty = 10 * grad_penalty
                grad_penalty.backward()
                grad_loss_val = grad_penalty.item()
                D_loss = (real_predict - fake_predict)

            elif args.loss == 'logistic':
                fake_predict = F.softplus(fake_predict).mean()
                fake_predict.backward()
                D_loss = (real_predict + fake_predict)

            writer.add_scalar("Loss/Discriminator_" + ADVERSARIAL, D_loss.item(), i + 1)
            writer.add_scalar("Loss/Gradient", grad_loss_val, i + 1)

            optimizer_D.step()

            if training_iters_done % args.critic == 0:
                # Generator loss and update
                G.zero_grad()

                self.requires_grad(G, True)
                self.requires_grad(D, False)

                fake_image = G(latent2, real_labels, step=step, alpha=alpha)

                predict = D(fake_image, real_labels, step=step, alpha=alpha)

                if args.loss == 'wgan-gp':
                    loss = -predict.mean()

                elif args.loss == 'logistic':
                    loss = F.softplus(-predict).mean()

                G_loss = loss
                writer.add_scalar("Loss/Generator_" + ADVERSARIAL, G_loss.item(), i + 1)

                loss.backward()
                optimizer_G.step()

                self.requires_grad(G, False)
                self.requires_grad(D, True)

            if training_iters_done % args.gen_sample_iter == 0:
                images = G(fixed_latent, fixed_labels, step, alpha, training=False)
                grid = torchvision.utils.make_grid(images, nrow=2)
                writer.add_image('Images/step_{}'.format(training_iters_done), grid, training_iters_done)

                if args.data != "mnist":
                    audios = self.mels2sound(images, resolution)
                    for digit, audio in enumerate(audios):
                        writer.add_audio(
                            'audio/step_{}_digit_{}'.format(training_iters_done, digit),
                            audio,
                            training_iters_done,
                            sample_rate=16000
                        )

            # save checkpoint or final model and calulcate fid
            if training_iters_done % args.save_checkpoint == 0 or training_iters_done == args.training_iters:
                destination_dir = os.path.join(checkpoint_dir, args.data)
                if training_iters_done == args.training_iters:
                    destination_dir = os.path.join(model_dir, args.data)
                if not os.path.exists(destination_dir):
                    os.mkdir(destination_dir)

                model_path = os.path.join(destination_dir,
                                          '{version}_{runs}.pkl'.format(
                                              version=args.version, runs=str(training_iters_done).zfill(6)
                                          )
                                          )

                self._save_model(model_path,
                                 step,
                                 alpha,
                                 used_sample,
                                 training_iters_done
                )

                if step == step_max:
                    fid_score, _ = fid.evaluate(model_path)

                    if min_fid is None or min_fid > fid_score:
                        mid_fid = fid_score
                        writer.add_scalar("Training/Min FID", mid_fid)

                    writer.add_scalar("Training/FID", fid_score, training_iters_done)

            training_samples += real_image.size(0)

            writer.add_scalar("Training/Alpha", alpha, training_iters_done)
            writer.add_scalar("Training/Size", resolution, training_iters_done)
            writer.add_scalar("Training/BatchSize", self.get_batch_size(resolution), training_iters_done)
            writer.add_scalar("Training/lr", self.get_lr(resolution), training_iters_done)
            writer.add_scalar("Training/Samples", training_samples, training_iters_done)


            msg_str = 'Res: {resolution}; ' \
                      'Alpha: {alpha}; ' \
                      'Batch {batch_size}; ' \
                      'G: {gen_loss_val:.3f}; ' \
                      'D: {disc_loss_val:.3f}; ' \
                      'Grad: {grad_loss_val:.3f};'
            state_msg = (
                msg_str.format(
                    gen_loss_val=round(G_loss.item(), 3),
                    disc_loss_val=round(D_loss.item(), 3),
                    resolution=resolution,
                    alpha=round(alpha, 3),
                    batch_size=batch_size,
                    grad_loss_val=round(grad_loss_val, 3)
                )
            )
            progress_bar.set_description(state_msg)
            used_sample += real_image.size(0)


if __name__ == "__main__":
    embedding_dim = 10
    torch.manual_seed(MANUAL_SEED)

    parser = argparse.ArgumentParser(description='Style GAN training')
    parser.add_argument('--version', type=float, help='network version')
    parser.add_argument('--data', default='sc09', type=str, help='dataset')
    parser.add_argument('--training_iters', default=1000000, type=int, help='dataset')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_default', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='# epochs')
    parser.add_argument('--phase', default=200000, type=int, help='learning phase for progressive growing')
    parser.add_argument('--critic', default=1, type=int, help='critic')
    parser.add_argument('--max_resolution', default=128, type=int, help='image resolution')
    parser.add_argument('--latent_size', default=64, type=int, help='latent vector size')
    parser.add_argument('--style_mixing_prob', default=None, type=float, help='style mixing probability')
    parser.add_argument('--loss', default='logistic', type=str, help='loss function')
    parser.add_argument('--loss_scaling', action='store_true', help='use loss scaling')
    parser.add_argument('--r1', default=0.0, type=float, help='R1 regularization gamma param for logistic loss')
    parser.add_argument('--r2', default=0.0, type=float, help='R2 regularization gamma param for logistic loss')
    parser.add_argument('--fmap_max', default=64, type=int, help='channel max')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint file')
    parser.add_argument('--save_checkpoint', default=1000, type=int, help='save checkpoint after # iteration')
    parser.add_argument('--num_mels', type=int, help='the number of frequency bins in mels')
    parser.add_argument('--gen_sample_iter', default=1000, type=int, help='generate sample output')
    parser.add_argument('--lr_mul', default=0.01, type=float, help='lr multiplier for mapping net')
    parser.add_argument('--structure', default="pro", type=str, help='progressive or fixed structure')
    parser.add_argument('--use_labels', action="store_true", help='condition on labels')
    parser.add_argument('--use_noise', action="store_true", help='inject noise to generator blocks labels')
    args = parser.parse_args()

    args.batch_schedule = {8: 256, 16: 128, 32: 64, 64: 32, 128: 32}
    args.lr_schedule = {128: 0.0015}

    num_channels = 1
    logger.info("args: {}".format(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("running on {}".format(device))

    G = nn.DataParallel(
        Generator(
            latent_size=args.latent_size,
            dlatent_size=args.latent_size * 2,
            resolution=args.max_resolution,
            num_channels=num_channels,
            fmap_max=args.fmap_max,
            style_mixing_prob=args.style_mixing_prob,
            embedding_dim=embedding_dim,
            lr_mul=args.lr_mul,
            structure=args.structure,
            use_labels=args.use_labels,
            use_noise=args.use_noise
        )
    ).to(device)

    D = nn.DataParallel(
        Discriminator(
            num_channels=num_channels,
            resolution=args.max_resolution,
            fmap_max=args.fmap_max,
            structure=args.structure,
            embedding_size=embedding_dim,
            use_labels=args.use_labels
        )
    ).to(device)

    mapping_net_params = [kv[1] for kv in G.named_parameters() if "mapping" in kv[0]]
    other_gen_params = [kv[1] for kv in G.named_parameters() if "mapping" not in kv[0]]
    optimizer_G = optim.Adam(
        [
            {'params': other_gen_params},
            {'params': mapping_net_params, 'lr': args.lr, 'mult': args.lr_mul}
        ],
        lr=args.lr, betas=(0.0, 0.99), eps=1e-8
    )

    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.99), eps=1e-8)

    trainer = Trainer()

    if args.checkpoint:
        state = torch.load(args.checkpoint)
        G.module.load_state_dict(state[G_NAME])
        D.module.load_state_dict(state[D_NAME])
        optimizer_G.load_state_dict(state[OPTIMIZER_G_NAME])
        optimizer_D.load_state_dict(state[OPTIMIZER_D_NAME])

    writer = SummaryWriter(log_dir="./logs/{}".format(args.version))
    trainer.train()
    writer.close()
