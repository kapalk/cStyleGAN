from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from c_style_gan.model.discriminator import Discriminator
import os
import torch
import numpy as np
import torch.nn as nn


from torch import nn
from c_style_gan.model.discriminator import View


class ClassifierBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ClassifierBlock, self).__init__()
        kernel_size = 5
        layers = list()
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Classifier(nn.Module):

    def __init__(self, n_logits, channels=128):
        super(Classifier, self).__init__()
        self._channels = 128
        self.input_block = ClassifierBlock(1, channels)

        blocks = [ClassifierBlock(channels, channels) for _ in range(4)]

        self.blocks = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        final = [
            View(channels),
            nn.BatchNorm1d(channels),
            nn.Linear(channels, n_logits),
            nn.LogSoftmax(dim=1)
        ]
        self.final_layers = nn.Sequential(*final)

    def forward(self, x):
        out = self.input_block(x)
        out = self.blocks(out)
        out = self.avg_pool(out)
        return self.final_layers(out), out.view(out.size(0), out.size(1))

    @property
    def channels(self):
        return self._channels


def get_dataloader(melpath, res, batch_size):
    return Trainer().spoken_digits_loader(mel_path=melpath, resolution=res, batch_size=batch_size, num_mels=res)


def accuracy(target, prediction):
    return (torch.eq(target.view(-1), prediction.max(1)[1]).sum().float() / torch.FloatTensor([target.size(0)])).item()

@torch.no_grad()
def validation(model, step):
    dataset = get_dataloader(meldir_valid, resolution, batch_size)
    loader = iter(dataset)
    iters = int((len(os.listdir(meldir_valid)) // batch_size))
    progress_bar = range(iters)
    model.eval()
    total_loss = []
    total_accuracy = []
    for _ in progress_bar:
        mels, labels = next(loader)
        mels.to(device)
        labels = labels.to(device)
        # prediction, _ = model(mels ,labels=None, step=None, alpha=None, classify=True)
        prediction, _ = model(mels)
        loss = criterion(prediction, labels.view(-1))
        total_loss.append(loss.item())
        total_accuracy.append(accuracy(labels, prediction))

    writer.add_scalar("Validation/Loss", np.mean(total_loss), step)
    writer.add_scalar("Validation/Accuracy", np.mean(total_accuracy), step)
    model.train()


def train():
    sample_size = len(os.listdir(meldir_train))
    iters = int((epochs * sample_size) // batch_size)
    progress_bar = tqdm(range(iters))

    dataset = get_dataloader(meldir_train, resolution, batch_size)
    loader = iter(dataset)

    clf = nn.DataParallel(
        Classifier(10)
    ).to(device)

    optimizer = optim.Adam(clf.parameters(), lr=0.001, betas=(0.0, 0.99), eps=1e-8)
    trained_samples = 0
    validation_step = 0
    for i in progress_bar:
        try:
            mels, labels = next(loader)
        except (OSError, StopIteration):
            loader = iter(dataset)
            mels, labels = next(loader)
        mels = mels.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        prediction, _ = clf(mels)
        loss = criterion(prediction, labels.view(-1))
        loss.backward()
        optimizer.step()

        trained_samples += mels.size(0)

        writer.add_scalar("Training/Loss", loss.item(), i + 1)
        writer.add_scalar("Training/Accuracy", accuracy(labels, prediction), i + 1)

        if trained_samples % 2000 == 0:
            validation(clf, validation_step)
            state = {"net": clf.state_dict()}
            torch.save(state, f"classifier_{trained_samples}.pkl")
            validation_step += 1

        state_msg = ('loss: {}'.format(np.round(loss.item(), 3)))
        progress_bar.set_description(state_msg)
    state = {"net": clf.state_dict()}
    torch.save(state, "classifier.pkl")

    writer.close()


if __name__ == "__main__":
    from c_style_gan.train import Trainer
    epochs = 50
    batch_size = 8
    resolution = 128
    meldir_train = "../data/sc09/train/melspec_128/"
    meldir_valid = "../data/sc09/valid/melspec_128/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.NLLLoss(reduction="sum")

    writer = SummaryWriter(log_dir="../logs/{}".format("classifier"))

    train()

    writer.close()



