from c_style_gan.audio import AudioToMel
from tqdm import tqdm

import os
import argparse


def wav_to_mel(filepath, num_mels, out_filepath):
    a2m = AudioToMel(num_mels=num_mels)
    audio = a2m.load_wav(path=filepath)
    mel = a2m.melspectrogram(audio)
    if mel.shape[1] <= args.length:
        a2m.save_spectrogram(out_filepath, mel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Style GAN training')
    parser.add_argument('--in_dirpath', type=str, help='path to the directory that contains .wav files')
    parser.add_argument('--mels', type=int, help='the number of mel frequency bins')
    parser.add_argument('--length', type=int, help='the length of mel time axis')
    parser.add_argument('--out_dirpath', type=str, help='path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out_dirpath):
        os.mkdir(args.out_dirpath)

    wav_files = os.listdir(args.in_dirpath)
    for wf in tqdm(wav_files):
        in_filepath = os.path.join(args.in_dirpath, wf)
        out_filepath = os.path.join(args.out_dirpath, wf.replace(".wav", ".npy"))
        wav_to_mel(in_filepath, args.mels, out_filepath)
