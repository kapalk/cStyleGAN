#!/usr/bin/env bash

cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)
source "./venv/bin/activate"
cd -

version=1.0
lr=0.001
max_resolution=128
loss="wgan-gp"
style_mixing_prob=0.9
data="sc09"
batch_default=32
phase=200000
num_mels=128
training_iters=100000
latent_size=256
fmap_max=512
structure="pro"

python3 train.py --version $version \
                 --max_resolution $max_resolution \
                 --lr $lr \
                 --loss $loss \
                 --style_mixing_prob $style_mixing_prob \
                 --data $data \
                 --batch_default $batch_default \
                 --latent_size $latent_size \
                 --fmap_max $fmap_max \
                 --structure $structure \
                 --num_mels $num_mels \
                 --phase $phase \
                 --training_iters $training_iters \
		         --use_noise