#!/bin/sh
python train_nonGAN.py 3
python train_nonGAN.py 5
python train_nonGAN.py 7
python train_nonGAN.py 9
python train_nonGAN.py 11
python train_nonGAN.py 13
# for i in {1..3}		# trains the models for 10*15 = 150 epochs
# do
# python train_nonGAN.py --continue
# done