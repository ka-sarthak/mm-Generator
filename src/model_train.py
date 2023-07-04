from utils.config_module import config
from train_module import train_nonGAN

def train():
    if config["experiment"]["model"] == "GAN":
        train_GAN.train()
    elif config["experiment"]["model"] == "nonGAN":
        train_nonGAN.train()
    else:
        raise AssertionError("Unexpected model. Should be GAN or nonGAN.")