import torch.nn as nn
from models.FNO import FNO
from models.UNet import UNet
from utils.config_module import config

class Generator(nn.Module):
	"""
		Picks up the network from the generator models: FNO, UNet, ...
		also acts as a wrapper function
	"""
	def __init__(self):
		super().__init__()
		if config["experiment"]["generator"] == "FNO":
			self.network = FNO()
		elif config["experiment"]["generator"] == "UNet":
			self.network = UNet()
		else:
			raise AssertionError("Unexpected argument for generator model.")

	def forward(self,x):
		x = self.network(x)
		return x
