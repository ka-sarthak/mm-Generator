import torch
import torch.nn as nn
import torch.nn.functional as F
from models.UNet import UNet
from utils.layers import PeriodicSeparableConv, SeparableConv
from utils.config_module import config

class Discriminator(nn.Module):
	"""
		PatchGAN network that evaluates pxp patch in the field as real or fake.
		It takes the input and the target image and returns a 2D probability field, where each pixel gives the probability that the 
		corresponding pxp patch is real.

		Warning -
		- works for default psize only.
		- works for 1 stress component (num_heads=1) only
	"""

	def __init__(self):
		super().__init__()
		self.version = config["model"]["GAN"]["discriminatorVersion"]
		num_heads = config["experiment"]["outputHeads"]
		
		if self.version == "UNet-enc":
			in_channels = config["experiment"]["inputHeads"] + num_heads
			enc_channels = config["model"]["UNet"]["encChannels"]
			kernel = config["model"]["UNet"]["kernel"]
			self.Enc0 =  GanDiscEncoding(channels=[    in_channels, enc_channels[0]],  kernel=kernel)
			self.Enc1 =  GanDiscEncoding(channels=[enc_channels[0], enc_channels[1]],  kernel=kernel)
			self.Enc2 =  GanDiscEncoding(channels=[enc_channels[1], enc_channels[2]],  kernel=kernel)
			self.Enc3 =  GanDiscEncoding(channels=[enc_channels[2], enc_channels[3]],  kernel=kernel)
			self.Enc4 =  GanDiscEncoding(channels=[enc_channels[3], 			  1],  kernel=1)
		
		elif self.version == "original":
		## Unlike original PatchGAN, we are not using padding. Therefore, the output fields are smaller than in the original paper.
			
			self.conv1 = nn.Conv2d(config["experiment"]["inputHeads"]+num_heads,64,4,stride=2,padding=0)
			self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=0)
			self.BN2 = nn.BatchNorm2d(128)
			self.conv3 = nn.Conv2d(128,256,4,stride=2,padding=0)
			self.BN3 = nn.BatchNorm2d(256)
			self.conv4 = nn.Conv2d(256,512,4,stride=1,padding=0)
			self.BN4 = nn.BatchNorm2d(512)
			self.conv5 = nn.Conv2d(512,1,4,stride=1,padding=0)
		
		elif self.version == "modified":
		## Convolution kernel size editted to accomodate 64x64 resolution fields
		## Also the convolutions are implemented as separable convolutions to reduce parameter space
			self.conv1 = SeparableConv(config["experiment"]["inputHeads"]+num_heads,64,3,stride=2)
			self.BN1 = nn.BatchNorm2d(64)
			self.conv2 = SeparableConv(64,128,3,stride=1)
			self.BN2 = nn.BatchNorm2d(128)
			self.conv3 = SeparableConv(128,256,3,stride=1)
			self.BN3 = nn.BatchNorm2d(256)
			self.conv4 = SeparableConv(256,512,3,stride=1)
			self.BN4 = nn.BatchNorm2d(512)
			self.conv5 = SeparableConv(512,1,3,stride=1)

		else:
			raise AssertionError("Unexpected argument for discriminator version.")

	def forward(self,inp,target):
		tmp = torch.cat((inp,target),dim=-3)		## (3x256x256),(5x256x256)-->(8x256x256)

		if self.version == "UNet-enc":
			tmp = self.Enc0(tmp)
			tmp = self.Enc1(tmp)
			tmp = self.Enc2(tmp)
			tmp = self.Enc3(tmp)
			tmp = torch.sigmoid(self.Enc4(tmp))
   
		elif self.version == "original":
			tmp = F.leaky_relu(self.conv1(tmp),0.2)
			tmp = F.leaky_relu(self.BN2(self.conv2(tmp)),0.2)
			tmp = F.leaky_relu(self.BN3(self.conv3(tmp)),0.2)
			tmp = F.leaky_relu(self.BN4(self.conv4(tmp)),0.2)
			tmp = F.sigmoid(self.conv5(tmp))
			
		elif self.version == "modified":
			tmp = F.leaky_relu(self.BN1(self.conv1(tmp)),0.2)
			tmp = F.leaky_relu(self.BN2(self.conv2(tmp)),0.2)
			tmp = F.leaky_relu(self.BN3(self.conv3(tmp)),0.2)
			tmp = F.leaky_relu(self.BN4(self.conv4(tmp)),0.2)
			tmp = torch.sigmoid(self.conv5(tmp))

		return tmp

class GanDiscEncoding(nn.Module):
	def __init__(self, channels, kernel):
		super().__init__()
		self.conv = PeriodicSeparableConv(channels[0], channels[1], kernel)
		self.bn = nn.BatchNorm2d(channels[1])
		self.maxPool = nn.MaxPool2d((2,2))
	
	def forward(self, x):
		x = self.conv(x)
		x = F.leaky_relu(self.bn(x),0.4)
		x = self.maxPool(x)
		return x