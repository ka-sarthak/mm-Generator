from generator_models.FNO import FNO
# from utilities import generator_loss, discriminator_loss

import torch.nn as nn

class generator(nn.Module):
	"""
		Picks up the network from the generator models: FNO, UNet, ...
	"""
	def __init__(self, modes=20, width=32, num_heads=1):
		super().__init__()
		self.network = FNO(modes, width, num_heads)
	
	def forward(self,tmp):
		tmp = self.network(tmp)
		return tmp

class discriminator(nn.Module):
	"""
		PatchGAN network that evaluates pxp patch in the field as real or fake.
		It takes the input and the target image and returns a 2D probability field, where each pixel gives the probability that the 
		corresponding pxp patch is real.

		Warning -
		- works for default psize only.
		- works for 1 stress component (num_heads=1) only
	"""

	def __init__(self,psize=70,num_heads=1):
		super().__init__()

		if psize == 70:
			self.conv1 = nn.Conv2d(3+num_heads,64,4,stride=2,padding=0)
			self.BN1 = nn.BatchNorm2d(64)
			self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=0)
			self.BN2 = nn.BatchNorm2d(128)
			self.conv3 = nn.Conv2d(128,256,4,stride=2,padding=0)
			self.BN3 = nn.BatchNorm2d(256)
			self.conv4 = nn.Conv2d(256,512,4,stride=1,padding=0)
			self.BN4 = nn.BatchNorm2d(256)
			self.conv4 = nn.Conv2d(512,512,4,stride=1,padding=0)
			self.BN4 = nn.BatchNorm2d(256)

		self.leakyrelu = nn.LeakyReLU()

	def forward(self,inp,target):
		tmp = torch.cat((inp,target),dim=-3)		## (3x256x256),(5x256x256)-->(8x256x256)
		
		tmp = self.conv1(tmp)
		tmp = self.BN1(tmp)
		tmp = self.leakyrelu(tmp)

		tmp = self.conv2(tmp)
		tmp = self.BN2(tmp)
		tmp = self.leakyrelu(tmp)

		tmp = self.conv3(tmp)
		tmp = self.BN3(tmp)
		tmp = self.leakyrelu(tmp)

		tmp = self.conv4(tmp)
		tmp = self.BN4(tmp)
		tmp = self.leakyrelu(tmp)

		tmp = self.conv5(tmp)
		tmp = self.BN5(tmp)
		tmp = self.leakyrelu(tmp)

		return tmp

