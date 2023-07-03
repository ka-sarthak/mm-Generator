import numpy as np
import torch
import torch.nn as nn
from torch.nn import L1Loss, BCELoss
import torch.nn.functional as F
from generator_models.FNO import FNOBlock2d
from generator_models.UNet import UNet

class Generator(nn.Module):
	"""
		Picks up the network from the generator models: FNO, UNet, ...
		also acts as a wrapper function
	"""
	def __init__(self, model="UNet", num_heads=1):
		super().__init__()
		if model == "FNO":
			hparams = {"modes":20, "width":32}
			self.network = FNOBlock2d(hparams["modes"],hparams["modes"], hparams["width"], num_heads)
		elif model == "UNet":
			hparams = {"kernel":9, "in_channels":3}
			self.network = UNet(hparams["kernel"], hparams["in_channels"], num_heads, version="standard")

	def forward(self,x):
		x = self.network(x)
		return x

class Discriminator(nn.Module):
	"""
		PatchGAN network that evaluates pxp patch in the field as real or fake.
		It takes the input and the target image and returns a 2D probability field, where each pixel gives the probability that the 
		corresponding pxp patch is real.

		Warning -
		- works for default psize only.
		- works for 1 stress component (num_heads=1) only
	"""

	def __init__(self,version="modified",num_heads=1):
		super().__init__()
		'''
			Unlike original PatchGAN, we are not using padding. Therefore, the output fields are smaller than in the original paper.
		'''
		self.version = version
		if version == "UNet-enc":
			kernel = 9
			in_channels = 3 + num_heads
			enc_channels = np.array([16,32,64,128])
			self.Enc0 =  Encoding_standard(channels=[    in_channels, enc_channels[0]],  kernel=kernel)
			self.Enc1 =  Encoding_standard(channels=[enc_channels[0], enc_channels[1]],  kernel=kernel)
			self.Enc2 =  Encoding_standard(channels=[enc_channels[1], enc_channels[2]],  kernel=kernel)
			self.Enc3 =  Encoding_standard(channels=[enc_channels[2], enc_channels[3]],  kernel=kernel)
			self.Enc4 =  Encoding_standard(channels=[enc_channels[3], 				1],  kernel=1)
		
		if version == "original":
		## patch size of 70 will be evaluated
			self.conv1 = nn.Conv2d(3+num_heads,64,4,stride=2,padding=0)
			# self.BN1 = nn.BatchNorm2d(64)
			self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=0)
			self.BN2 = nn.BatchNorm2d(128)
			self.conv3 = nn.Conv2d(128,256,4,stride=2,padding=0)
			self.BN3 = nn.BatchNorm2d(256)
			self.conv4 = nn.Conv2d(256,512,4,stride=1,padding=0)
			self.BN4 = nn.BatchNorm2d(512)
			self.conv5 = nn.Conv2d(512,1,4,stride=1,padding=0)
			# self.BN5 = nn.BatchNorm2d(1)
		
		if version == "modified":
			self.conv1 = SeparableConv(3+num_heads,64,3,stride=2)
			self.BN1 = nn.BatchNorm2d(64)
			self.conv2 = SeparableConv(64,128,3,stride=1)
			self.BN2 = nn.BatchNorm2d(128)
			self.conv3 = SeparableConv(128,256,3,stride=1)
			self.BN3 = nn.BatchNorm2d(256)
			self.conv4 = SeparableConv(256,512,3,stride=1)
			self.BN4 = nn.BatchNorm2d(512)
			self.conv5 = SeparableConv(512,1,3,stride=1)

	def forward(self,inp,target):
		tmp = torch.cat((inp,target),dim=-3)		## (3x256x256),(5x256x256)-->(8x256x256)

		if self.version == "UNet-enc":
			tmp = self.Enc0(tmp)
			tmp = self.Enc1(tmp)
			tmp = self.Enc2(tmp)
			tmp = self.Enc3(tmp)
			tmp = torch.sigmoid(self.Enc4(tmp))
   
		elif self.version == "original" or self.version == "modified":
			tmp = F.leaky_relu(self.BN1(self.conv1(tmp)),0.2)
			tmp = F.leaky_relu(self.BN2(self.conv2(tmp)),0.2)
			tmp = F.leaky_relu(self.BN3(self.conv3(tmp)),0.2)
			tmp = F.leaky_relu(self.BN4(self.conv4(tmp)),0.2)
			tmp = torch.sigmoid(self.conv5(tmp))
		# tmp = F.leaky_relu(self.BN4(self.conv4(tmp)),0.2)
		# tmp = torch.sigmoid(self.conv5(tmp))

		return tmp

def GeneratorLoss(generated_field, target_field, disc_generated_field, LAMBDA=100):
	BCEloss = BCELoss()
	L1loss = L1Loss()
	
	L1 = L1loss(generated_field,target_field)
	g = BCEloss(disc_generated_field,torch.ones_like(disc_generated_field))
	
	gen_loss = g + L1*LAMBDA
	
	return {"total_loss":gen_loss, "gan_loss":g, "L1_loss":L1}

def DiscriminatorLoss(disc_generated_field, disc_target_field):
	BCEloss = BCELoss()
	
	generated_loss = BCEloss(disc_generated_field,torch.zeros_like(disc_generated_field))
	real_loss = BCEloss(disc_target_field,torch.ones_like(disc_target_field))

	disc_loss = (generated_loss+real_loss)*0.5

	return {"total_loss":disc_loss, "gan_generated_loss":generated_loss, "gan_real_loss":real_loss}


class SeparableConv(nn.Module):
	"""
		first convolution done individually over all channels
		second convolution is 1x1 convolution, which convolves at same spatial position for all channels 
	"""
	def __init__(self, in_channels, out_channels, kernel, stride):
		super().__init__()
		self.convSpatial   = nn.Conv2d(in_channels,in_channels ,kernel_size=kernel,stride=stride,padding="valid",groups=in_channels,bias=False)
		self.convDepthwise = nn.Conv2d(in_channels,out_channels,kernel_size=1,     stride=stride,padding="valid")
	
	def forward(self, x):
		return self.convDepthwise(self.convSpatial(x))


class Encoding_standard(nn.Module):
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

class PeriodicSeparableConv(nn.Module):
	"""
		first convolution done individually over all channels
		second convolution is 1x1 convolution, which convolves at same spatial position for all channels 
	"""
	def __init__(self, in_channels, out_channels, kernel):
		super().__init__()
		# self.convSpatial   = nn.Conv2d(in_channels,in_channels ,kernel_size=kernel,groups=in_channels,padding="same",bias=False)
		# self.convDepthwise = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding="same")
		self.convSpatial   = nn.Conv2d(in_channels,in_channels ,kernel_size=kernel,groups=in_channels,padding="valid",bias=False)
		self.convDepthwise = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding="valid")
		self.padding = int((kernel-1)/2)
	
	def forward(self, x):
		x = periodic_padding(x,[2,3],[self.padding,self.padding])
		return self.convDepthwise(self.convSpatial(x))

def periodic_padding(tensor, axis, padding):
    """
		Add periodic padding to a tensor for specified axis.

		:param tensor: the input tensor.
		:param axis: one or multiple axis for padding; an integer or a tuple of ints.
		:param padding: the padding size; int or tuple of ints corresponding to axis.
		:return: padded tensor.
    """

    if isinstance(axis, int):
        axis = (axis, )
    if isinstance(padding, int):
        padding = (padding, )
    assert len(axis) == len(padding), 'the number of axis and paddings are different.'
    ndim = len(tensor.shape)
    for ax, p in zip(axis, padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left
        ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = torch.cat([right, middle, left], axis=ax)
    return tensor