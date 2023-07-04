import torch.nn as nn
from utils.utilities import periodic_padding

class PeriodicConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel):
		super().__init__()
		self.conv = nn.Conv2d(in_channels,out_channels,kernel,padding="valid")
		self.padding = int((kernel-1)/2)
	
	def forward(self, x):
		x = periodic_padding(x,[2,3],[self.padding,self.padding])
		return self.conv(x)


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