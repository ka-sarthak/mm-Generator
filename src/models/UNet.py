import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.layers import PeriodicSeparableConv
from utils.config_module import config

class UNet(nn.Module):
	'''
		UNet Wrapper class
	'''
	def __init__(self):
		super().__init__()
		in_channels = config["experiment"]["inputHeads"]
		out_channels = config["experiment"]["outputHeads"]
		kernel = config["model"]["UNet"]["kernel"]
		version = config["model"]["UNet"]["version"]
		if version == "standard":
			self.network = UNet_standard(kernel, in_channels, out_channels)
		elif version == "standard_from_output":
			self.network = UNet_standard_from_output(kernel, in_channels, out_channels)
		elif version == "standard_from_enc":
			self.network = UNet_standard_from_enc(kernel, in_channels, out_channels)
		elif version == "modified":
			self.network = UNet_modified(kernel, in_channels, out_channels)
		else:
			raise AssertionError("Unexpected UNet version.")

	def	forward(self, x):
		return self.network(x)	

class UNet_standard(nn.Module):
	'''
		Skip connections are saved right after the conv. 
		First skip connection is not the input tensor. 
		Includes an interface convolution between enc and dec stages.
		Includes an ending convolution that reduced #channels to 1.

		input shape : (batchsize, dimx, dimy, 3)
        output shape: (batchsize, dimx, dimy, num_heads)
	'''
	def __init__(self, kernel, in_channels, out_channels):
		super().__init__()
		self.num_heads = out_channels

		enc_channels = config["model"]["UNet"]["encChannels"]
		interface_channel = config["model"]["UNet"]["interfaceChannels"]
		dec_channels = config["model"]["UNet"]["decChannels"]

		self.Enc0 =  Encoding_standard(channels=[    in_channels, enc_channels[0]],  kernel=kernel)
		self.Enc1 =  Encoding_standard(channels=[enc_channels[0], enc_channels[1]],  kernel=kernel)
		self.Enc2 =  Encoding_standard(channels=[enc_channels[1], enc_channels[2]],  kernel=kernel)
		self.Enc3 =  Encoding_standard(channels=[enc_channels[2], enc_channels[3]],  kernel=kernel)

		self.InterfaceConv = Interface(channels=[enc_channels[3], interface_channel], kernel=kernel)

		self.Dec0List = nn.ModuleList([Decoding(channels=[interface_channel+enc_channels[3], dec_channels[0]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec1List = nn.ModuleList([Decoding(channels=[  dec_channels[0]+enc_channels[2], dec_channels[1]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec2List = nn.ModuleList([Decoding(channels=[  dec_channels[1]+enc_channels[1], dec_channels[2]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec3List = nn.ModuleList([Decoding(channels=[  dec_channels[2]+enc_channels[0], dec_channels[3]], kernel=kernel) for _ in range(self.num_heads)])

		self.EndConvList = nn.ModuleList([EndConv(channels=[dec_channels[3], 1],kernel=1) for _ in range(self.num_heads)])
		
	def forward(self, x):
		in_shape = x.shape
		# four layers of encoding
		encoded_layers = []
		x, skip_conn = self.Enc0(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc1(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc2(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc3(x)
		encoded_layers.append(skip_conn)

		enc = self.InterfaceConv(x)

		# four layers of decoding and individual decoding for each head
		output = torch.empty((in_shape[0],self.num_heads,in_shape[2],in_shape[3]))
		for head in range(self.num_heads):
			x =    self.Dec0List[head](enc, encoded_layers[3])
			x =    self.Dec1List[head](  x, encoded_layers[2])
			x =    self.Dec2List[head](  x, encoded_layers[1])
			x =    self.Dec3List[head](  x, encoded_layers[0])
			x =    self.EndConvList[head](  x)
			
			x = torch.sigmoid(x)
			output[:,head,:,:] = x[:,0,:,:]
		
		return output

class UNet_standard_from_enc(nn.Module):
	'''
		Skip connections are saved right after the conv. 
		First skip connection is not the input tensor. 
		Includes an interface convolution between enc and dec stages.
		Includes an ending convolution that reduced #channels to 1.

		input shape : (batchsize, dimx, dimy, 3)
        output shape: (batchsize, dimx, dimy, num_heads)
	'''
	def __init__(self, kernel, in_channels, out_channels):
		super().__init__()
		self.num_heads = out_channels

		enc_channels = config["model"]["UNet"]["encChannels"]
		interface_channel = config["model"]["UNet"]["interfaceChannels"]
		dec_channels = config["model"]["UNet"]["decChannels"]

		self.Enc0List =  nn.ModuleList([Encoding_standard(channels=[    in_channels, enc_channels[0]],  kernel=kernel) for _ in range(self.num_heads)])
		self.Enc1List =  nn.ModuleList([Encoding_standard(channels=[enc_channels[0], enc_channels[1]],  kernel=kernel) for _ in range(self.num_heads)])
		self.Enc2List =  nn.ModuleList([Encoding_standard(channels=[enc_channels[1], enc_channels[2]],  kernel=kernel) for _ in range(self.num_heads)])
		self.Enc3List =  nn.ModuleList([Encoding_standard(channels=[enc_channels[2], enc_channels[3]],  kernel=kernel) for _ in range(self.num_heads)])

		self.InterfaceConvList = nn.ModuleList([Interface(channels=[enc_channels[3], interface_channel], kernel=kernel) for _ in range(self.num_heads)])

		self.Dec0List = nn.ModuleList([Decoding(channels=[interface_channel+enc_channels[3], dec_channels[0]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec1List = nn.ModuleList([Decoding(channels=[  dec_channels[0]+enc_channels[2], dec_channels[1]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec2List = nn.ModuleList([Decoding(channels=[  dec_channels[1]+enc_channels[1], dec_channels[2]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec3List = nn.ModuleList([Decoding(channels=[  dec_channels[2]+enc_channels[0], dec_channels[3]], kernel=kernel) for _ in range(self.num_heads)])

		self.EndConvList = nn.ModuleList([EndConv(channels=[dec_channels[3], 1],kernel=1) for _ in range(self.num_heads)])
		
	def forward(self, x):
		in_shape = x.shape
		_x = x
		
		output = torch.empty((in_shape[0],self.num_heads,in_shape[2],in_shape[3]))
		for head in range(self.num_heads):
			# four layers of encoding
			encoded_layers = []
			x, skip_conn = self.Enc0List[head](_x)
			encoded_layers.append(skip_conn)
			x, skip_conn = self.Enc1List[head](x)
			encoded_layers.append(skip_conn)
			x, skip_conn = self.Enc2List[head](x)
			encoded_layers.append(skip_conn)
			x, skip_conn = self.Enc3List[head](x)
			encoded_layers.append(skip_conn)

			enc = self.InterfaceConvList[head](x)

			# four layers of decoding	
			x =    self.Dec0List[head](enc, encoded_layers[3])
			x =    self.Dec1List[head](  x, encoded_layers[2])
			x =    self.Dec2List[head](  x, encoded_layers[1])
			x =    self.Dec3List[head](  x, encoded_layers[0])
			x =    self.EndConvList[head](  x)
			
			x = torch.sigmoid(x)
			output[:,head,:,:] = x[:,0,:,:]
		
		return output

class UNet_standard_from_output(nn.Module):
	'''
		Skip connections are saved right after the conv. 
		First skip connection is not the input tensor. 
		Includes an interface convolution between enc and dec stages.
		Includes an ending convolution that reduced #channels to 1.

		input shape : (batchsize, dimx, dimy, 3)
        output shape: (batchsize, dimx, dimy, num_heads)
	'''
	def __init__(self, kernel, in_channels, out_channels):
		super().__init__()
		self.num_heads = out_channels

		enc_channels = config["model"]["UNet"]["encChannels"]
		interface_channel = config["model"]["UNet"]["interfaceChannels"]
		dec_channels = config["model"]["UNet"]["decChannels"]

		self.Enc0 =  Encoding_standard(channels=[    in_channels, enc_channels[0]],  kernel=kernel)
		self.Enc1 =  Encoding_standard(channels=[enc_channels[0], enc_channels[1]],  kernel=kernel)
		self.Enc2 =  Encoding_standard(channels=[enc_channels[1], enc_channels[2]],  kernel=kernel)
		self.Enc3 =  Encoding_standard(channels=[enc_channels[2], enc_channels[3]],  kernel=kernel)

		self.InterfaceConv = Interface(channels=[enc_channels[3], interface_channel], kernel=kernel)

		self.Dec0 = Decoding(channels=[interface_channel+enc_channels[3], dec_channels[0]], kernel=kernel)
		self.Dec1 = Decoding(channels=[  dec_channels[0]+enc_channels[2], dec_channels[1]], kernel=kernel)
		self.Dec2 = Decoding(channels=[  dec_channels[1]+enc_channels[1], dec_channels[2]], kernel=kernel)
		self.Dec3 = Decoding(channels=[  dec_channels[2]+enc_channels[0], dec_channels[3]], kernel=kernel)

		self.EndConv = EndConv(channels=[dec_channels[3], self.num_heads],kernel=1)
		
	def forward(self, x):
		in_shape = x.shape
		# four layers of encoding
		encoded_layers = []
		x, skip_conn = self.Enc0(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc1(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc2(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc3(x)
		encoded_layers.append(skip_conn)

		enc = self.InterfaceConv(x)

		# four layers of decoding 
		x = self.Dec0(enc, encoded_layers[3])
		x = self.Dec1(  x, encoded_layers[2])
		x = self.Dec2(  x, encoded_layers[1])
		x = self.Dec3(  x, encoded_layers[0])

		x = self.EndConv(x)
		output = torch.sigmoid(x)
		
		return output

class UNet_modified(nn.Module):
	'''
		This version of UNet stores the skip connection after convolution+BN+Pooling.
		As such, the oldest skip connection has to be the untouched input. 
		Additionally, at the interface of encoding-decoding, we have 
			conv+BN+pool (en3) -> upsample+concat -> conv
		That is, an additional conv between pool and upsample is absent.

		input shape : (batchsize, dimx, dimy, 3)
        output shape: (batchsize, dimx, dimy, num_heads)
	'''
	def __init__(self, kernel, in_channels, out_channels):
		super().__init__()
		self.num_heads = out_channels
		
		enc_channels = config["model"]["UNet"]["encChannels"]
		dec_channels = config["model"]["UNet"]["decChannels"]

		self.Enc0 = Encoding_modified(channels=[    in_channels, enc_channels[0]],  kernel=kernel)
		self.Enc1 = Encoding_modified(channels=[enc_channels[0], enc_channels[1]],  kernel=kernel)
		self.Enc2 = Encoding_modified(channels=[enc_channels[1], enc_channels[2]],  kernel=kernel)
		self.Enc3 = Encoding_modified(channels=[enc_channels[2], enc_channels[3]],  kernel=kernel)

		self.Dec0List = nn.ModuleList([Decoding(channels=[dec_channels[0] + enc_channels[2], dec_channels[1]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec1List = nn.ModuleList([Decoding(channels=[dec_channels[1] + enc_channels[1], dec_channels[2]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec2List = nn.ModuleList([Decoding(channels=[dec_channels[2] + enc_channels[0], dec_channels[3]], kernel=kernel) for _ in range(self.num_heads)])
		self.Dec3List = nn.ModuleList([Decoding(channels=[dec_channels[3] +     in_channels,  1], kernel=kernel) for _ in range(self.num_heads)])

	def forward(self, x):
		in_shape = x.shape
		# four layers of encoding
		encoded_layers = []
		encoded_layers.append(x)
		x, skip_conn = self.Enc0(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc1(x)
		encoded_layers.append(skip_conn)
		x, skip_conn = self.Enc2(x)
		encoded_layers.append(skip_conn)
		enc, skip_conn = self.Enc3(x)

		# four layers of decoding and individual decoding for each head
		output = torch.empty((in_shape[0],self.num_heads,in_shape[2],in_shape[3]))
		for head in range(self.num_heads):
			x = self.Dec0List[head](enc, encoded_layers[3])
			x = self.Dec1List[head](  x, encoded_layers[2])
			x = self.Dec2List[head](  x, encoded_layers[1])
			x = self.Dec3List[head](  x, encoded_layers[0])
			x = torch.sigmoid(x)
			output[:,head,:,:] = x[:,0,:,:]
		
		return output


class Encoding_modified(nn.Module):
	def __init__(self, channels, kernel):
		super().__init__()
		self.conv = PeriodicSeparableConv(channels[0], channels[1], kernel)
		self.bn = nn.BatchNorm2d(channels[1])
		self.maxPool = nn.MaxPool2d((2,2))
	
	def forward(self, x):
		x = self.conv(x)
		x = self.maxPool(self.bn(x))
		return x, x
	
class Encoding_standard(nn.Module):
	'''
		Taken from the original UNet 2016 paper
	'''
	def __init__(self, channels, kernel):
		super().__init__()
		self.conv = PeriodicSeparableConv(channels[0], channels[1], kernel)
		self.bn = nn.BatchNorm2d(channels[1])
		self.maxPool = nn.MaxPool2d((2,2))
	
	def forward(self, x):
		x = self.conv(x)
		x = F.leaky_relu(self.bn(x),0.4)
		skip_connection = x
		x = self.maxPool(x)
		return x, skip_connection

class Interface(nn.Module):
	def __init__(self, channels, kernel):
		super().__init__()
		self.conv = PeriodicSeparableConv(channels[0], channels[1], kernel)
		self.bn = nn.BatchNorm2d(channels[1])

	def forward(self, x):
		x = F.leaky_relu(self.bn(self.conv(x)),0.4)
		return x

class Decoding(nn.Module):
	def __init__(self, channels, kernel):
		super().__init__()
		self.conv = PeriodicSeparableConv(channels[0], channels[1], kernel)
		self.bn = nn.BatchNorm2d(channels[1])

	def forward(self,x,skip_connection):
		x = F.interpolate(x,scale_factor=2,mode='bilinear')
		x = torch.cat((x,skip_connection),axis=1)	# concat along the channel dim
		x = self.conv(x)
		x = F.leaky_relu(self.bn(x),0.4)
		return x
		
class EndConv(nn.Module):
	def __init__(self, channels, kernel=1):
		super().__init__()
		self.conv = nn.Conv2d(channels[0],channels[1],kernel_size=kernel,padding="valid")

	def forward(self, x):
		return self.conv(x)
