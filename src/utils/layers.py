import torch
import torch.nn as nn
from utils.utilities import periodic_padding
from utils.config_module import config
from utils.probe_fourier_modes import probeFourierModes

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

class SpectralConv2d(nn.Module):
    """
    Taken from FNO paper.
    2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input_, weight):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy",input_,weight)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # determine the Fourier modes
        if not self.training and config["model"]["FNO"]["probeFourierModes"]: 
            probe_x = x_ft.detach().abs().numpy()
            probeFourierModes.collectData(probe_x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # spatial dimensions of the input and output activation maps is the same.

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=( x.size(-2), x.size(-1)))
        return x

class SpectralConv2dDropout(SpectralConv2d):
    def __init__(self, in_channels, out_channels, modes1, modes2, dropout_p = 0.8):
        super().__init__(in_channels, out_channels, modes1, modes2)
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # determine the Fourier modes
        if not self.training and config["model"]["FNO"]["probeFourierModes"]: 
            probe_x = x_ft.detach().abs().numpy()
            probeFourierModes.collectData(probe_x)

        mask = self.dropout(torch.ones_like(x_ft.real))
        x_ft = x_ft * mask 

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # spatial dimensions of the input and output activation maps is the same.

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=( x.size(-2), x.size(-1)))
        return x