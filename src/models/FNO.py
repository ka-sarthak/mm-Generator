import torch
import torch.nn as nn
import torch.nn.functional as F

class FNO(nn.Module):
    '''
        FNO wrapper class
    ''' 
    def __init__(self,modes1,modes2,width,num_heads,version="standard"):
        super().__init__()
        if version == "standard":
            self.network = FNOBlock2d(modes1,modes2,width,num_heads)
        elif version == "standard_from_firstFL":
            self.network = FNOBlock2d_from_firstFL(modes1,modes2,width,num_heads)
        elif version == "standard_from_thirdFL":
            self.network = FNOBlock2d_from_thirdFL(modes1,modes2,width,num_heads)
        else:
            raise AssertionError("Unexpected FNO version.")
    
    def forward(self,x):
        return self.network(x)

class FNOBlock2d(nn.Module):
    """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input shape : (batchsize, dimx, dimy, 3)
        output shape: (batchsize, dimx, dimy, num_heads)
    """
    def __init__(self, modes1, modes2,  width, num_heads):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_heads = num_heads
        self.fc0 = nn.Linear(3, self.width) # input channel is 3

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.num_heads)                        # change this to number of PK components

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        return x

class FNOBlock2d_from_thirdFL(nn.Module):
    """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input shape : (batchsize, dimx, dimy, 3)
        output shape: (batchsize, dimx, dimy, num_heads)
    """
    def __init__(self, modes1, modes2,  width, num_heads):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_heads = num_heads
        self.fc0 = nn.Linear(3, self.width) # input channel is 3

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        
        self.conv2List = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_heads)])
        self.conv3List = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_heads)])
        self.w2List =    nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_heads)])
        self.w3List =    nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_heads)])

        self.fc1List =   nn.ModuleList([nn.Linear(self.width, 128) for _ in range(self.num_heads)])
        self.fc2List =   nn.ModuleList([nn.Linear(128, 1) for _ in range(self.num_heads)])                        # change this to number of PK components

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        _x = F.relu(x)

        output = torch.tensor([])
        for head in range(self.num_heads):
            x1 = self.conv2List[head](_x)
            x2 = self.w2List[head](_x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2
            x = F.relu(x)

            x1 = self.conv3List[head](x)
            x2 = self.w3List[head](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2

            x = x.permute(0, 2, 3, 1)
            x = self.fc1List[head](x)
            x = F.relu(x)
            x = self.fc2List[head](x)
            x = x.permute(0, 3, 1, 2)

            output = torch.cat((output,x),dim=1)

        return output
    
class FNOBlock2d_from_firstFL(nn.Module):
    """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input shape : (batchsize, dimx, dimy, 3)
        output shape: (batchsize, dimx, dimy, num_heads)
    """
    def __init__(self, modes1, modes2,  width, num_heads):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_heads = num_heads
        self.fc0 = nn.Linear(3, self.width) # input channel is 3

        self.conv0List = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_heads)])
        self.conv1List = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_heads)])
        self.conv2List = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_heads)])
        self.conv3List = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_heads)])
        self.w0List =    nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_heads)])
        self.w1List =    nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_heads)])
        self.w2List =    nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_heads)])
        self.w3List =    nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_heads)])

        self.fc1List =   nn.ModuleList([nn.Linear(self.width, 128) for _ in range(self.num_heads)])
        self.fc2List =   nn.ModuleList([nn.Linear(128, 1) for _ in range(self.num_heads)])                       # change this to number of PK components

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        _x = x.permute(0, 3, 1, 2)

        output = torch.tensor([])
        for head in range(self.num_heads):
            x1 = self.conv0List[head](_x)
            x2 = self.w0List[head](_x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2
            x = F.relu(x)

            x1 = self.conv1List[head](x)
            x2 = self.w1List[head](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2
            x = F.relu(x)

            x1 = self.conv2List[head](x)
            x2 = self.w2List[head](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2
            x = F.relu(x)

            x1 = self.conv3List[head](x)
            x2 = self.w3List[head](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2

            x = x.permute(0, 2, 3, 1)
            x = self.fc1List[head](x)
            x = F.relu(x)
            x = self.fc2List[head](x)
            x = x.permute(0, 3, 1, 2)

            output = torch.cat((output,x),dim=1)

        return output

class SpectralConv2d(nn.Module):
    """
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
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # spatial dimensions of the input and output activation maps is the same.

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=( x.size(-2), x.size(-1)))
        return x


