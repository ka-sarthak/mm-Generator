import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import SpectralConv2d, SpectralConv2dDropout, SpectralConv2dAmplitude
from utils.config_module import config

class FNO(nn.Module):
    '''
        FNO wrapper class
    ''' 
    def __init__(self):
        super().__init__()

        num_heads = config["experiment"]["outputHeads"]
        version = config["model"]["FNO"]["version"]
        modes1 = config["model"]["FNO"]["modes1"]
        modes2 = config["model"]["FNO"]["modes2"]
        width = config["model"]["FNO"]["width"]

        if version == "standard":
            self.network = FNOBlock2d(modes1,modes2,width,num_heads)
        elif version == "standard_from_firstFL":
            self.network = FNOBlock2d_from_firstFL(modes1,modes2,width,num_heads)
        elif version == "standard_from_thirdFL":
            self.network = FNOBlock2d_from_thirdFL(modes1,modes2,width,num_heads)
        elif version == "dropout":
            self.network = FNOBlock2d_dropout(modes1,modes2,width,num_heads)
        elif version == "amplitude":
            self.network = FNOBlock2d_amplitude(modes1,modes2,width,num_heads)
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
        self.fc0 = nn.Linear(config["experiment"]["inputHeads"], self.width) # input channel is 3

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

class FNOBlock2d_dropout(FNOBlock2d):
    def __init__(self, modes1, modes2, width, num_heads):
        super().__init__(modes1, modes2, width, num_heads)
        self.conv0 = SpectralConv2dDropout(self.width, self.width, self.modes1, self.modes2, config["model"]["FNO"]["dropout"])
        self.conv1 = SpectralConv2dDropout(self.width, self.width, self.modes1, self.modes2, config["model"]["FNO"]["dropout"])
        self.conv2 = SpectralConv2dDropout(self.width, self.width, self.modes1, self.modes2, config["model"]["FNO"]["dropout"])
        self.conv3 = SpectralConv2dDropout(self.width, self.width, self.modes1, self.modes2, config["model"]["FNO"]["dropout"])
        
    def forward(self, x):
        return super().forward(x)
    
class FNOBlock2d_amplitude(FNOBlock2d):
    def __init__(self, modes1, modes2, width, num_heads):
        super().__init__(modes1, modes2, width, num_heads)
        self.conv0 = SpectralConv2dAmplitude(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2dAmplitude(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2dAmplitude(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2dAmplitude(self.width, self.width, self.modes1, self.modes2)
        
    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        return super().forward(x)
    
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
        self.fc0 = nn.Linear(config["experiment"]["inputHeads"], self.width) # input channel is 3

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
        self.fc0 = nn.Linear(config["experiment"]["inputHeads"], self.width) # input channel is 3

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
