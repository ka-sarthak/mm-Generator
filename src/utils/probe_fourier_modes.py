import os 
from utils.config_module import config
from utils.postprocessing import plotAllChannels

class ProbeFourierModes():
    '''
        module to compute the Fourier modes in 2D FFT.
        module is in parallel to training module.
        Fourier modes are calculated and plotted at every epoch during eval on validation data.
            - the first example of the validation data it taken
            - when it passes through a Fourier layer, the magnitude of forward FFT is plotted for all the channels
            - this is done for all the Fourier layers
        plots are made for rescaled data of range [0,1], using cmap Greys where white is lower end (0), and black is higher end (1). 
    '''
    def __init__(self):
        self.layer = 1
        self.epoch = 1
        self.createPath()
 
    def createPath(self):
        self.path = os.path.join(config["path"]["saveModel"],config["experiment"]["model"], config["experiment"]["generator"], config["experiment"]["name"],"FourierModesCalculation","")
        os.makedirs(self.path,exist_ok=True)
        
    def evaluate(self, x):
        # performing computations on the first example (batch) of field
        x_channels = x[0,:,:,:].abs()
        img_name = self.path+"epoch_"+str(self.epoch)+"_layer_"+str(self.layer)
        self.plotAllChannels(x_channels,img_name,rescaling=True)
        self.layer += 1
     
    def plotAllChannels(self,data,img_name,rescaling=False):
        plotAllChannels(data=data,cmap="Greys",path_and_name=img_name,rescaling=rescaling)
        
    def setEpoch(self,ep):
        self.epoch = ep
        
    def flush(self):
        self.layer = 1

def initProbeFourierModes():
    global probeFourierModes
    probeFourierModes = ProbeFourierModes()