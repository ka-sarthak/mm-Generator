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
        data: {epoch{layer[channel,height,width]}}
    '''
    def __init__(self):
        self.layer = 1
        self.epoch = None
        self.collect = None
        self.data = {}
        self.epochData = {}
 
    def startCollection(self,epoch=1):
        self.collect = True
        self.epoch = epoch
        print("Probe Start :: probeFourierMode.")
        
    def stopCollection(self):
        self.collect = False
        self._plot(rescaling=True)
        print(f"Probe Stop :: probeFourierMode has data from {max(list(self.data.keys()))} epochs.")
        
    def collectData(self, x):
        '''
            performing computations on the first example (batch) of field if collect is on
            saving the results from each epoch
        '''
        if self.collect == True:
            self.epochData[self.layer] = x[0,:,:,:]
        self.layer += 1
             
    def epochCompleted(self):
        self.data[self.epoch] = self.epochData
        self.layer = 1
        self.epoch += 1
        
    def getData(self):
        return self.data
    
    def setData(self,data):
        self.data = data
        
    def _plot(self,rescaling=False):
        self._createPath()
        for ep,ep_data in self.data.items():
            for layer,layer_data in ep_data.items():
                img_name = self.path+"epoch_"+str(ep)+"_layer_"+str(layer)
                plotAllChannels(data=layer_data,cmap="Greys",path_and_name=img_name,rescaling=rescaling)

    def _createPath(self):
        self.path = os.path.join(config["path"]["saveModel"],config["experiment"]["model"], config["experiment"]["generator"], config["experiment"]["name"],"FourierModesCalculation","")
        os.makedirs(self.path,exist_ok=True)

def initProbeFourierModes():
    global probeFourierModes
    probeFourierModes = ProbeFourierModes()