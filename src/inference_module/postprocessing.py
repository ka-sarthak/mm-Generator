import os
import numpy as np
from utils.config_module import config
from utils.data_processing import filterPaths
from utils.logger_module import Logger

class Postprocessing(object):
    def __init__(self,main_path):
        self.path = main_path
        self.getCaseTypePaths(main_path)
        self.getPostProcessingFunctions()
    
    def processData(self):
        '''
            loops over each case, collects data, and executes processing
        '''
        print("Postprocessing the inferred cases as decribed in config ...")
        for casePath in self.caseTypePaths:
            pred, true = self.getData(casePath)
            logger = Logger(os.path.join(self.path,casePath,"postprocessing.log"))
            self._processData(pred,true,logger)
            logger.close()
            print(f"... {casePath.split('/')[-2]} done")
        print("Postprocessing completed.")

    def _processData(self,pred,true,logger):
        '''
            applies each of the processing function
        '''
        # TODO: add the functions for postprocessing
        print(pred.shape,true.shape)
        
        
    def getData(self,casePath):
        '''
            returns the saved data if found, else raises error 
        '''
        if os.path.isfile(os.path.join(casePath,"pred.npy")):
            pred = np.load(os.path.join(casePath,"pred.npy"))
        else:   raise AssertionError("Inference files are missing.")
        if os.path.isfile(os.path.join(casePath,"true.npy")):
            true = np.load(os.path.join(casePath,"true.npy"))
        else:   raise AssertionError("Inference files are missing.")
        
        return pred, true
    
    def getCaseTypePaths(self,path):
        self.caseTypePaths = filterPaths(path,config["postprocessing"]["include"],config["postprocessing"]["exclude"])
    
    def getPostProcessingFunctions(self):
        self.function_list = config["postprocessing"]["functions"]
    