import os
import time
import torch
from inference_module.inference import Inference

def inference():
    ## driver for the inference and postprocessing
    inferObj = Inference()
    inferObj.loadData()
    inferObj.infer(interpolative=True,inferAll=False)
    
    inferObj.postprocessInferences()