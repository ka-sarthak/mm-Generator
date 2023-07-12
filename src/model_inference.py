import os
import time
import torch
from inference_module.inference import Inference
from inference_module.postprocessing import Postprocessing

# from inference_module import 

def inference():
    ## driver for the inference and postprocessing classes
    inferObj = Inference()
    inferObj.loadData()
    inferObj.infer(interpolative=True,inferAll=False)
    
    # ppObj = Postprocessing()
    
    exit()
    

    ## interpolative generalization on the test data from the split.
    metric      =   L1Loss()
    wtime, loss =   0, 0
    test_loader =   torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    for x,y in test_loader:
        break
        t1      =   time.time()
        y_pred  =   g_model(x).detach()
        t2      =   time.time()
        
        y_pred  =   y_scaler.decode(y_pred)
        y       =   y_scaler.decode(y)
        
        wtime   +=  t2-t1
        loss    +=  metric(y_pred,y).item()/1e6
    with open(os.path.join(path_inference,"interpolative.out"), "w") as f:
        f.write(f"Inference on the test dataset from the split containing {len(test_loader)} cases:\n")
        f.write(f"Average wall time for inference (s) = {round(wtime/len(test_loader),4)}\n")
        f.write(f"Average MAE for inference (MPa) = {round(loss/len(test_loader),6)}\n")
    
    ## import the test cases and loop over them
    # get the list of all the test cases 
    test_cases  =   os.listdir(config["path"]["testingData"])
    test_paths  =   [os.path.join(path_inference,case) for case in test_cases]
    for path in test_paths:
        os.makedirs(path,exist_ok=True)
        
        
    ## use the paths from the config to selectively infer for test cases
        