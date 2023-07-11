import os
import time
import torch
import numpy as np
from torch.nn import L1Loss
from utils.config_module import config
from utils.data_processing import makePathAndDirectories, importTestDataset, importTrainDataset, scaleDataset
from models.generator import Generator

class Inference:
    def __init__(self):
        self.makePathAndDirectories()
        self.loadGeneratorAndScaler()
          
    def infer(self,interpolative=False,inferAll=False):
        '''
            Checks if the inference should be done for all or only un-inferred ones.
            Checks if inference for the test data (interpolative) from the train-val-test should be done.
            If interpolative inference exists, results are not inferred again.
        '''
        self.loadData()
        if not inferAll:
            print("Inferring for all the selected test case types which were not inferred before ...")
            for case_type in self.nonInferredCaseTypes():
                self._infer(case_type)
        else:
            print("Inferring for all the selected test case types ...")
            for case_type in self.data.keys():
                self._infer(case_type)
                
        ## putting inferCaseByCase after other inferences because once num_threads is set to 1,
        ## PyTorch doesn't let to set is back to other number for native parallel backend.
        if interpolative:
            print("Inferring for interpolative test case ...")
            self._inferCaseByCase(case_type="interpolative")
        
        print("Inference completed.")
                
    def _inferCaseByCase(self,case_type="interpolative"):
        '''
            generates the inference for the data case by case, while limiting number of threads to 1.
            also records the time taken for inference
        '''
        
        if case_type=="interpolative":
            if self.isInferenceNotExist(case_type):
                ## set the number of threads to 1
                threads_initial = torch.get_num_threads()
                torch.set_num_threads(1)
                
                data = importTrainDataset(only_test=True)
                x = self.x_scaler.encode(data["input"])
                
                time_list = []
                pred = np.empty(data["output"].shape)
                for case_idx in range(len(x)):
                    start = time.time()
                    _pred = self.g_model(x[case_idx][None,:,:,:]).detach()
                    _pred = self.y_scaler.decode(_pred).numpy()
                    end   = time.time()
                    time_list.append(end-start)
                    pred[case_idx] = _pred.squeeze()
                
                case_path = os.path.join(self.path_inference,case_type)
                os.makedirs(case_path,exist_ok=True)
                
                np.save(os.path.join(case_path,"pred.npy"),pred)
                with open(os.path.join(case_path,"time_recording.out"),"w") as f:
                    for i,t in enumerate(time_list):
                        f.write(f"{i+1}:\t\t{t}\n")
                    f.write(f"\nAverage time: {sum(time_list)/len(time_list)}")
                
                print(f"... {case_type} done")
        
                ## reset the number of threads to initial value
                torch.set_num_threads(threads_initial)
        
        else:
            # TODO add the support for inferring case by case for other test case types
            raise AssertionError("Feature not supported yet.")

    
    def _infer(self,case_type):
        '''
            infers the test case and saves .npy file
        '''
        data = self.data[case_type]
        x = self.x_scaler.encode(data["input"])
        pred = self.g_model(x).detach()
        pred = self.y_scaler.decode(pred).numpy()
        
        case_path = os.path.join(self.path_inference,case_type)
        os.makedirs(case_path,exist_ok=True)
        np.save(os.path.join(case_path,"pred.npy"),pred)
        print(f"... {case_type} done")
            
    def loadData(self):
        self.data = importTestDataset(ntest=10)
    
    def nonInferredCaseTypes(self):
        '''
            returns a list of cases for which inferred results are not available.
        '''
        inclusion_list = []
        for case_type in self.data.keys():
            if self.isInferenceNotExist(case_type=case_type):
                inclusion_list.append(case_type)
        print(f"Non-inferred test case types - \n{inclusion_list}")
        return inclusion_list
    
    def isInferenceNotExist(self,case_type):
        '''
            return True if the inference for the test case type does not exist, otherwise return False.
            checks if the folder is available, and then checks if the .npy file is available

        '''
        if case_type not in os.listdir(self.path_inference):
                return True
        else:
            if not os.path.isfile(os.path.join(self.path_inference,case_type,"pred.npy")):
                return True
        return False
    
    def postprocessInferences(self):
        # TODO
        raise AssertionError("Feature not supported yet.")

    
    def makePathAndDirectories(self):
        self.path_save_model, self.path_inference = makePathAndDirectories(training=False)
    
    def loadGeneratorAndScaler(self):
        '''
            loads the generator and scalers.
        '''
        g_model = Generator()
        gcheckpoint =   torch.load(self.path_save_model+"generator.pt")
        g_model.load_state_dict(gcheckpoint["model_state_dict"])
        self.g_model = g_model.eval()
        
        if "scaler" in gcheckpoint.keys():
            # TODO: load the scaler from saved state
            raise AssertionError("Feature not supported yet.")
        else:
            training_data,_,_ = importTrainDataset()
            _,_, self.x_scaler, self.y_scaler = scaleDataset(training_data)
            del training_data