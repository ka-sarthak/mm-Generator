import os
import time
import torch
import numpy as np
from utils.utilities import lossFunction
from utils.config_module import config
from utils.logger_module import Logger
from utils.data_processing import makePathAndDirectories, importTestDataset, importTrainDataset, scaleDataset
from models.generator import Generator
from inference_module.postprocessing import Postprocessing

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
            self._inferCaseByCase(case_type="interpolative")
        
        print("Inference completed.")
                
    def _inferCaseByCase(self,case_type="interpolative"):
        '''
            generates the inference for the data case by case, while limiting number of threads to 1.
            also records the time taken for inference and the errors generated
        '''
        
        if case_type=="interpolative":
            if self.isInferenceNotExist(case_type):
                print("Inferring for interpolative test case ...")
                
                ## set the number of threads to 1
                threads_initial = torch.get_num_threads()
                torch.set_num_threads(1)
                
                data = importTrainDataset(only_test=True)
                x = self.x_scaler.encode(data["input"])
                y = data["output"]
                
                time_list = []
                error_list = []
                metric = lossFunction(type=config["training"]["metric"])
                pred = np.empty(data["output"].shape)
                for case_idx in range(len(x)):
                    start = time.time()
                    _pred = self.g_model(x[case_idx][None,:,:,:]).detach()
                    _pred = self.y_scaler.decode(_pred).squeeze()
                    end   = time.time()
                    error = [metric(y[case_idx][i],_pred[i]).item()/1e6 for i in range(len(_pred))]

                    time_list.append([case_idx+1, round(end-start,6)])
                    error_list.append([case_idx+1]+error)
                
                    pred[case_idx] = _pred.numpy()
                
                case_path = os.path.join(self.path_inference,case_type)
                os.makedirs(case_path,exist_ok=True)
                
                avg_time = np.mean(np.array(time_list)[:,1:],axis=0)[0]    
                avg_error_list = list(np.mean(np.array(error_list)[:,1:],axis=0))
                
                logger = Logger(os.path.join(case_path,"inference.log"))
                logger.addTable(time_list,"Time taken (sec) per inference using 1 thread")
                logger.addLine(f"Average time (sec): {np.round(avg_time,6)}")
                logger.addTable(error_list,f"Aggregated error (MPa) using {config['training']['metric']} metric for {config['experiment']['outputHeads']} component(s)")
                logger.addLine(f"Average aggregate error (MPa):")
                logger.addRow(avg_error_list)
                logger.close()
                
                np.save(os.path.join(case_path,"pred.npy"),np.array(pred))
                np.save(os.path.join(case_path,"true.npy"),np.array(y.numpy()))
                print(f"... {case_type} done")
        
                ## reset the number of threads to initial value
                torch.set_num_threads(threads_initial)
            else:
                print("Inferring for interpolative test case already done.")
                
        else:
            # TODO add the support for inferring case by case for other test case types
            raise AssertionError("Feature not supported yet.")

    
    def _infer(self,case_type):
        '''
            infers the test case and saves .npy file
        '''
        data = self.data[case_type]
        x = self.x_scaler.encode(data["input"])
        y = data["output"]
        
        pred = self.g_model(x).detach()
        pred = self.y_scaler.decode(pred).numpy()
        
        case_path = os.path.join(self.path_inference,case_type)
        os.makedirs(case_path,exist_ok=True)
        np.save(os.path.join(case_path,"pred.npy"),pred)
        np.save(os.path.join(case_path,"true.npy"),y.numpy())
        print(f"... {case_type} done")
            
    def loadData(self):
        self.data = importTestDataset(ntest=config["inference"]["nTest"])
    
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
        '''
            initialize an Postprocessing object, which goes over all the existing inferences, 
            filters them based on config, and computes the postprocessing results for the functions
            based on config.
        '''
        processor = Postprocessing(self.path_inference)
        processor.processCaseTypes()
    
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