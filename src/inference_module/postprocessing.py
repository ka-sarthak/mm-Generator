import os
import torch
import numpy as np
from utils.config_module import config
from utils.data_processing import filterPaths
from utils.postprocessing import plot, gradientImg
from utils.logger_module import Logger

class Postprocessing(object):
    def __init__(self,main_path):
        self.path = main_path
        self.getCaseTypePaths(main_path)
        self.getPostProcessingFunctions()
    
    def processCaseTypes(self):
        '''
            loops over each case, collects data, and executes processing
        '''
        print("Postprocessing the inferred cases as decribed in config ...")
        for caseTypePath in self.caseTypePaths:
            self.caseTypePath = caseTypePath
            self.getCaseTypeData()
            self.processCaseType()
            print(f"... {caseTypePath.split('/')[-2]} done")
        print("Postprocessing completed.")

    def processCaseType(self):
        '''
            applies each of the processing function
        '''
        self.logger = Logger(os.path.join(self.path,self.caseTypePath,"postprocessing.log"))
        
        self.aggregatedError()
        self.plotFields()
        self.plotErrorFields()
        self.gradientFields(data_type="true")
        self.gradientFields(data_type="pred")
        
        self.logger.close()
        
    
    def aggregatedError(self):
        '''
            computes the aggregated field error for each case,
            logs the results.
            - MAE: mean absolute error computed over a given field, and averaged over all cases
            - NMAE: MAE normalized with range over a given field, and averaged over all cases
        '''

        abs_error = np.abs(self.pred - self.true)
        mae_per_case = np.mean(abs_error,axis=(-2,-1))
        mae_per_caseType = np.mean(mae_per_case,axis=0)
    
        min_per_case = np.min(abs_error,axis=(-2,-1))
        max_per_case = np.max(abs_error,axis=(-2,-1))
        nmae_per_case = mae_per_case/(max_per_case-min_per_case)
        nmae_per_caseType = np.mean(nmae_per_case,axis=0)

        self.logger.addLine(f"MAE averaged over {self.pred.shape[0]} cases (MPa):")
        self.logger.addRow(mae_per_caseType)
        self.logger.addLine(f"NMAE averaged over {self.pred.shape[0]} cases (%):")
        self.logger.addRow(nmae_per_caseType*100.0)
    
    def plotFields(self):
        '''
            plots the field for all the components individually,
            both for the predictions and label/true. 
        '''
        plot_path = os.path.join(self.caseTypePath,"plot_field","")
        plot_cmap = config["postprocessing"]["plotFieldCMap"]
        os.makedirs(plot_path,exist_ok=True)
        for case_idx in range(len(self.true)):
            for comp_idx in range(len(self.true[case_idx])):
                vmin = np.min(self.true[case_idx,comp_idx,:,:])
                vmax = np.max(self.true[case_idx,comp_idx,:,:])
                
                # plot true field
                plot(data=self.true[case_idx,comp_idx,:,:],vmin=vmin,vmax=vmax,
                     cmap=plot_cmap,path_and_name=os.path.join(plot_path,f"{case_idx+1}_true_{comp_idx+1}"))
                
                # plot predicted field with same vmin and vmax 
                plot(data=self.pred[case_idx,comp_idx,:,:],vmin=vmin,vmax=vmax,
                     cmap=plot_cmap,path_and_name=os.path.join(plot_path,f"{case_idx+1}_pred_{comp_idx+1}"))
        
    def plotErrorFields(self):
        '''
            plots the field for all the components individually,
            both for the predictions and label/true. 
        '''
        plot_path = os.path.join(self.caseTypePath,"plot_error_field","")
        plot_cmap = config["postprocessing"]["plotErrorFieldCMap"]
        os.makedirs(plot_path,exist_ok=True)
        for case_idx in range(len(self.true)):
            for comp_idx in range(len(self.true[case_idx])):
                error = self.true[case_idx,comp_idx,:,:] - self.pred[case_idx,comp_idx,:,:]
                vmin = np.min(error)
                vmax = np.max(error)
                
                # plot error field
                plot(data=error,vmin=vmin,vmax=vmax,
                     cmap=plot_cmap,path_and_name=os.path.join(plot_path,f"{case_idx+1}_error_{comp_idx+1}"))
                
    
    def gradientFields(self,data_type="pred"):
        '''
            # for the purpose of measuring sharpness of the prediction
            plots the gradient field for all the predicted components individually,
            logs the mean-std of the gradient images
        '''
        if data_type=="pred":
            data  = self.pred
        elif data_type=="true":
            data  = self.true
        else:
            raise AssertionError("Unexpected argument for gradientFields.")
        
        plot_path = os.path.join(self.caseTypePath,f"plot_gradient_field_{data_type}","")
        plot_cmap = config["postprocessing"]["plotGradientFieldCMap"]
        os.makedirs(plot_path,exist_ok=True)
        
        cases_grad_mean = np.zeros(data.shape[1])
        cases_grad_std  = np.zeros(data.shape[1])
        for case_idx in range(len(data)):
            for comp_idx  in range(len(data[case_idx])):
                grad_img = gradientImg(torch.from_numpy(data[case_idx,comp_idx,:,:])).detach().numpy().squeeze()
                vmax = np.max(grad_img)
                vmin = np.min(grad_img)
                
                # plot error field
                plot(data=grad_img,vmin=vmin,vmax=vmax,
                     cmap=plot_cmap,path_and_name=os.path.join(plot_path,f"{case_idx+1}_grad_{comp_idx+1}"))
                cases_grad_mean[comp_idx] += np.mean(grad_img)
                cases_grad_std[comp_idx]  += np.std(grad_img)
        
        cases_grad_mean /= data.shape[0]
        cases_grad_std  /= data.shape[0]
        self.logger.addLine(f"Gradient image mean averaged over {data.shape[0]} {data_type} cases:")
        self.logger.addRow(cases_grad_mean)
        self.logger.addLine(f"Gradient image standard deviation averaged over {data.shape[0]} {data_type} cases:")
        self.logger.addRow(cases_grad_std)
        
    def getCaseTypeData(self):
        '''
            returns the saved data if found, else raises error 
        '''
        if os.path.isfile(os.path.join(self.caseTypePath,"pred.npy")):
            self.pred = np.load(os.path.join(self.caseTypePath,"pred.npy"))/1e6
        else:   raise AssertionError("Inference files are missing.")
        if os.path.isfile(os.path.join(self.caseTypePath,"true.npy")):
            self.true = np.load(os.path.join(self.caseTypePath,"true.npy"))/1e6
        else:   raise AssertionError("Inference files are missing.")
    
    def getCaseTypePaths(self,path):
        self.caseTypePaths = filterPaths(path,config["postprocessing"]["include"],config["postprocessing"]["exclude"])
    
    def getPostProcessingFunctions(self):
        self.function_list = config["postprocessing"]["functions"]
    