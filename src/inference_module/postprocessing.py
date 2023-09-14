import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
        self.logger = Logger(os.path.join(self.path,self.caseTypePath,"postprocessing.log"),config["postprocessing"]["overwriteLogger"])

        
        if "plotFields" in self.function_list:
            self.plotFields()
        if "plotValueHistograms" in self.function_list:
            self.plotValueHistograms()
        
        if "aggregatedError" in self.function_list:
            self.aggregatedError()
        if "plotErrorFields" in self.function_list:
            self.plotErrorFields()
        if "plotMeanErrorFields" in self.function_list:
            self.plotMeanErrorFields()
        if "plotErrorHistograms" in self.function_list:
            self.plotErrorHistograms()
        
        if "gradientFieldsTRUE" in self.function_list:
            self.gradientFields(data_type="true")
        if "gradientFieldsPRED" in self.function_list:
            self.gradientFields(data_type="pred")
        
        if "mechEquilibriumCondition" in self.function_list:
            self.mechEquilibriumCondition()
        if "periodicityCondition" in self.function_list:
            self.periodicityCondition()
            
        if "FourierAnalysis" in self.function_list:
            self.FourierAnalysis()
        
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

        print("... aggregatedError done")
    
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
        
        print("... plotFields done")
        
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
        
        print("... plotErrorFields done")        
    
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

        print(f"... gradientFields{data_type.upper()} done")
        
    def FourierAnalysis(self):
        plot_path = os.path.join(self.caseTypePath,"plot_Fourier_analysis","")
        os.makedirs(plot_path,exist_ok=True)
        
        ## 1D FT
        for comp, (comp_pred, comp_true) in enumerate(zip(np.moveaxis(self.pred,0,1),np.moveaxis(self.true,0,1))):
                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.semilogy(np.mean(np.mean(np.abs(np.fft.rfft(comp_true,axis=1)),axis=2),axis=0),label="reference")
                plt.semilogy(np.mean(np.mean(np.abs(np.fft.rfft(comp_pred,axis=1)),axis=2),axis=0),label="predicted")
                plt.xlabel("Wavenumber")
                plt.ylabel("Energy")
                plt.title(f"FT magnitudes of rows \naveraged over cols and {comp_true.shape[0]} test cases")
                plt.subplot(1,2,2)
                plt.semilogy(np.mean(np.mean(np.abs(np.fft.rfft(comp_true,axis=2)),axis=1),axis=0),label="reference")
                plt.semilogy(np.mean(np.mean(np.abs(np.fft.rfft(comp_pred,axis=2)),axis=1),axis=0),label="predicted")
                plt.xlabel("Wavenumber")
                plt.title(f"FT magnitudes of cols \naveraged over rows and {comp_true.shape[0]} test cases")
                plt.tight_layout()
                plt.legend()
                plt.savefig(os.path.join(plot_path,f"log_all_{comp+1}.png"),dpi=300,transparent=True)
                
                plt.clf()
                plt.subplot(1,2,1)
                plt.plot(np.mean(np.mean(np.abs(np.fft.rfft(comp_true,axis=1)),axis=2),axis=0),label="reference")
                plt.plot(np.mean(np.mean(np.abs(np.fft.rfft(comp_pred,axis=1)),axis=2),axis=0),label="predicted")
                plt.xlabel("Wavenumber")
                plt.ylabel("Energy")
                plt.title(f"FT magnitudes of rows \naveraged over cols and {comp_true.shape[0]} test cases")
                plt.subplot(1,2,2)
                plt.plot(np.mean(np.mean(np.abs(np.fft.rfft(comp_true,axis=2)),axis=1),axis=0),label="reference")
                plt.plot(np.mean(np.mean(np.abs(np.fft.rfft(comp_pred,axis=2)),axis=1),axis=0),label="predicted")
                plt.xlabel("Wavenumber")
                plt.title(f"FT magnitudes of cols \naveraged over rows and {comp_true.shape[0]} test cases")
                plt.tight_layout()
                plt.legend()
                plt.savefig(os.path.join(plot_path,f"all_{comp+1}.png"),dpi=300,transparent=True)
                plt.close()
        
        ## 2D FT      
        for comp, (comp_pred, comp_true) in enumerate(zip(np.moveaxis(self.pred,0,1),np.moveaxis(self.true,0,1))):
                plt.figure(figsize=(5,5))
                ft = np.abs(np.fft.rfft2(comp_true))
                ft = np.mean(ft,axis=0)
                vmin=np.min(ft)
                vmax=np.max(ft)
                plt.imshow(ft,vmin=vmin,vmax=vmax,cmap="Greys")
                plt.colorbar()
                plt.title(f"2D FT \naveraged over {comp_true.shape[0]} test cases.")
                plt.savefig(os.path.join(plot_path,f"2d_true_all_{comp+1}.png"),dpi=300,transparent=True)
                
                plt.clf()
                ft = np.abs(np.fft.rfft2(comp_pred))
                ft = np.mean(ft,axis=0)
                plt.imshow(ft,vmin=vmin,vmax=vmax,cmap="Greys")
                plt.colorbar()
                plt.title(f"2D FT \naveraged over {comp_pred.shape[0]} test cases.")
                plt.savefig(os.path.join(plot_path,f"2d_pred_all_{comp+1}.png"),dpi=300,transparent=True)
                
                plt.clf()
                ft = np.abs(np.fft.rfft2(comp_true))
                ft = np.mean(ft,axis=0)
                plt.imshow(ft,norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap="Greys")
                plt.colorbar()
                plt.title(f"2D FT \naveraged over {comp_true.shape[0]} test cases.")
                plt.savefig(os.path.join(plot_path,f"2d_true_log_all_{comp+1}.png"),dpi=300,transparent=True)
                
                plt.clf()
                ft = np.abs(np.fft.rfft2(comp_pred))
                ft = np.mean(ft,axis=0)
                plt.imshow(ft,norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap="Greys")
                plt.colorbar()
                plt.title(f"2D FT \naveraged over {comp_pred.shape[0]} test cases.")
                plt.savefig(os.path.join(plot_path,f"2d_pred_log_all_{comp+1}.png"),dpi=300,transparent=True)
                plt.close()
        
        ## 2D FT discrepancies
        for comp, (comp_pred, comp_true) in enumerate(zip(np.moveaxis(self.pred,0,1),np.moveaxis(self.true,0,1))):
                plt.figure(figsize=(5,5))
                
                # discrepancy = np.abs(np.abs(np.fft.rfft2(comp_true))-np.abs(np.fft.rfft2(comp_pred)))
                discrepancy = np.abs( (np.abs(np.fft.rfft2(comp_true))-np.abs(np.fft.rfft2(comp_pred))) / np.abs(np.fft.rfft2(comp_true)) )
                mean_discrepancy = np.mean(discrepancy,axis=0)
                plt.imshow(mean_discrepancy,vmin=np.min(mean_discrepancy),vmax=np.max(mean_discrepancy),cmap="Greys")
                plt.colorbar()
                plt.title(f"Relative discrepancy in 2D FT \naveraged over {comp_true.shape[0]} test cases.")
                plt.savefig(os.path.join(plot_path,f"2d_all_{comp+1}.png"),dpi=300,transparent=True)
                
                plt.clf()
                # discrepancy = np.abs(np.abs(np.fft.rfft2(comp_true))-np.abs(np.fft.rfft2(comp_pred)))
                discrepancy = np.abs( (np.abs(np.fft.rfft2(comp_true))-np.abs(np.fft.rfft2(comp_pred))) / np.abs(np.fft.rfft2(comp_true)) )
                mean_discrepancy = np.mean(discrepancy,axis=0)
                plt.imshow(mean_discrepancy,norm=colors.LogNorm(vmin=np.min(mean_discrepancy),vmax=np.max(mean_discrepancy)),cmap="Greys")
                plt.colorbar()
                plt.title(f"Relative discrepancy in 2D FT \naveraged over {comp_true.shape[0]} test cases.")
                plt.savefig(os.path.join(plot_path,f"2d_log_all_{comp+1}.png"),dpi=300,transparent=True)
                plt.close()
                
    def periodicityCondition(self):
        # TODO
        pass
        
    def mechEquilibriumCondition(self):
        # TODO
        pass
    
    def plotMeanErrorFields(self):
        # TODO
        pass
    
    def plotErrorHistograms(self):
        # TODO
        pass
    
    def plotValueHistograms(self):
        # TODO
        pass
       
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
    