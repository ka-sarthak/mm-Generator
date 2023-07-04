import torch
import sys
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from generator_models.UNet import UNet
from generator_models.FNO import FNOBlock2d
from modelGAN import Generator
from utilities import ImportDataset, NMAE,MAE, GaussianScaling, MinMaxScaling, LossPlots, ArgumentCheck
from torch.autograd import Variable

torch.set_num_threads(16)
torch.manual_seed(0)

def post_processing(model, path_save, path_fields, y_test, pred):	
	pred = pred.numpy() / 1e6
	y_test = y_test.numpy() / 1e6
	
	for i in range(pred.shape[1]):
		bound1 = y_test[0,i,0,:]
		bound2 = y_test[0,i,-1,:]
		bound1_pred = pred[0,i,0,:]
		bound2_pred = pred[0,i,-1,:]


		plt.figure()
		plt.plot(bound1,label="b1_DAMASK")
		plt.plot(bound2,label="b2_DAMASK")
		plt.plot(bound1_pred,label=f"b1_{model}")
		plt.plot(bound2_pred,label=f"b2_{model}")
		plt.legend()
		plt.savefig(path_fields + f"{i+1}_b1_b2.jpg",format="jpg", dpi=400, bbox_inches="tight", transparent=True)
		plt.close()

		plt.figure()
		plt.plot(bound1-bound2,label="DAMASK")
		plt.plot(bound1_pred - bound2_pred,label=f"{model}")
		plt.legend()
		plt.savefig(path_fields + f"{i+1}_b1_minus_b2.jpg",format="jpg", dpi=400, bbox_inches="tight", transparent=True)
		plt.close()

		vmin = np.min(y_test[0,i,:,:])
		vmax = np.max(y_test[0,i,:,:])
		fig = plt.figure()
		im1 = plt.imshow(y_test[0,i,:,:].squeeze(),vmin=vmin,vmax=vmax)
		
		cbar = fig.colorbar(im1,shrink=0.5, ticks=[vmin,0,vmax], location='bottom')

		plt.savefig(path_fields + f"{i+1}_DAMASK_field.jpg",format="jpg", dpi=400, bbox_inches="tight", transparent=True)
		plt.close()

if __name__== "__main__":
	experiment = sys.argv[1]
	train_resolution = 256
	model_UNet = "nonGAN_UNet_standard_256_nontriv_from_dec_1"
	# model_UNet = "nonGAN_UNet_standard_64_nontriv_from_dec"
	model_FNO = "nonGAN_FNO_256_nontriv_from_output"
	model_GAN = "GAN_UNet_256_nontriv_LAMBDA_100_batch_2"

	## common parameters
	num_heads = 5
	train_val_test_split = [800,100,100]
	batch_size = 1
	main_path = "../"
	path_train_data 	 = os.path.join(main_path, f"../../article1_FNO_UNet/data/elasto_plastic/{train_resolution}/")
	path_test_data 		 = os.path.join(main_path, f"../../article1_FNO_UNet/data/elasto_plastic/testing/{experiment}/")
	train_data, _, _ = ImportDataset(path_train_data, train_val_test_split,num_heads=num_heads)
	_, _, test_data  = ImportDataset(path_test_data, train_val_test_split,num_heads=num_heads,only_test=True)
	###############################################################
	## UNet
	###############################################################
	kernel = 9
	path_save_model = os.path.join(main_path,f"saved_models/nonGAN/UNet/{model_UNet}.pt")
	path_save 		= os.path.join(main_path,f"testing/{train_resolution}/{experiment}/UNet/periodicity/")
	path_fields 	= os.path.join(path_save, "fields/")
	os.makedirs(path_fields,exist_ok=True)

	x_normalizer = MinMaxScaling(train_data["input"])
	y_normalizer = MinMaxScaling(train_data["output"])
	x_test = x_normalizer.encode(test_data["input"])
	y_test = test_data["output"]

	g_model = UNet(kernel=kernel, in_channels=3, out_channels=num_heads, version="standard")
	gcheckpoint = torch.load(path_save_model)
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_model.eval()
	pred = y_normalizer.decode(g_model(x_test).detach())
	post_processing("UNet", path_save, path_fields, y_test.detach(), pred)
	del(x_normalizer,y_normalizer,x_test,y_test,g_model,gcheckpoint,pred)

	###############################################################
	## FNO 
	###############################################################
	modes 	= 20
	width 	= 32 
	path_save_model = os.path.join(main_path,f"saved_models/nonGAN/FNO/{model_FNO}.pt")
	path_save 		= os.path.join(main_path,f"testing/{train_resolution}/{experiment}/FNO/periodicity/")
	path_fields 	= os.path.join(path_save, "fields/")
	os.makedirs(path_fields,exist_ok=True)

	x_normalizer = GaussianScaling(train_data["input"])
	y_normalizer = GaussianScaling(train_data["output"])
	x_test = x_normalizer.encode(test_data["input"])
	y_test = test_data["output"]

	g_model = FNOBlock2d(modes1=modes, modes2=modes, width=width, num_heads=num_heads)
	gcheckpoint = torch.load(path_save_model)
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_model.eval()
	pred = y_normalizer.decode(g_model(x_test).detach())
	post_processing("FNO", path_save, path_fields, y_test.detach(), pred)
	del(x_normalizer,y_normalizer,x_test,y_test,g_model,gcheckpoint,pred)



	###############################################################
	## GAN
	###############################################################
	kernel = 9
	path_save_model = os.path.join(main_path,f"saved_models/GAN/UNet/{model_GAN}/generator.pt")
	path_save 		= os.path.join(main_path,f"testing/{train_resolution}/{experiment}/GAN-UNet/periodicity/")
	path_fields 	= os.path.join(path_save, "fields/")
	os.makedirs(path_fields,exist_ok=True)

	x_normalizer = MinMaxScaling(train_data["input"])
	y_normalizer = MinMaxScaling(train_data["output"])
	x_test = x_normalizer.encode(test_data["input"])
	y_test = test_data["output"]

	g_model = Generator(model="UNet", num_heads=num_heads)
	gcheckpoint = torch.load(path_save_model)
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_model.eval()
	pred = y_normalizer.decode(g_model(x_test).detach())
	post_processing("cGAN-UNet", path_save, path_fields, y_test.detach(), pred)
	del(x_normalizer,y_normalizer,x_test,y_test,g_model,gcheckpoint,pred)
