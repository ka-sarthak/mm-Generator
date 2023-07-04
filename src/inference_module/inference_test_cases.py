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

def field_error_prediction(model, path_save, path_fields, x_test, y_test, test_loader, g_model, y_normalizer, num_heads):
	g_test_loss1_list = []
	g_test_loss2_list = []
	summed_me_v_pred = np.array([])
	summed_me_v_true = np.array([])
	grad_mean = []
	grad_std = []
	time_list = []
	count = 0
	with open(path_save+"log.txt","w") as f:
		for batch_x,batch_target in test_loader:
			count+=1

			if count>10: continue

			t1 = time.time()
			y_pred = g_model(batch_x)
			y_pred = y_normalizer.decode(y_pred)
			t2 = time.time()

			y_pred = y_pred.detach()/1e9
			batch_target = batch_target.detach()/1e9

			####################
			Vy1 = batch_target.numpy()[0,1,:,:][np.newaxis,:,:] ## uses P22
			Vz1 = batch_target.numpy()[0,2,:,:][np.newaxis,:,:] ## uses P23
			Vy2 = batch_target.numpy()[0,3,:,:][np.newaxis,:,:] ## uses P32
			Vz2 = batch_target.numpy()[0,4,:,:][np.newaxis,:,:] ## uses P33
			# Vy = P32[900+id_case]
			# Vz = P33[900+id_case]
			divP_true1 = calc_div_fft(Vy1, Vz1)
			divP_true2 = calc_div_fft(Vy2, Vz2)


			## finding the norm at each pixel point
			norm_field = np.power(np.power(divP_true1,2) + np.power(divP_true2,2), 0.5) 
			summed_me_v_true = np.append(summed_me_v_true, mae(norm_field))

			# predictions
			Vy1 = y_pred.numpy()[0,1,:,:][np.newaxis,:,:] ## uses P22
			Vz1 = y_pred.numpy()[0,2,:,:][np.newaxis,:,:] ## uses P23
			Vy2 = y_pred.numpy()[0,3,:,:][np.newaxis,:,:] ## uses P32
			Vz2 = y_pred.numpy()[0,4,:,:][np.newaxis,:,:] ## uses P33
			# Vy = P32[id_case]
			# Vz = P33[id_case]
			divP_pred1 = calc_div_fft(Vy1, Vz1)
			divP_pred2 = calc_div_fft(Vy2, Vz2)

			## finding the norm at each pixel point
			norm_field = np.power(np.power(divP_pred1,2) + np.power(divP_pred2,2), 0.5) 

			summed_me_v_pred = np.append(summed_me_v_pred, mae(norm_field))

			####################
			
			y_pred = y_pred/1e6
			batch_target = batch_target/1e6

			loss1 = [MAE(batch_target[:,i,:,:],y_pred[:,i,:,:]).numpy()*1 for i in range(num_heads)]
			loss2 = [NMAE(batch_target[:,i,:,:],y_pred[:,i,:,:]).numpy()*100 for i in range(num_heads)]
			f.writelines(f"{count}: \t  MAE\t = {loss1}\n")
			f.writelines(f"   \t%NMAE\t = {loss2}\n")
			g_test_loss1_list.append(loss1)
			g_test_loss2_list.append(loss2)
			time_list.append(t2-t1)

			if count>0: continue

	# 		y_pred = y_pred.numpy()
	# 		batch_target = batch_target.numpy()

	# 		for i in range(num_heads):
	# 			plt.set_cmap('viridis')
	# 			vmax = np.max(batch_target[:,i,:,:])
	# 			vmin = np.min(batch_target[:,i,:,:])
	# 			fig, axis = plt.subplots(nrows=1, ncols=2)
	# 			axis[0].title.set_text('Label')
	# 			axis[1].title.set_text('Prediction')
	# 			im1 = axis[0].imshow(batch_target[:,i,:,:].squeeze(),vmin=vmin,vmax=vmax)
	# 			axis[0].tick_params(left=False,
	# 						bottom=False,
	# 						labelleft=False,
	# 						labelbottom=False)	
	# 			im2 = axis[1].imshow(y_pred[:,i,:,:].squeeze(),vmin=vmin,vmax=vmax)
	# 			axis[1].tick_params(left=False,
	# 						bottom=False,
	# 						labelleft=False,
	# 						labelbottom=False)	

	# 			cbar = fig.colorbar(im2, ax=[axis[0], axis[1]],shrink=0.5, ticks=[vmin,0,vmax], location='bottom')
	# 			# cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
	# 			plt.savefig(path_fields+f"{count}_{i}.png", dpi=400, bbox_inches=0, transparent=True)
	# 			plt.close()
			

	# 			### error fields
	# 			vmax = np.max(batch_target[:,i,:,:]-y_pred[:,i,:,:])
	# 			vmin = np.min(batch_target[:,i,:,:]-y_pred[:,i,:,:])
	# 			try:
	# 				norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
	# 				plt.set_cmap('seismic')
	# 				ax = plt.subplot()
	# 				im = ax.imshow((batch_target[:,i,:,:]-y_pred[:,i,:,:]).squeeze(), norm=norm)
	# 				plt.tick_params(left=False,
	# 						bottom=False,
	# 						labelleft=False,
	# 						labelbottom=False)
	# 			except ValueError:
	# 				plt.set_cmap('seismic')
	# 				ax = plt.subplot()
	# 				im = ax.imshow((batch_target[:,i,:,:]-y_pred[:,i,:,:]).squeeze(), norm=norm)
	# 				plt.tick_params(left=False,
	# 						bottom=False,
	# 						labelleft=False,
	# 						labelbottom=False)				
	# 			divider = make_axes_locatable(ax)
	# 			cax = divider.append_axes("right", size="8%", pad=0.1)
	# 			cbar = plt.colorbar(im, cax=cax, shrink=0.5, ticks=[vmin,0,vmax])
	# 			# cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
	# 			plt.savefig(path_fields+f"{count}_err_{i}.png", dpi=400, bbox_inches=0, transparent=True)
	# 			plt.close()

	# 			rel_err = np.abs((batch_target[:,i,:,:]-y_pred[:,i,:,:])/batch_target[:,i,:,:])*100.0
	# 			rel_err = rel_err>10

	# 			ax = plt.subplot()
	# 			plt.set_cmap('Greys')
	# 			im = ax.imshow(rel_err.squeeze())
	# 			plt.tick_params(left=False,
	# 						bottom=False,
	# 						labelleft=False,
	# 						labelbottom=False)
	# 			plt.savefig(path_fields+f"{count}_{i}_rel_err_threshold.png", dpi=400, bbox_inches=0, transparent=True)
	# 			plt.close()

	# 			## grad image plotting and statistics
	# 			grad_img = gradient_img(torch.from_numpy(y_pred[:,i,:,:]))
	# 			grad_mean.append(torch.mean(grad_img).item())
	# 			grad_std.append(torch.std(grad_img).item())
	# 			vmax = torch.max(grad_img).item()
	# 			vmin = torch.min(grad_img).item()
				
	# 			try:
	# 				norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
	# 				plt.set_cmap('Reds')
	# 				ax = plt.subplot()
	# 				im = ax.imshow((batch_target[:,i,:,:]-y_pred[:,i,:,:]).squeeze(), norm=norm)
	# 				plt.tick_params(left=False,
	# 						bottom=False,
	# 						labelleft=False,
	# 						labelbottom=False)
	# 			except ValueError:
	# 				plt.set_cmap('Reds')
	# 				ax = plt.subplot()
	# 				im = ax.imshow((batch_target[:,i,:,:]-y_pred[:,i,:,:]).squeeze(), vmin=vmin, vmax=vmax)
	# 				plt.tick_params(left=False,
	# 						bottom=False,
	# 						labelleft=False,
	# 						labelbottom=False)				
	# 			divider = make_axes_locatable(ax)
	# 			cax = divider.append_axes("right", size="8%", pad=0.1)
	# 			cbar = plt.colorbar(im, cax=cax, shrink=0.5, ticks=[vmin,0,vmax])
	# 			# cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
	# 			plt.savefig(path_fields+f"{count}_{i}_gradImg.png", dpi=400, bbox_inches=0, transparent=True)
	# 			plt.close()

		print(np.mean(summed_me_v_true))
		print(np.mean(summed_me_v_pred))

		f.writelines(f"\nAverage MAE = {np.mean(g_test_loss1_list,axis=0)}")
		f.writelines(f"\nAverage %NMAE = {np.mean(g_test_loss2_list,axis=0)}")
		f.writelines(f"\nAverage time (s) = {np.mean(np.array(time_list))}")
		f.writelines(f"\nMean of Gradient mean (s) = {np.mean(np.array(grad_mean))}")
		f.writelines(f"\nMean of Gradient std (s) = {np.mean(np.array(grad_std))}")
		f.writelines(f"\nMech. equi. violation true = {np.mean(summed_me_v_true)}")
		f.writelines(f"\nMech. equi. violation pred = {np.mean(summed_me_v_pred)}")

	# y_pred = g_model(x_test)
	# y_pred = y_normalizer.decode(y_pred)
	# y_pred = y_pred.detach()

	# for i in range(num_heads):
	# 	field_error = (y_test[:,i,:,:]-y_pred[:,i,:,:]).numpy().flatten()/1e6
	# 	if model == "UNet":
	# 		color = "#737373"
	# 	else:
	# 		color = "#FFB900"
	# 	ax = plt.hist(field_error, bins = [i for i in np.arange(-25,25,0.2)], alpha= 1.0, color=color)
	# 	plt.xlim((-25,25))
	# 	plt.ylim((0,120000))
	# 	plt.xticks(font="Times New Roman",fontsize=24)
	# 	plt.yticks([])
	# 	plt.savefig(path_fields + f"error_hist_{i}.png", dpi=400, bbox_inches=0, transparent=True)
	# 	plt.close()

	## hist for tensile and shear components: use 2400000 for 100 cases. 
	# y_test_tensile = y_test[:,[0,1,4],:,:]
	# y_pred_tensile = y_pred[:,[0,1,4],:,:]
	# y_test_shear = y_test[:,[2,3],:,:]
	# y_pred_shear = y_pred[:,[2,3],:,:]

	# field_error = (y_test_tensile[:,:,:,:]-y_pred_tensile[:,:,:,:]).numpy().flatten()/1e6
	# if model == "UNet":
	# 	color = "#737373"
	# else:
	# 	color = "#FFB900"
	# ax = plt.hist(field_error, bins = [i for i in np.arange(-25,25,0.2)], alpha= 1.0, color=color)
	# plt.xlim((-25,25))
	# plt.ylim((0,2400000))
	# plt.xticks(font="Times New Roman",fontsize=24)
	# plt.yticks([])
	# plt.savefig(path_fields + f"error_hist_tensile.png", dpi=400, bbox_inches=0, transparent=True)
	# plt.close()

	# field_error = (y_test_shear[:,:,:,:]-y_pred_shear[:,:,:,:]).numpy().flatten()/1e6
	# if model == "UNet":
	# 	color = "#737373"
	# else:
	# 	color = "#FFB900"
	# ax = plt.hist(field_error, bins = [i for i in np.arange(-25,25,0.2)], alpha= 1.0, color=color)
	# plt.xlim((-25,25))
	# plt.ylim((0,2400000))
	# plt.xticks(font="Times New Roman",fontsize=24)
	# plt.yticks([])
	# plt.savefig(path_fields + f"error_hist_shear.png", dpi=400, bbox_inches=0, transparent=True)
	# plt.close()

	## histograms for stress values
	# field_error = (y_pred_tensile[:,:,:,:]).numpy().flatten()/1e6
	# if model == "UNet":
	# 	color = "#737373"
	# else:
	# 	color = "#FFB900"
	# ax = plt.hist(field_error, bins = [i for i in np.arange(-30,30,0.2)], alpha= 1.0, color=color)
	# plt.xlim((-30,30))
	# plt.ylim((0,400000))
	# plt.xticks(font="Times New Roman",fontsize=24)
	# plt.yticks([])
	# plt.savefig(path_fields + f"stressValue_hist_tensile.png", dpi=400, bbox_inches=0, transparent=True)
	# plt.close()

	# # field_error = (y_test_shear[:,:,:,:]-y_pred_shear[:,:,:,:]).numpy().flatten()/1e6
	# field_error = (y_pred_shear[:,:,:,:]).numpy().flatten()/1e6
	# if model == "UNet":
	# 	color = "#737373"
	# else:
	# 	color = "#FFB900"
	# ax = plt.hist(field_error, bins = [i for i in np.arange(-30,30,0.2)], alpha= 1.0, color=color)
	# plt.xlim((-30,30))
	# plt.ylim((0,400000))
	# plt.xticks(font="Times New Roman",fontsize=24)
	# plt.yticks([])
	# plt.savefig(path_fields + f"stressValue_hist_shear.png", dpi=400, bbox_inches=0, transparent=True)
	# plt.close()

def gradient_img(img):
	img = img.squeeze(0)
	# ten=torch.unbind(img)
	x=img.unsqueeze(0).unsqueeze(0)
	
	a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
	conv1= torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
	conv1.weight= torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
	G_x=conv1(Variable(x)).data.view(1,x.shape[2],x.shape[3])

	b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
	conv2= torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
	conv2.weight= torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
	G_y=conv2(Variable(x)).data.view(1,x.shape[2],x.shape[3])

	G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
	return G

def calc_div_fft(Vx, Vy):
	Nx = np.shape(Vx)[-1]; Ny = Nx
	
	kx = np.fft.fftfreq(Nx).reshape(Nx)*Nx / Nx        # reshaping so that ky multiplies pointwise with y-index in 2D
	ky = np.fft.fftfreq(Ny).reshape(Ny,1)*Ny / Ny
	
	kx[Nx//2] = 0
	ky[Ny//2] = 0             # removing nyquist frequency to reduce noise
	# kx = Nx/(4*dx) * (np.exp(1j*kx_)-1)*(np.exp(1j*ky_)+1)
	# ky = Ny/(4*dy) * (np.exp(1j*kx_)+1)*(np.exp(1j*ky_)-1)
	div_V = np.fft.ifftn((np.fft.fftn(Vx) * kx + np.fft.fftn(Vy) * ky) * 1j * 2. * np.pi)

	return np.real(div_V).squeeze()

def calc_div_fd(Vx,Vy):
	Vx = np.squeeze(Vx)
	Vy = np.squeeze(Vy)
	Vxdx = np.roll(Vx, -1, axis=0) - np.roll(Vx, 1, axis=0) # divide it by step size to make it physical value.
	Vydy = np.roll(Vy, -1, axis=1) - np.roll(Vy, 1, axis=1) # change to spectral based derivation for mech equi. gives close to exact
	div_V = Vxdx + Vydy

	return div_V

def mae(pred):
	return np.mean(np.abs(pred))

def norml2(a):
	return np.sqrt(np.matmul(a,a.T))

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
	path_test_data 		 = os.path.join(main_path, f"../../article1_FNO_UNet/data/elasto_plastic/testing/256_basicFFT/")
	train_data, _, _ = ImportDataset(path_train_data, train_val_test_split,num_heads=num_heads)
	_, _, test_data  = ImportDataset(path_test_data, train_val_test_split,num_heads=num_heads,only_test=True)
	###############################################################
	## UNet
	###############################################################
	kernel = 9
	path_save_model = os.path.join(main_path,f"saved_models/nonGAN/UNet/{model_UNet}.pt")
	path_save 		= os.path.join(main_path,f"testing/{train_resolution}/{experiment}/UNet/")
	path_fields 	= os.path.join(path_save, "fields/")
	os.makedirs(path_fields,exist_ok=True)

	x_normalizer = MinMaxScaling(train_data["input"])
	y_normalizer = MinMaxScaling(train_data["output"])
	x_test = x_normalizer.encode(test_data["input"])
	y_test = test_data["output"]
	test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

	g_model = UNet(kernel=kernel, in_channels=3, out_channels=num_heads, version="standard")
	gcheckpoint = torch.load(path_save_model)
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_model.eval()
	# g_model(x_test)
	field_error_prediction("UNet", path_save, path_fields, x_test, y_test, test_loader, g_model, y_normalizer, num_heads)
	del(g_model,y_normalizer,x_normalizer, test_loader,x_test,y_test,gcheckpoint)

	###############################################################
	## FNO 
	###############################################################
	modes 	= 20
	width 	= 32 
	path_save_model = os.path.join(main_path,f"saved_models/nonGAN/FNO/{model_FNO}.pt")
	path_save 		= os.path.join(main_path,f"testing/{train_resolution}/{experiment}/FNO/")
	path_fields 	= os.path.join(path_save, "fields/")
	os.makedirs(path_fields,exist_ok=True)

	x_normalizer = GaussianScaling(train_data["input"])
	y_normalizer = GaussianScaling(train_data["output"])
	x_test = x_normalizer.encode(test_data["input"])
	y_test = test_data["output"]
	test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

	g_model = FNOBlock2d(modes1=modes, modes2=modes, width=width, num_heads=num_heads)
	gcheckpoint = torch.load(path_save_model)
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_model.eval()
	field_error_prediction("FNO", path_save, path_fields, x_test, y_test, test_loader, g_model, y_normalizer, num_heads)
	del(g_model,y_normalizer,x_normalizer, test_loader,x_test,y_test,gcheckpoint)


	###############################################################
	## GAN
	###############################################################
	kernel = 9
	path_save_model = os.path.join(main_path,f"saved_models/GAN/UNet/{model_GAN}/generator.pt")
	path_save 		= os.path.join(main_path,f"testing/{train_resolution}/{experiment}/GAN-UNet/")
	path_fields 	= os.path.join(path_save, "fields/")
	os.makedirs(path_fields,exist_ok=True)

	x_normalizer = MinMaxScaling(train_data["input"])
	y_normalizer = MinMaxScaling(train_data["output"])
	x_test = x_normalizer.encode(test_data["input"])
	y_test = test_data["output"]
	test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

	g_model = Generator(model="UNet", num_heads=num_heads)
	gcheckpoint = torch.load(path_save_model)
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_model.eval()
	field_error_prediction("cGAN-UNet", path_save, path_fields, x_test, y_test, test_loader, g_model, y_normalizer, num_heads)
	del(g_model,y_normalizer,x_normalizer, test_loader,x_test,y_test,gcheckpoint)