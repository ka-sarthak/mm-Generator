import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from generator_models.FNO import FNOBlock2d, FNOBlock2d_from_firstFL, FNOBlock2d_from_thirdFL
from utilities import ImportDataset, NMAE,MAE, GaussianScaling, MinMaxScaling, LossPlots

torch.set_num_threads(1)
torch.manual_seed(0)

exp_name = "nonGAN_FNO_256_nontriv_from_output"

## parameters
train_val_test_split = [800,100,100]
batch_size = 1

## FNO params
num_heads = 5
modes 	= 20
width 	= 32 

## paths
main_path = "../"
path_data 		= os.path.join(main_path, "../../article1_FNO_UNet/data/elasto_plastic/256/")
path_save_model = os.path.join(main_path,f"saved_models/nonGAN/FNO/{exp_name}.pt")
path_save 		= os.path.join(main_path,f"testing/nonGAN/FNO/{exp_name}/")
path_fields 	= os.path.join(path_save, "fields/")
path_plots		= os.path.join(path_save, "loss_plots/")
os.makedirs(path_fields,exist_ok=True)
os.makedirs(path_plots,exist_ok=True)


## import dataset
train_data, val_data, test_data = ImportDataset(path_data, train_val_test_split,num_heads=num_heads)

## define normalizers based on training data
# x_normalizer = MinMaxScaling(train_data["input"])
# y_normalizer = MinMaxScaling(train_data["output"])
x_normalizer = GaussianScaling(train_data["input"])
y_normalizer = GaussianScaling(train_data["output"])
x_test = x_normalizer.encode(test_data["input"])
y_test = test_data["output"]


## define dataloaders, load GENERATOR model
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)

g_model = FNOBlock2d(modes1=modes, modes2=modes, width=width, num_heads=num_heads)
# g_model = FNOBlock2d_from_firstFL(modes1=modes, modes2=modes, width=width, num_heads=num_heads)
# g_model = FNOBlock2d_from_thirdFL(modes1=modes, modes2=modes, width=width, num_heads=num_heads)
gcheckpoint = torch.load(path_save_model)

## plot loss plots
gen_train_loss 	= gcheckpoint['training_loss']
gen_val_loss 	= gcheckpoint['validation_loss']
LossPlots(gen_train_loss,gen_val_loss,path_plots,exp_name)

# exit()
g_model.load_state_dict(gcheckpoint['model_state_dict'])
g_model.eval()

## evaluate the model
'''
	here, the validation loss is computed based on MAE
'''

g_test_loss1_list = []
g_test_loss2_list = []
time_list = []
count = 0
with open(path_save+"log.txt","w") as f:
	for batch_x,batch_target in test_loader:
		count+=1
		t1 = time.time()
		y_pred = g_model(batch_x)
		y_pred = y_normalizer.decode(y_pred)
		t2 = time.time()
		
		y_pred = y_pred.detach()/1e6
		batch_target = batch_target.detach()/1e6

		loss1 = [MAE(batch_target[:,i,:,:],y_pred[:,i,:,:]).numpy()*1 for i in range(num_heads)]
		loss2 = [NMAE(batch_target[:,i,:,:],y_pred[:,i,:,:]).numpy()*100 for i in range(num_heads)]
		f.writelines(f"{count}: \t  MAE\t = {loss1}\n")
		f.writelines(f"   \t%NMAE\t = {loss2}\n")
		g_test_loss1_list.append(loss1)
		g_test_loss2_list.append(loss2)
		time_list.append(t2-t1)

		if count>10: continue
		### stress fields
		
		y_pred = y_pred.numpy()
		batch_target = batch_target.numpy()

		for i in range(num_heads):
			plt.set_cmap('viridis')
			vmax = np.max(batch_target[:,i,:,:])
			vmin = np.min(batch_target[:,i,:,:])
			fig, axis = plt.subplots(nrows=1, ncols=2)
			axis[0].title.set_text('Label')
			axis[1].title.set_text('Prediction')
			im1 = axis[0].imshow(batch_target[:,i,:,:].squeeze(),vmin=vmin,vmax=vmax)
			axis[0].tick_params(left=False,
						bottom=False,
						labelleft=False,
						labelbottom=False)	
			im2 = axis[1].imshow(y_pred[:,i,:,:].squeeze(),vmin=vmin,vmax=vmax)
			axis[1].tick_params(left=False,
						bottom=False,
						labelleft=False,
						labelbottom=False)	

			cbar = fig.colorbar(im2, ax=[axis[0], axis[1]],shrink=0.5, ticks=[vmin,0,vmax], location='bottom')
			# cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
			plt.savefig(path_fields+f"{count}_{i}.png", format="png")
			plt.close()
		

			### error fields
			vmax = np.max(batch_target[:,i,:,:]-y_pred[:,i,:,:])
			vmin = np.min(batch_target[:,i,:,:]-y_pred[:,i,:,:])
			try:
				norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
				plt.set_cmap('seismic')
				ax = plt.subplot()
				im = ax.imshow((batch_target[:,i,:,:]-y_pred[:,i,:,:]).squeeze(), norm=norm)
				plt.tick_params(left=False,
						bottom=False,
						labelleft=False,
						labelbottom=False)
			except ValueError:
				plt.set_cmap('seismic')
				ax = plt.subplot()
				im = ax.imshow((batch_target[:,i,:,:]-y_pred[:,i,:,:]).squeeze(), norm=norm)
				plt.tick_params(left=False,
						bottom=False,
						labelleft=False,
						labelbottom=False)				
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="8%", pad=0.1)
			cbar = plt.colorbar(im, cax=cax, shrink=0.5, ticks=[vmin,0,vmax])
			# cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
			plt.savefig(path_fields+f"{count}_err_{i}.png", format="png")
			plt.close()


	f.writelines(f"\nAverage MAE = {np.mean(g_test_loss1_list,axis=0)}")
	f.writelines(f"\nAverage %NMAE = {np.mean(g_test_loss2_list,axis=0)}")
	f.writelines(f"\nAverage time (s) = {np.mean(np.array(time_list))}")

torch.set_num_threads(16)
y_pred = g_model(x_test)
y_pred = y_normalizer.decode(y_pred)
print(y_pred.shape)

for comp in range(y_pred.shape[1]):
	
	field_error = torch.abs(y_test[:,comp,:,:]-y_pred[:,comp,:,:])
	field_error = field_error.detach().numpy()
	mean_field_error = np.mean(field_error,axis=0)# taken over all the examples
	vmax = np.max(mean_field_error)
	vmin = np.min(mean_field_error)
	# imdata = np.round(255.*(field_error-vmin)/(vmax-vmin))
	# vcenter = (-vmin/(vmax-vmin))*255.
	# vcenter = (-vmin/(vmax-vmin))
	
	# if vmin<0:
	# 	norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
	# 	# norm = colors.TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=255)
	# 	plt.set_cmap('seismic')
	# 	# im = Image.new("L", size=(xdim,ydim))
	# 	# im.putdata(imdata.flatten())
	# 	ax = plt.subplot()
	# 	im = ax.imshow(mean_field_error, norm=norm)#, vmin=vmin, vmax=vmax, )
	# 	plt.tick_params(left=False,
	# 			bottom=False,
	# 			labelleft=False,
	# 			labelbottom=False)
	# else:
# 	print('except')
	plt.set_cmap('Reds')
	# im = Image.new("L", size=(xdim,ydim))
	# im.putdata(imdata.flatten())
	ax = plt.subplot()
	im = ax.imshow(mean_field_error, vmin=vmin, vmax=vmax)
	plt.tick_params(left=False,
			bottom=False,
			labelleft=False,
			labelbottom=False)

	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="8%", pad=0.1)
	cbar = plt.colorbar(im, cax=cax, shrink=0.5, ticks=[])
	cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)

	# print(f"{vmin}\t{np.mean(field_error)}\t{vmax}")
	plt.title(f"[{vmin},{vmax}]")
	plt.savefig(os.path.join(path_fields,f"err_mean_field_{comp+1}.png"), dpi=400, bbox_inches="tight", transparent=True)
	plt.close()