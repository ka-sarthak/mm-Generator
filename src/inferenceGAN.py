import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model import Generator
from utilities import ImportDataset, NMAE, GaussianNormalizer, LossPlots

torch.set_num_threads(1)
torch.manual_seed(0)

exp_name = "UNet_256_P11_separableConv"
os.makedirs(f"./testing/{exp_name}/fields/",exist_ok=True)
save_path = f"./testing/{exp_name}"

## import dataset
data_path = "../../article1_FNO_UNet/data/elasto_plastic/256/"
train_val_test_split = [800,100,100]
train_data, val_data, test_data = ImportDataset(data_path, train_val_test_split)

## define normalizers based on training data
x_normalizer = GaussianNormalizer(train_data["input"])
y_normalizer = GaussianNormalizer(train_data["output"])
x_test = x_normalizer.encode(test_data["input"])
y_test = test_data["output"]

batch_size = 1

## define dataloaders, load GENERATOR model
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)

g_model = Generator(model="UNet",num_heads=1)
gcheckpoint = torch.load(f"./saved_models/generator/{exp_name}.pt")
dcheckpoint = torch.load(f"./saved_models/discriminator/{exp_name}.pt")

## plot loss plots
gen_train_loss 	= gcheckpoint['training_loss']
gen_val_loss 	= gcheckpoint['validation_loss']
disc_train_loss = dcheckpoint['training_loss']
LossPlots(gen_train_loss,gen_val_loss,disc_train_loss,save_path,exp_name)

# exit()
g_model.load_state_dict(gcheckpoint['model_state_dict'])
g_model.eval()

## evaluate the model
'''
	here, the validation loss is computed based on MAE
'''

g_test_loss_list = []
time_list = []
count = 0
with open(f"./testing/{exp_name}/log.txt","w") as f:
	for batch_x,batch_target in test_loader:
		count+=1

		t1 = time.time()
		y_pred = g_model(batch_x)
		y_pred = y_normalizer.decode(y_pred)
		t2 = time.time()
		f.writelines(f"{count}: \t%NMAE = {NMAE(batch_target,y_pred).detach().numpy()*100}\n")
		g_test_loss_list.append(NMAE(batch_target,y_pred).detach().numpy()*100)
		time_list.append(t2-t1)

		y_pred = y_pred.detach().numpy().squeeze()/1e6
		batch_target = batch_target.detach().numpy().squeeze()/1e6

		
		### stress fields
		plt.set_cmap('viridis')
		vmax = np.max([y_pred,batch_target])
		vmin = np.min([y_pred,batch_target])
		fig, axis = plt.subplots(nrows=1, ncols=2)
		axis[0].title.set_text('Label')
		axis[1].title.set_text('Prediction')
		im1 = axis[0].imshow(batch_target,vmin=vmin,vmax=vmax)
		axis[0].tick_params(left=False,
					bottom=False,
					labelleft=False,
					labelbottom=False)	
		im2 = axis[1].imshow(y_pred,vmin=vmin,vmax=vmax)
		axis[1].tick_params(left=False,
					bottom=False,
					labelleft=False,
					labelbottom=False)	

		cbar = fig.colorbar(im2, ax=[axis[0], axis[1]],shrink=0.5, ticks=[vmin,0,vmax], location='bottom')
		# cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
		plt.savefig(f"./testing/{exp_name}/fields/{count}_P11.png", format="png")
		plt.close()
		

		### error fields
		vmax = np.max(batch_target-y_pred)
		vmin = np.min(batch_target-y_pred)
		try:
			norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
			plt.set_cmap('seismic')
			ax = plt.subplot()
			im = ax.imshow((batch_target-y_pred), norm=norm)
			plt.tick_params(left=False,
					bottom=False,
					labelleft=False,
					labelbottom=False)
		except ValueError:
			plt.set_cmap('seismic')
			ax = plt.subplot()
			im = ax.imshow((batch_target-y_pred), norm=norm)
			plt.tick_params(left=False,
					bottom=False,
					labelleft=False,
					labelbottom=False)				
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="8%", pad=0.1)
		cbar = plt.colorbar(im, cax=cax, shrink=0.5, ticks=[vmin,0,vmax])
		# cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
		plt.savefig(f"./testing/{exp_name}/fields/{count}_err_P11.png", format="png")
		plt.close()


	f.writelines(f"\nAverage %NMAE = {np.mean(g_test_loss_list)}")
	f.writelines(f"\nAverage time (s) = {np.mean(np.array(time_list))}")