import os
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.config_module import config

def lossPlots(gen_train_loss,gen_val_loss,save_path):
	exp_name = config["experiment"]["name"]
	
	plt.figure()
	plt.title(f"Generator training loss {exp_name}")
	plt.xlabel("epochs")
	plt.plot(gen_train_loss, label="generator_train_loss")
	plt.legend()
	plt.savefig(f"{save_path}/gen_train_loss.png")
	plt.close()

	plt.figure()
	plt.title(f"Generator validation loss {exp_name}")
	plt.xlabel("epochs")
	plt.plot(gen_val_loss, label="generator_val_loss")
	plt.legend()
	plt.savefig(f"{save_path}/gen_val_loss.png")
	plt.close()

	plt.figure()
	plt.title(f"G training-validation loss {exp_name}")
	plt.xlabel("epochs")
	plt.plot(gen_train_loss, label="generator_train_loss")
	plt.plot(gen_val_loss, label="generator_val_loss")
	plt.legend()
	plt.savefig(f"{save_path}/gen_train_val_loss.png")
	plt.close()

def lossPlotsGAN(gen_train_loss,gen_val_loss,disc_train_loss,save_path):
	exp_name = config["experiment"]["name"]
	save_path = save_path + "/loss_plots"
	os.makedirs(save_path,exist_ok=True)
	
	plt.figure()
	plt.title(f"Generator training loss {exp_name}")
	plt.xlabel("epochs")
	plt.plot([i for i in map(lambda x: x['total_loss'],gen_train_loss)], label="total_loss")
	plt.plot([i for i in map(lambda x: x['gan_loss'],gen_train_loss)], label="gan_loss")
	plt.plot([i for i in map(lambda x: x['L1_loss'],gen_train_loss)], label="L1_loss")
	plt.legend()
	plt.savefig(f"{save_path}/gen_train_loss.png")
	plt.close()

	if disc_train_loss != None:
		plt.figure()
		plt.title(f"Discriminator training loss {exp_name}")
		plt.xlabel("epochs")
		plt.plot([i for i in map(lambda x: x['total_loss'],disc_train_loss)], label="total_loss")
		plt.plot([i for i in map(lambda x: x['gan_real_loss'],disc_train_loss)], label="real_gan_loss")
		plt.plot([i for i in map(lambda x: x['gan_generated_loss'],disc_train_loss)], label="gen_gan_loss")
		plt.legend()
		plt.savefig(f"{save_path}/disc_train_loss.png")
		plt.close()

		plt.figure()
		plt.title(f"G-D training loss {exp_name}")
		plt.xlabel("epochs")
		plt.plot([i for i in map(lambda x: x['total_loss'],gen_train_loss)], label="generator_gan_loss")
		plt.plot([i for i in map(lambda x: x['total_loss'],disc_train_loss)], label="discriminator_gan_loss")
		plt.legend()
		plt.savefig(f"{save_path}/gd_train_loss.png")
		plt.close()

	plt.figure()
	plt.title(f"Generator validation loss {exp_name}")
	plt.xlabel("epochs")
	plt.plot(gen_val_loss, label="generator_val_loss")
	plt.legend()
	plt.savefig(f"{save_path}/gen_val_loss.png")
	plt.close()
 
def plot(data,vmin,vmax,cmap,path_and_name):
        '''
            generate the plots at the plot path
        '''
        plt.set_cmap(cmap)
        plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        
        ax = plt.subplot()
        if vmin*vmax < 0:
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = ax.imshow(data,norm=norm)
        else:
            im = ax.imshow(data,vmin=vmin,vmax=vmax)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="8%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, shrink=0.5, ticks=[vmin,0,vmax])
        cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()], font="Times New Roman",fontsize=20)
  
        plt.savefig(f"{path_and_name}.png", format="png", dpi=300, bbox_inches=0, transparent=True)
        plt.close()
        
def plotAllChannels(data,cmap,path_and_name,rescaling):
    '''
		Given the data with channels c, such that the dimensions are (c,w,h),
		plot the subplots containing a matrix of plots for each channel
		Remark: a subplot will be empty and transparent if all the values are same (most probably 0)
    '''
    cols = 6
    rows = int(np.ceil(data.shape[0]/cols))
    plt.figure(figsize=(cols*2,rows*2))
    
    for i, channel in enumerate(data):
        if rescaling == True:
            channel = (channel-torch.min(channel))/(torch.max(channel)-torch.min(channel))
        plt.subplot(rows,cols,i+1)
        plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        plt.imshow(channel,vmin=torch.min(channel),vmax=torch.max(channel),cmap=cmap)
    
    plt.tight_layout(pad=0,h_pad=0,w_pad=0)
    plt.savefig(f"{path_and_name}.png", format="png", dpi=500, bbox_inches=0, transparent=True)
    plt.close()

def gradientImg(img):
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
