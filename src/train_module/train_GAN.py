import sys
import torch
import os
import time
from modelGAN import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss
from utilities import ImportDataset, CountParameters, MinMaxScaling, GaussianScaling, LossPlotsGAN
from collections import Counter
from torch.nn import L1Loss
import os, psutil

## parameters
exp_name = "GAN_UNet_256_nontriv_LAMBDA_100_batch_2_Discriminator_UNet-enc"
gen_model_name = "UNet"
train_val_test_split = [800,100,100]
g_LAMBDA = 100
num_heads = 5

## training parameters
# continued = ArgumentCheck(sys.argv[1:])
continued=False
epoch_goal=200
learning_rate=0.001
gamma=0.5
step_size=100
batch_size=2
## !!!!!! need to divide the loss by batch size if batch_size != 1

## make directories
main_path         = "../"
path_data         = os.path.join(main_path,  "../../article1_FNO_UNet/data/elasto_plastic/256/")
path_save_model   = os.path.join(main_path, f"saved_models/GAN/UNet/{exp_name}/")
path_training_log = os.path.join(main_path, f"training_log/GAN/UNet/{exp_name}/")
path_loss_plot    = os.path.join(path_training_log, "loss_plot/")
os.makedirs(path_training_log,exist_ok=True)
os.makedirs(path_loss_plot,exist_ok=True)
os.makedirs(path_save_model,exist_ok=True)

## continued training or new training
if continued == True:
	gcheckpoint = torch.load(path_save_model+"generator.pt")
	dcheckpoint = torch.load(path_save_model+"discriminator.pt")
	epoch_completed   = gcheckpoint["epoch_completed"]
	g_loss_training   = gcheckpoint["training_loss"]
	g_loss_validation = gcheckpoint["validation_loss"]
	d_loss_training   = dcheckpoint["training_loss"]
	logfile = open(path_training_log+f"{exp_name}.txt", "a")
else:
	epoch_completed = 0
	g_loss_training   = []
	g_loss_validation = []
	d_loss_training   = []
	logfile = open(path_training_log+f"{exp_name}.txt", "w")

## import dataset
train_data, val_data, _ = ImportDataset(path_data, train_val_test_split, num_heads)

## define normalizers based on training data
x_normalizer = MinMaxScaling(train_data["input"])
y_normalizer = MinMaxScaling(train_data["output"])
# x_normalizer = GaussianScaling(train_data["input"])
# y_normalizer = GaussianScaling(train_data["output"])
x_train      = x_normalizer.encode(train_data["input"])
y_train 	 = y_normalizer.encode(train_data["output"])
x_val 		 = x_normalizer.encode(val_data["input"])
y_val   	 = y_normalizer.encode(val_data["output"])

print(y_train.shape)
print(y_val.shape)

## define dataloaders, models, and optimizers
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

g_model = Generator(model=gen_model_name, num_heads=num_heads)
d_model = Discriminator(version="UNet-enc",num_heads=num_heads)
if epoch_completed == 0:
	with open("train_GAN.py","r") as input:
		for line in input:
			logfile.write(line)
	logfile.write("\n----------------------\n")
	with open("modelGAN.py","r") as input:
		for line in input:
			logfile.write(line)
	logfile.write("\n----------------------\n")
	CountParameters(g_model)
	CountParameters(d_model)

g_optimizer = torch.optim.Adam(g_model.parameters(), lr=learning_rate, weight_decay=1e-4)
g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=step_size, gamma=gamma)
d_optimizer = torch.optim.Adam(d_model.parameters(), lr=learning_rate, weight_decay=1e-4)
d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=step_size, gamma=gamma)

if continued == True:
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_optimizer.load_state_dict(gcheckpoint['optimizer_state_dict'])
	g_scheduler.load_state_dict(gcheckpoint['scheduler_state_dict'])
	d_model.load_state_dict(dcheckpoint['model_state_dict'])
	d_optimizer.load_state_dict(dcheckpoint['optimizer_state_dict'])
	d_scheduler.load_state_dict(dcheckpoint['scheduler_state_dict'])

## train the model
'''
	here, the validation loss is computed based on MAE
'''
Loss = L1Loss()
ep = epoch_completed + 1

while(ep <= epoch_goal + epoch_completed):
	g_model.train()
	d_model.train()

	g_loss_epoch = {"total_loss":0,"gan_loss":0,"L1_loss":0}
	d_loss_epoch = {"total_loss":0,"gan_generated_loss":0,"gan_real_loss":0}

	t1 = time.time()
	for batch_x,batch_target in train_loader:
		## training discriminator
		g_model.eval()
		d_model.train()
		d_optimizer.zero_grad()
		generated_fake 			= g_model(batch_x)
		d_output_generated_fake = d_model(batch_x,generated_fake)
		d_output_real 			= d_model(batch_x,batch_target)
		d_loss 					= DiscriminatorLoss(d_output_generated_fake,d_output_real)
		
		d_loss_item = {k: v.detach().numpy() for (k,v) in d_loss.items()}
		d_loss_epoch = dict(Counter(d_loss_epoch) + Counter(d_loss_item))
		
		d_loss["total_loss"].backward()
		d_optimizer.step()


		## training generator
		g_model.train()
		d_model.eval()
		g_optimizer.zero_grad()
		generated_fake 			= g_model(batch_x)
		d_output_generated_fake = d_model(batch_x,generated_fake)
		g_loss 					= GeneratorLoss(generated_fake, batch_target, d_output_generated_fake, LAMBDA=g_LAMBDA)
		
		g_loss_item = {k: v.detach().numpy() for (k,v) in g_loss.items()}
		g_loss_epoch = dict(Counter(g_loss_epoch) + Counter(g_loss_item))
		
		g_loss["total_loss"].backward()
		g_optimizer.step()


	g_scheduler.step()
	d_scheduler.step()
	t2 = time.time()
	
	## average the losses over all batches
	g_loss_epoch = {k: v/len(train_loader) for (k,v) in g_loss_epoch.items()}
	d_loss_epoch = {k: v/len(train_loader) for (k,v) in d_loss_epoch.items()}
	g_loss_training.append(g_loss_epoch)
	d_loss_training.append(d_loss_epoch)

	## computing validation loss
	g_model.eval()
	loss = 0
	for x,target in val_loader:
		g_output = g_model(x).detach()
		g_output = y_normalizer.decode(g_output).detach()
		target   = y_normalizer.decode(target)
		loss += Loss(g_output,target).item()/1e6
	g_loss_validation.append( loss/len(val_loader))

	logfile.write(f"\nEpoch: {ep} \t Time(s): {t2-t1} \t g_train_loss: {g_loss_epoch['total_loss']} \t d_train_loss: {d_loss_epoch['total_loss']} \t g_val_loss: {g_loss_validation[-1]}")
	print(f"{ep} \t t: {t2-t1} \t g_train_loss: {g_loss_epoch['total_loss']} \t d_train_loss: {d_loss_epoch['total_loss']} \t g_val_loss: {g_loss_validation[-1]} \t MEM:{round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3,2)}Gb", flush=True)
	ep+=1

## save the model
g_model.train()
torch.save(
	{
		'epoch_completed': ep-1,
		'model_state_dict': g_model.state_dict(),
		'optimizer_state_dict': g_optimizer.state_dict(),
		'scheduler_state_dict': g_scheduler.state_dict(),
		'training_loss': g_loss_training,
		'validation_loss': g_loss_validation
	},
	path_save_model+"generator.pt")

d_model.train()
torch.save(
	{
		'epoch_completed': ep-1,
		'model_state_dict': d_model.state_dict(),
		'optimizer_state_dict': d_optimizer.state_dict(),
		'scheduler_state_dict': d_scheduler.state_dict(),
		'training_loss': d_loss_training
	},
	path_save_model+"discriminator.pt")

logfile.close()
LossPlotsGAN(g_loss_training,g_loss_validation,d_loss_training,path_loss_plot,exp_name)
