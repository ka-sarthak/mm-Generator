import sys
import torch
import os, psutil
import time
from generator_models.UNet import UNet
from utilities import ImportDataset, CountParameters, GaussianScaling, MinMaxScaling, ArgumentCheck, LossPlots
from torch.nn import L1Loss 

UNet_version = "standard"
exp_name = f"nonGAN_UNet_{UNet_version}_256_nontriv_from_dec_1"
# continued = ArgumentCheck(sys.argv[1:])
continued = False
num_heads = 5

## training parameters
train_val_test_split = [800,100,100]
epoch_goal=200
kernel = 9
learning_rate=0.001
gamma=0.5
step_size=100
batch_size = 16 # change 1,2,4,8,16,32,64

## make directories
main_path         = "../"
path_data         = os.path.join(main_path,  "../../article1_FNO_UNet/data/elasto_plastic/256/")
path_save_model   = os.path.join(main_path,  "saved_models/nonGAN/UNet/")
path_training_log = os.path.join(main_path, f"training_log/nonGAN/UNet/{exp_name}/")
path_loss_plot    = os.path.join(path_training_log, "loss_plot/")
os.makedirs(path_training_log,exist_ok=True)
os.makedirs(path_loss_plot,exist_ok=True)
os.makedirs(path_save_model,exist_ok=True)

## continued training or new training
if continued == True:
	gcheckpoint = torch.load(path_save_model+f"{exp_name}.pt")
	epoch_completed   = gcheckpoint["epoch_completed"]
	g_loss_training   = gcheckpoint["training_loss"]
	g_loss_validation = gcheckpoint["validation_loss"]
	logfile = open(path_training_log + f"{exp_name}.txt", "a")
else:
	epoch_completed = 0
	g_loss_training   = []
	g_loss_validation = []
	logfile = open(path_training_log + f"{exp_name}.txt", "w")

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
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=train_val_test_split[1], shuffle=False)

g_model = UNet(kernel=kernel, in_channels=3, out_channels=num_heads, version=UNet_version)
if epoch_completed == 0:
	with open("train_nonGAN.py","r") as input:
		for line in input:
			logfile.write(line)
	logfile.write("\n----------------------\n")
	with open("generator_models/UNet.py","r") as input:
		for line in input:
			logfile.write(line)
	logfile.write("\n----------------------\n")
	CountParameters(g_model,logfile)

g_optimizer = torch.optim.Adam(g_model.parameters(), lr=learning_rate, weight_decay=1e-4)
g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=step_size, gamma=gamma)

if continued == True:
	g_model.load_state_dict(gcheckpoint['model_state_dict'])
	g_optimizer.load_state_dict(gcheckpoint['optimizer_state_dict'])
	g_scheduler.load_state_dict(gcheckpoint['scheduler_state_dict'])

'''
	training and validation loss are MAE
'''
Loss = L1Loss()
ep = epoch_completed + 1

while(ep <= epoch_goal + epoch_completed):
	g_model.train()

	t1 = time.time()
	g_loss_epoch = 0
	for batch_x,batch_target in train_loader:
		g_optimizer.zero_grad()
		
		g_output 	 = g_model(batch_x)
		g_loss_batch = Loss(g_output, batch_target)
		g_loss_epoch +=  g_loss_batch.item()
		g_loss_batch.backward()
		
		g_optimizer.step()
		
	g_scheduler.step()
	t2 = time.time()
	
	## average the losses over all batches
	g_loss_epoch /= len(train_loader)
	g_loss_training.append(g_loss_epoch)

	## computing validation loss
	g_model.eval()
	loss = 0
	for x,target in val_loader:
		g_output = g_model(x).detach()
		g_output = y_normalizer.decode(g_output).detach()
		target   = y_normalizer.decode(target)
		loss    += Loss(g_output,target).item()/1e6
	g_loss_validation.append( loss/len(val_loader))

	logfile.write(f"\nEpoch: {ep}\t Time(s): {t2-t1} \t g_train_loss: {g_loss_epoch}\t g_val_loss: {g_loss_validation[-1]}")
	print(f"{ep} \t t:{t2-t1} \t g_train_loss:{g_loss_epoch} \t g_val_loss:{g_loss_validation[-1]} \t MEM: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}", flush=True)
	
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
	path_save_model+f"{exp_name}.pt")

logfile.close()
LossPlots(g_loss_training,g_loss_validation,path_loss_plot,exp_name)
