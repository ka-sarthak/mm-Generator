import sys
import torch
import os, psutil
import time
from collections import Counter
import json 
from torch.nn import L1Loss
from models.generator import Generator
from models.discriminator import Discriminator
from models.ganLoss import generatorLoss, discriminatorLoss
from utils.config_module import config
from utils.data_processing import makePathAndDirectories, importDataset, scaleDataset
from utils.utilities import countParameters
from utils.postprocessing import lossPlotsGAN

def train():
	## configs and paths
	training_config = config["training"]
	exp_config = config["experiment"]

	path_save_model, path_training_log, path_loss_plot = makePathAndDirectories()
	
	## import dataset
	train_data, val_data, _ = importDataset()

	## get scaled dataset and trained Scaler objects
	x_train, y_train, x_val, y_val, x_scaler, y_scaler = scaleDataset(train_data,val_data)
	print("Shape of the training data (X,y): ", x_train.shape, y_train.shape)
	print("Shape of the validation data (X,y): ", x_val.shape, y_val.shape)

	## define dataloaders
	train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=training_config["batchSize"], shuffle=True)
	val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=training_config["trainValTestSplit"][1], shuffle=False)

	## setup logger for continued training or new training
	if training_config["continueTraining"] == True:
		gcheckpoint = torch.load(path_save_model+"generator.pt")
		dcheckpoint = torch.load(path_save_model+"discriminator.pt")
		epoch_completed   = gcheckpoint["epoch_completed"]
		g_loss_training   = gcheckpoint["training_loss"]
		g_loss_validation = gcheckpoint["validation_loss"]
		d_loss_training   = dcheckpoint["training_loss"]
		logfile = open(path_training_log+"training.txt", "a")
	else:
		epoch_completed = 0
		g_loss_training   = []
		g_loss_validation = []
		d_loss_training   = []
		logfile = open(path_training_log+"training.txt", "w")

	## define the generator model, optimizer, scheduler, loss function
	g_model = Generator()
	d_model = Discriminator()
	
	g_optimizer = torch.optim.Adam(g_model.parameters(), lr=training_config["learningRate"], weight_decay=training_config["weightDecay"])
	g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=training_config["stepSize"], gamma=training_config["gamma"])
	d_optimizer = torch.optim.Adam(d_model.parameters(), lr=training_config["learningRate"], weight_decay=training_config["weightDecay"])
	d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=training_config["stepSize"], gamma=training_config["gamma"])
	Loss = L1Loss()
	metric = L1Loss()

	## load states if continued Training
	if training_config["continueTraining"] == True:
		g_model.load_state_dict(gcheckpoint['model_state_dict'])
		g_optimizer.load_state_dict(gcheckpoint['optimizer_state_dict'])
		g_scheduler.load_state_dict(gcheckpoint['scheduler_state_dict'])
		d_model.load_state_dict(dcheckpoint['model_state_dict'])
		d_optimizer.load_state_dict(dcheckpoint['optimizer_state_dict'])
		d_scheduler.load_state_dict(dcheckpoint['scheduler_state_dict'])

	## logging
	if epoch_completed == 0:
		logfile.write("-----------config-----------\n")
		logfile.write(json.dumps(config,indent=4))
		logfile.write("\n--------------------------------\n")
		countParameters(g_model)
		countParameters(d_model)

	## training loop
	ep = epoch_completed + 1
	while(ep <= training_config["epochs"] + epoch_completed):
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
			d_loss 					= discriminatorLoss(d_output_generated_fake,d_output_real)
			
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
			g_loss 					= generatorLoss(generated_fake, batch_target, d_output_generated_fake)
			
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
			g_output = y_scaler.decode(g_output).detach()
			target   = y_scaler.decode(target)
			loss 	+= metric(g_output,target).item()/1e6
		g_loss_validation.append( loss/len(val_loader))

		logfile.write(f"\nEpoch: {ep} \t Time(s): {t2-t1} \t g_train_loss: {g_loss_epoch['total_loss']} \t d_train_loss: {d_loss_epoch['total_loss']} \t g_val_loss: {g_loss_validation[-1]}")
		print(f"{ep} \t t: {t2-t1} \t g_train_loss: {g_loss_epoch['total_loss']} \t d_train_loss: {d_loss_epoch['total_loss']} \t g_val_loss: {g_loss_validation[-1]} \t MEM:{round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3,2)} GB", flush=True)
		
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
	lossPlotsGAN(g_loss_training,g_loss_validation,d_loss_training,path_loss_plot)

	logfile.close()
