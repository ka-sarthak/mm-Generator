import numpy as np
import torch
from torch.nn import L1Loss, BCELoss
from utils.config_module import config

def generatorLoss(generated_field, target_field, disc_generated_field):
	BCEloss = BCELoss()
	L1loss = L1Loss()
	
	L1 = L1loss(generated_field,target_field)
	g = BCEloss(disc_generated_field,torch.ones_like(disc_generated_field))
	
	gen_loss = g + L1*config["model"]["GAN"]["LAMBDA"]
	
	return {"total_loss":gen_loss, "gan_loss":g, "L1_loss":L1}

def discriminatorLoss(disc_generated_field, disc_target_field):
	BCEloss = BCELoss()
	
	generated_loss = BCEloss(disc_generated_field,torch.zeros_like(disc_generated_field))
	real_loss = BCEloss(disc_target_field,torch.ones_like(disc_target_field))

	disc_loss = (generated_loss+real_loss)*0.5

	return {"total_loss":disc_loss, "gan_generated_loss":generated_loss, "gan_real_loss":real_loss}

