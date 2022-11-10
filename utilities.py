import torch.nn as nn

def receptive_field(output, stride, ksize):
	"""
		output field size is computed by Floor( (in+2p-ksize)/stride + 1)
		therefore, in = (out-1)*stride -2p + ksize
	"""
	return (output-1)*stride+ksize

def generator_loss(LAMBDA,disc_generated_field, generated_output, target_field):
	BCElogits = nn.BCEWithLogitsLoss()
	L1loss = nn.L1loss()
	
	L1 = L1loss(generated_output,target_field)
	g = BCElogits(torch.ones_like(disc_generated_field),disc_generated_field)
	
	gen_loss = g + L1*LAMBDA
	
	return gen_loss, g, L1

def discriminator_loss(disc_generated_field,disc_target_field):
	BCElogits = nn.BCEWithLogitsLoss()
	
	generated_loss = BCElogits(torch.zeros_like(disc_generated_field),disc_generated_field)
	real_loss = BCElogits(torch.ones_like(disc_target_field),disc_target_field)

	disc_loss = generated_loss+real_loss

	return disc_loss, generated_loss, real_loss
