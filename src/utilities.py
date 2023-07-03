import torch
from prettytable import PrettyTable
import scipy.io
import matplotlib.pyplot as plt
import os

def MAE(f1,f2):
	return torch.mean(torch.abs(f1-f2))

def NMAE(f1,f2):
	'''
		Normalized MAE
	'''
	return MAE(f1,f2)/(torch.max(f1)-torch.min(f1))

def ArgumentCheck(argv):
	if not argv:
		continued = False
	elif argv[0] in ["--c","--continue"]:
		continued = True
	else: 
		print("ERROR: Bad arguments. Should be one of {},{--c},{--continue}.")
		exit()
	return continued

def ReceptiveField(output, stride, ksize):
	"""
		output field size is computed by Floor( (in+2p-ksize)/stride + 1)
		therefore, in = (out-1)*stride -2p + ksize
	"""
	return (output-1)*stride+ksize

def CountParameters(model,logfile=None):
		table = PrettyTable(["Modules", "Parameters"])
		total_params = 0
		for name, parameter in model.named_parameters():
			if not parameter.requires_grad: continue
			params = parameter.numel()
			table.add_row([name, params])
			total_params+=params
		if logfile:
			logfile.write(str(table))
			logfile.write("\nTotal Trainable Params: " + "{:,}".format(total_params))
		else:
			print(table)
			print("Total Trainable Params: ", "{:,}".format(total_params))
		return total_params

def ImportDataset(path, train_val_test_split, num_heads, only_test=False):
	"""
		returns dictionary of features-labels for train, val, test datasets
		return shape: (batch,channel,height,width)
	"""
	ntrain = train_val_test_split[0]
	nval = train_val_test_split[1]
	ntest = train_val_test_split[2]

	if num_heads==1:
		pre_filename = "./mat_files/PK11"
	elif num_heads==5:
		pre_filename = "./mat_files/PK15689"
	else: 
		raise ValueError("Bad num_heads.")

	if only_test:
		test = scipy.io.loadmat(path+"./mat_files/PK15689"+ "_test.mat")
		test["input"] = torch.tensor(test["input"][:ntest]).permute(0,3,1,2).float()	
		if len(test["output"].shape) == 3:
			test["output"] = torch.tensor(test["output"][:ntest])[:,None,:,:].float()
		elif len(test["output"].shape) == 4:
			test["output"] = torch.tensor(test["output"][:ntest]).permute(0,3,1,2).float()
		else: raise ValueError("Bad shape for y.")
		return [],[], test
	
	else:
		train = scipy.io.loadmat(path+pre_filename+"_train.mat")
		val = scipy.io.loadmat(path+pre_filename+  "_val.mat")
		test = scipy.io.loadmat(path+"./mat_files/test_100/PK15689"+ "_test.mat")

		train["input"] = torch.tensor(train["input"][:ntrain]).permute(0,3,1,2).float()
		val["input"] = torch.tensor(val["input"][:nval]).permute(0,3,1,2).float()
		test["input"] = torch.tensor(test["input"][:ntest]).permute(0,3,1,2).float()	

		if len(train["output"].shape) == 3:
			train["output"] = torch.tensor(train["output"][:ntrain])[:,None,:,:].float()
			val["output"] = torch.tensor(val["output"][:nval])[:,None,:,:].float()
			test["output"] = torch.tensor(test["output"][:ntest])[:,None,:,:].float()
		elif len(train["output"].shape) == 4:
			train["output"] = torch.tensor(train["output"][:ntrain]).permute(0,3,1,2).float()
			val["output"] = torch.tensor(val["output"][:nval]).permute(0,3,1,2).float()
			test["output"] = torch.tensor(test["output"][:ntest]).permute(0,3,1,2).float()
		else: raise ValueError("Bad shape for y.")
		return train, val, test
	

def LossPlots(gen_train_loss,gen_val_loss,save_path,exp_name=""):
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

def LossPlotsGAN(gen_train_loss,gen_val_loss,disc_train_loss,save_path,exp_name=""):
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

	if disc_train_loss!= None:
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

class GaussianScaling():
	def __init__(self, x, eps=0.00001):
		# works for shape (batchsize, channels, dim1, dim2)
		# computes mean, std of shape (channels)
		# 1 set of parameters for each channel
		self.mean = torch.mean(x, (0,2,3))      
		self.std = torch.std(x, (0,2,3))
		self.eps = eps

	def encode(self, x):
		z = torch.ones_like(x)
		for i in range(len(self.mean)):
			z[:,i,:,:] = (x[:,i,:,:] - self.mean[i]) / (self.std[i] + self.eps)
		return z

	def decode(self, x):
		z = torch.ones_like(x)
		for i in range(len(self.mean)):
			z[:,i,:,:] = (x[:,i,:,:] * (self.std[i] + self.eps)) + self.mean[i]
		return z

class MinMaxScaling():
	# works for shape (batchsize, channels, dim1, dim2)
	def __init__(self, x):
		self.min = torch.tensor([torch.min(x[:,i,:,:]) for i in range(x.shape[1])])
		self.max = torch.tensor([torch.max(x[:,i,:,:]) for i in range(x.shape[1])])

	def encode(self, x):
		z = torch.ones_like(x)
		for i in range(len(self.min)):
			z[:,i,:,:] = (x[:,i,:,:] - self.min[i])/(self.max[i] - self.min[i])
		return z

	def decode(self, x): 
		z = torch.ones_like(x)
		for i in range(len(self.min)):
			z[:,i,:,:] = (x[:,i,:,:] * (self.max[i] - self.min[i])) +  self.min[i]
		return z

# normalization, scaling by range
class RangeNormalizer(object):
	"""
		need fixing: number of channels
	"""
	def __init__(self, x, low=0.0, high=1.0):
		super(RangeNormalizer, self).__init__()
		mymin = torch.min(x, 0)[0].view(-1)
		mymax = torch.max(x, 0)[0].view(-1)

		self.a = (high - low)/(mymax - mymin)
		self.b = -self.a*mymax + high

	def encode(self, x):
		s = x.size()
		x = x.view(s[0], -1)
		x = self.a*x + self.b
		x = x.view(s)
		return x

	def decode(self, x):
		s = x.size()
		x = x.view(s[0], -1)
		x = (x - self.b)/self.a
		x = x.view(s)
		return x
