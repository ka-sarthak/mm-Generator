import torch
from prettytable import PrettyTable

def mae(f1,f2):
	return torch.mean(torch.abs(f1-f2))

def nmae(f1,f2):
	'''
		Normalized MAE
	'''
	return MAE(f1,f2)/(torch.max(f1)-torch.min(f1))

def argumentCheck(argv):
	if not argv:
		continued = False
	elif argv[0] in ["--c","--continue"]:
		continued = True
	else: 
		print("ERROR: Bad arguments. Should be one of {},{--c},{--continue}.")
		exit()
	return continued

def receptiveField(output, stride, ksize):
	"""
		output field size is computed by Floor( (in+2p-ksize)/stride + 1)
		therefore, in = (out-1)*stride -2p + ksize
	"""
	return (output-1)*stride+ksize

def countParameters(model,logfile=None):
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