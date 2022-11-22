import torch
from prettytable import PrettyTable
import scipy.io

def MAELoss(f1,f2):
	return torch.mean(torch.abs(f1-f2))

def receptive_field(output, stride, ksize):
	"""
		output field size is computed by Floor( (in+2p-ksize)/stride + 1)
		therefore, in = (out-1)*stride -2p + ksize
	"""
	return (output-1)*stride+ksize

def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print("Total Trainable Params: ", "{:,}".format(total_params))
        return total_params

def importDataset(path, train_val_test_split):
	"""
		returns dictionary of features-labels for train, val, test datasets
		return shape: (batch,channel,height,width)
	"""
	ntrain = train_val_test_split[0]
	nval = train_val_test_split[1]
	ntest = train_val_test_split[2]
	train = scipy.io.loadmat(path+"./mat_files/PK11_train.mat")
	val = scipy.io.loadmat(path+"./mat_files/PK11_val.mat")
	test = scipy.io.loadmat(path+"./mat_files/test_100/PK11_test.mat")

	train["input"] = torch.tensor(train["input"][:ntrain]).permute(0,3,1,2)
	val["input"] = torch.tensor(val["input"][:nval]).permute(0,3,1,2)
	test["input"] = torch.tensor(test["input"][:ntest]).permute(0,3,1,2)

	if len(train["output"].shape) == 3:
		train["output"] = torch.tensor(train["output"][:ntrain])[:,:,:,None]
		val["output"] = torch.tensor(val["output"][:nval])[:,:,:,None]
		test["output"] = torch.tensor(test["output"][:ntest])[:,:,:,None]

	val["output"] = val["output"].permute(0,3,1,2)
	train["output"] = train["output"].permute(0,3,1,2)
	test["output"] = test["output"].permute(0,3,1,2)

	return train, val, test

class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        # works for shape (batchsize, channels, dim1, dim2)
        # computes mean, std of shape (channels)
        # 1 set of parameters for each channel
        self.mean = torch.mean(x, (0,2,3))      
        self.std = torch.std(x, (0,2,3))
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

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
