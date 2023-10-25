import torch
from prettytable import PrettyTable
from torch.nn import L1Loss, MSELoss

def mae(f1,f2):
	return torch.mean(torch.abs(f1-f2))

def nmae(f1,f2):
	'''
		Normalized MAE
	'''
	return mae(f1,f2)/(torch.max(f1)-torch.min(f1))

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
			logfile.write("\n\n")
		else:
			print(table)
			print("Total Trainable Params: ", "{:,}".format(total_params))
			print("\n\n")
		return total_params

def periodic_padding(tensor, axis, padding):
    """
		Implemented by Nima
		
		Add periodic padding to a tensor for specified axis.

		:param tensor: the input tensor.
		:param axis: one or multiple axis for padding; an integer or a tuple of ints.
		:param padding: the padding size; int or tuple of ints corresponding to axis.
		:return: padded tensor.
    """

    if isinstance(axis, int):
        axis = (axis, )
    if isinstance(padding, int):
        padding = (padding, )
    assert len(axis) == len(padding), 'the number of axis and paddings are different.'
    ndim = len(tensor.shape)
    for ax, p in zip(axis, padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left
        ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = torch.cat([right, middle, left], axis=ax)
    return tensor

def lossFunction(type="L1"):
    if type=="L1":
        return L1Loss()
    elif type=="L2":
        return MSELoss()
    else:
        raise AssertionError("Unexpected argument for lossFunction.")
    
def topKAmplitudes(tensor,k):
    '''
		return a tensor of same shape as the input where
		top k elements in terms of magnitude are retained 
  		and other elements are substituted with 0
    '''
    topK = torch.topk(tensor.flatten(-2,-1).abs(),k,-1)
    topKth = topK.values[...,-1].unsqueeze(-1).unsqueeze(-1)
    condition = tensor.abs() >= topKth
    res = torch.where(condition,tensor,0)
    return res
    
if __name__=="__main__":
    
    x = torch.randn((2,2,4,2)) + 1 
    res = topKvalues(x,3)
    print(x)
    print(res)
    
    # modes = 2
    # xdim,ydim = 8,5
    # x_ft = torch.randn((1,2,xdim,ydim))+1
    # w1 = torch.randn(2,2,xdim,ydim)
    # w2 = torch.randn(2,2,xdim,ydim)
    
    # x_ft_ = x_ft
    # w1_ = w1
    # w2_ = w2
    
    # x_ft_ = torch.einsum("bixy,ioxy->boxy",x_ft_[:,:,:modes,:modes],w1_[:,:,:modes,:modes])
    # x_ft_ = torch.einsum("bixy,ioxy->boxy",x_ft_[:,:,-modes:,:modes],w2_[:,:,-modes:,:modes])
    
    # output = x_ft_
    # w1 = w1_
    # w2 = w2_