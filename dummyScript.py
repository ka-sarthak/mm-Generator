import torch
from utilities import receptive_field, generator_loss, discriminator_loss

"""
	Finding the parameters for 70x70 patchGAN
"""
rf=[1]			# starting from 1 pixel of the output field
rf.append(receptive_field(rf[-1],1,4))
rf.append(receptive_field(rf[-1],1,4))
rf.append(receptive_field(rf[-1],2,4))
rf.append(receptive_field(rf[-1],2,4))
rf.append(receptive_field(rf[-1],2,4))
print("Receptive fields per layer (last to first) for 70x70 patchGAN:\t",rf)

f = torch.zeros((32,1,256,256))
t = torch.ones((32,1,256,256))

print(generator_loss(100,f,f,f))
print(discriminator_loss(t,f))
