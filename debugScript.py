import torch
from utilities import ReceptiveField
from model import GeneratorLoss, DiscriminatorLoss, Discriminator, Generator

"""
	Finding the parameters for 70x70 patchGAN
"""
rf=[1]			# starting from 1 pixel of the output field
rf.append(ReceptiveField(rf[-1],1,4))
rf.append(ReceptiveField(rf[-1],1,4))
rf.append(ReceptiveField(rf[-1],2,4))
rf.append(ReceptiveField(rf[-1],2,4))
rf.append(ReceptiveField(rf[-1],2,4))
print("Receptive fields per layer (last to first) for 70x70 patchGAN:\t",rf)

f = torch.zeros((32,1,256,256))
t = torch.ones((32,1,256,256))

print(GeneratorLoss(f,f,f,100))
print(DiscriminatorLoss(t,f))

x = torch.ones(1,3,256,256)
y = torch.ones(1,1,256,256)

d = Discriminator()
g = Generator()

print(g(x).shape)
print(d(x,g(x)).shape)
