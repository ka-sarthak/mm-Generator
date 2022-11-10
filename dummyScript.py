from utilities import receptive_field

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