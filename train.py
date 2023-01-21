import torch
import os
from time import time
import matplotlib.pyplot as plt
from scipy.io import savemat
from model import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss
from utilities import ImportDataset, CountParameters, MAELoss, GaussianNormalizer

exp_name = "256_test800_1"
os.makedirs(f"./training_output/{exp_name}",exist_ok=True)

## import dataset
data_path = "../../article1_FNO_UNet/data/elasto_plastic/256/"
train_val_test_split = [800,100,100]
train_data, val_data, _ = ImportDataset(data_path, train_val_test_split)

## define normalizers based on training data
x_normalizer = GaussianNormalizer(train_data["input"])
y_normalizer = GaussianNormalizer(train_data["output"])
x_train = x_normalizer.encode(train_data["input"])
x_val = x_normalizer.encode(val_data["input"])
y_train = train_data["output"]
# y_train = y_normalizer.encode(train_data["output"])

## training parameters
epoch_goal=30
learning_rate=0.001
gamma=0.5
step_size=100
batch_size = 1

## define dataloaders, models, and optimizers
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, val_data["output"]), batch_size=batch_size, shuffle=False)

g_model = Generator(modes=20,width=32,num_heads=1)
d_model = Discriminator(psize=70,num_heads=1)
CountParameters(g_model)
CountParameters(d_model)

g_optimizer = torch.optim.Adam(g_model.parameters(), lr=learning_rate, weight_decay=1e-4)
g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=step_size, gamma=gamma)
d_optimizer = torch.optim.Adam(d_model.parameters(), lr=learning_rate, weight_decay=1e-4)
d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=step_size, gamma=gamma)

## train the model
'''
	here, the validation loss is computed based on MAE
'''
# g_train_loss_list = []
# g_train_loss_gan_list = []
# g_train_loss_l1_list = []
# d_train_loss_list = []
# d_train_loss_gen_list = []
# d_train_loss_real_list = []
# g_val_loss_list = []
ep = 1
while(ep <= epoch_goal):
	g_model.train()
	d_model.train()

	g_train_loss = 0
	# g_train_loss_gan = 0
	# g_train_loss_l1 = 0
	d_train_loss = 0
	# d_train_loss_gen = 0
	# d_train_loss_real = 0

	t1 = time()
	for batch_x,batch_target in train_loader:
		## training discriminator
		d_optimizer.zero_grad()
		generated_fake = y_normalizer.decode(g_model(batch_x))
		d_output_generated_fake = d_model(batch_x,generated_fake)
		d_output_real = d_model(batch_x,batch_target)
		d_loss, d_loss_gen, d_loss_real = DiscriminatorLoss(d_output_generated_fake,d_output_real)
		d_loss.backward()
		d_optimizer.step()

		d_train_loss += d_loss.item()
		# d_train_loss_gen += d_loss_gen.item()
		# d_train_loss_real += d_loss_real.item()

		## training generator
		g_optimizer.zero_grad()
		generated_fake = y_normalizer.decode(g_model(batch_x))
		d_output_generated_fake = d_model(batch_x,generated_fake)
		g_loss, g_loss_gan, g_loss_l1 = GeneratorLoss(generated_fake, batch_target, d_output_generated_fake, LAMBDA=1e-6)
		g_loss.backward()
		g_optimizer.step()

		g_train_loss += g_loss.item()
		# g_train_loss_gan += g_loss_gan.item()
		# g_train_loss_l1 += g_loss_l1.item()
		
	g_scheduler.step()
	d_scheduler.step()
	
	g_train_loss /= float(train_val_test_split[0])
	# g_train_loss_gan /= float(train_val_test_split[0])
	# g_train_loss_l1 /= float(train_val_test_split[0])
	d_train_loss /= float(train_val_test_split[0])
	# d_train_loss_gen /= float(train_val_test_split[0])
	# d_train_loss_real /= float(train_val_test_split[0])
	
	t2 = time()
	print(f"Epoch: {ep}\t Completed in {t2-t1}sec\t g_train_loss={g_train_loss}\t d_train_loss={d_train_loss}", flush=True)
    
	ep+=1

	## computing validation loss
	# g_model.eval()
	# g_val_loss = 0
	# for x,target in val_loader:
	# 	g_output = g_model(x)
	# 	g_output = y_normalizer.decode(g_output)
	# 	g_val_loss += MAELoss(g_output,target).item()
	# g_val_loss /= float(train_val_test_split[1])
	
	# plt.imshow(g_output.squeeze().detach().numpy())
	# plt.savefig(f"./training_output/{exp_name}/epoch{ep}_pred.png", format="png")
	# plt.close()
	# plt.imshow(target.squeeze().detach().numpy())
	# plt.savefig(f"./training_output/{exp_name}/epoch{ep}_true.png", format="png")
	# plt.close()


	## saving all the losses into lists
	# g_train_loss_list.append(g_train_loss)
	# g_train_loss_gan_list.append(g_train_loss_gan)
	# g_train_loss_l1_list.append(g_train_loss_l1)
	# d_train_loss_list.append(d_train_loss)
	# d_train_loss_gen_list.append(d_train_loss_gen)
	# d_train_loss_real_list.append(d_train_loss_real)
	# g_val_loss_list.append(g_val_loss)

	# print(f"Epoch: {ep}\t Completed in {t2-t1}sec\t g_train_loss={g_train_loss}\t d_train_loss={d_train_loss}\t g_val_loss (MAE)={g_val_loss}")

## saving the losses in training log file
# loss_dict = {"g_train_loss":g_train_loss_list,
# 			 "g_train_loss_gan":g_train_loss_gan_list,
# 			 "g_train_loss_l1":g_train_loss_l1_list,
# 			 "d_train_loss":d_train_loss_list,
# 			 "d_train_gen_loss":d_train_loss_gen_list,
# 			 "d_train_real_loss":d_train_loss_real_list,
# 			 "g_val_loss":g_val_loss_list}
# savemat(f"./training_log/{exp_name}",loss_dict)

## save the model
torch.save(
	{
		'epoch_completed': ep-1,
		'model_state_dict': g_model.state_dict(),
		'optimizer_state_dict': g_optimizer.state_dict(),
		'scheduler_state_dict': g_scheduler.state_dict()
	},
	f"./saved_models/generator/{exp_name}.pt")

torch.save(
	{
		'epoch_completed': ep-1,
		'model_state_dict': d_model.state_dict(),
		'optimizer_state_dict': d_optimizer.state_dict(),
		'scheduler_state_dict': d_scheduler.state_dict()
	},
	f"./saved_models/discriminator/{exp_name}.pt")


