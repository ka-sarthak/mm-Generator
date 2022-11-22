import torch
from tqdm import tqdm
from time import time
from model import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss
from utilities import ImportDataset, CountParameters, MAELoss, GaussianNormalizer

exp_name = "test1"

## import dataset
data_path = "../../article1_FNO_UNet/data/elasto_plastic/256/"
train_val_test_split = [800,100,100]
train_data, val_data, test_data = ImportDataset(data_path, train_val_test_split)

## define normalizers based on training data
x_normalizer = GaussianNormalizer(train_data["input"])
y_normalizer = GaussianNormalizer(train_data["output"])
x_train = x_normalizer.encode(train_data["input"])
x_val = x_normalizer.encode(val_data["input"])
y_train = y_normalizer.encode(train_data["output"])

## training parameters
epochs=500
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
for ep in len(epochs):
	t1 = time()
	g_train_loss = 0
	d_train_loss = 0
	for i,batch in tqdm(enumerate(train_loader)):
		x = batch[0]
		target = batch[0]

		## evaluation from generator and discriminator
		g_model.eval()
		d_model.eval()
		generated_fake = y_normalizer.decode(g_model(x))
		d_output_generated_fake = d_model(x,generated_fake)
		d_output_real = d_model(x,target)
	
		## training generator
		g_model.train()
		g_optimizer.zero_grad()
		g_loss = GeneratorLoss(generated_fake, target, d_output_generated_fake)
		g_loss.backward()
		g_optimizer.step()
		g_train_loss += g_loss.item()

		## training discriminator
		d_model.train()
		d_optimizer.zero_grad()
		d_loss = DiscriminatorLoss(d_output_generated_fake,d_output_real)
		d_loss.backward()
		d_optimizer.step()
		d_train_loss += d_loss.item()
	g_scheduler.step()
	d_scheduler.step()
	t2 = time()

	## computing validation loss
	g_val_loss = 0
	for x,target in val_loader:
		g_output = Generator(x)
		g_output = y_normalizer.decode(g_output)
		g_val_loss += MAELoss(g_output,target).item()
	
	print(f"Epoch: {ep}\t Completed in {t2-t1}sec\t g_train_loss={g_train_loss}\t d_train_loss={d_train_loss}\t g_val_loss (MAE)={g_val_loss}")


## save the model
torch.save(g_model, f"./saved_models/generator/{exp_name}")
torch.save(d_model, f"./saved_models/discriminator/{exp_name}")


