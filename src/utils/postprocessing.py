import matplotlib.pyplot as plt
import os

def lossPlots(gen_train_loss,gen_val_loss,save_path,exp_name=""):
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

def lossPlotsGAN(gen_train_loss,gen_val_loss,disc_train_loss,save_path,exp_name=""):
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