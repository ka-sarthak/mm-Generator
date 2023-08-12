import sys
from utils.config_module import configLoad, config
from utils.probe_fourier_modes import initProbeFourierModes

if __name__ == "__main__":
	# initialize the config
	configLoad("../config.yml")
	initProbeFourierModes()
	from tasks.model_train import train
	from tasks.model_inference import inference
 
	if sys.argv[1] == "--train" or sys.argv[1] == "--t":
		train()
	elif sys.argv[1] == "--inference" or sys.argv[1] == "--i":
		inference()
	else:
		raise AssertionError("ArgumentNotFound: Should be --train (--t) or --inference (--i).")
