import sys
from utils.config_module import configLoad
from model_train import train
from model_inference import inference

if __name__ == "__main__":
	# initialize the config
	configLoad("../config.yml")

	if sys.argv[1] == "--train" or sys.argv[1] == "--t":
		train()
	elif sys.argv[1] == "--inference" or sys.argv[1] == "--i":
		inference()
	else:
		raise AssertionError("ArgumentNotFound: Should be --train (--t) or --inference (--i).")
