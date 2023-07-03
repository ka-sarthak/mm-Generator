import sys
from config_module import configLoad
from train import train
from inference import inference

if __name__ == "__main__":
	# initialize the config
	configLoad("../config.yml")
	
	try:
		if sys.argv[1] == "--train" or sys.argv[1] == "--t":
			train()
		elif sys.argv[1] == "--inference" or sys.argv[1] == "--i":
			inference()
		else:
			raise(TypeError)
	except TypeError:
		print("Argument not found. Should be one of --train (--t) or --inference (--i).")