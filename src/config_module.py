import yaml

# define a global variable to be initialized by configLoad
# then use it across all the scripts, modules, functions
config = {}

def configLoad(configPath: str) -> None:
	with open(configPath,"r") as f:
			config.update(yaml.safe_load(f))