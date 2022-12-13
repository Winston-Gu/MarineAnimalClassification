from .baseline import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

config['data'] = 'pro'
config['model'] = 'vit'
config['optimizer'] = 'Adam'