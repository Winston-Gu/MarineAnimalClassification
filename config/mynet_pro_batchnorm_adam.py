from .baseline import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

config['data'] = 'pro'
config['model'] = 'mynet'
config['BatchNorm'] = True
config['optimizer'] = 'Adam'