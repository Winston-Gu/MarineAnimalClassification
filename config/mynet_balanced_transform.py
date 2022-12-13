from .baseline import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

config['data'] = 'balanced'
config['model'] = 'mynet'
config['transform'] = True