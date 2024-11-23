'''
from functools import partial

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


register_act('swish', partial(SWISH, inplace=cfg.mem.inplace))
register_act('lrelu_03', partial(nn.LeakyReLU, 0.3, inplace=cfg.mem.inplace))

# Add Gaussian Error Linear Unit (GELU).
register_act('gelu', nn.GELU)
'''

from functools import partial
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.register import register_act

# Initialize cfg with default settings
set_cfg(cfg)  # This initializes cfg with default values

# Optionally, load a configuration file if you have one
# load_cfg(cfg, 'path_to_config.yaml')

# Ensure cfg.mem.inplace is set
if not hasattr(cfg, 'mem'):
    cfg.mem = type('', (), {})()  # Create a simple empty object
cfg.mem.inplace = False  # Set to True if you prefer in-place operations

class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

register_act('swish', partial(SWISH, inplace=cfg.mem.inplace))
register_act('lrelu_03', partial(nn.LeakyReLU, 0.3, inplace=cfg.mem.inplace))
register_act('gelu', nn.GELU)
