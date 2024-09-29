import torch
import omegaconf
from omegaconf import OmegaConf
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


class TConfig:
    def __init__(self, d):
        self.d = d
        for key, value in d.items():
            setattr(self, key, value)
    
    def __get_item__(self, key):
        return self.d[key]


def str_to_tuple(string):  # note: assumes parentheses at first/last position
    return tuple(map(float, string[1:-1].split(',')))

def dict_str_to_tuple(d):
    for k, v in d.items():
        if isinstance(v, omegaconf.DictConfig) or isinstance(v, dict):
            d[k] = dict_str_to_tuple(v)
        if isinstance(v, str) and len(v) > 0 and v[0] == '(':
            d[k] = str_to_tuple(v)
    return d

def dict_list_to_tensor(config):
    t_config = {}
    for k, v in config.items():
        if isinstance(v, omegaconf.ListConfig) or isinstance(v, list):
            t_config[k] = torch.tensor(v).cuda()
        else:
            t_config[k] = v
    t_config = TConfig(t_config)
    return t_config

def resolve_config(config):
    OmegaConf.resolve(config)
    return dict_str_to_tuple(config)
