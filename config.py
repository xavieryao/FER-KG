import json
from pprint import pprint


def load_config(filename, config_name):
    with open(filename) as f:
        configs = json.load(f)
    config_mapping = {}
    for cfg in configs:
        config_mapping[cfg['name']] = cfg
    cfg = config_mapping[config_name]
    par = cfg
    while par:
        if 'parent' in par:
            par = config_mapping[par['parent']]
            for k, v in par.items():
                if k not in cfg:
                    cfg[k] = v
        else:
            break
    pprint(cfg)
    return cfg