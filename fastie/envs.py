import os
from argparse import ArgumentParser

parser: ArgumentParser = ArgumentParser(prog='fastie-train',
                                        conflict_handler='resolve')

FASTIE_HOME = f"{os.environ['HOME']}/.fastie"

flag = None
parser_flag = 'dataclass'  # "comment"
config_flag = 'dict'  # class


def set_flag(_flag: str = 'train'):
    global flag
    if _flag not in ['train', 'eval', 'infer', 'interact', 'web']:
        _flag = 'train'
    else:
        flag = _flag


def get_flag():
    return flag


type_dict = {
    'int': int,
    'bool': bool,
    'float': float,
    'str': str,
    'list': list,
    'dict': dict
}

global_config: dict = dict()


def set_config(_config: object):
    global global_config
    if isinstance(_config, dict):
        for key, value in _config.items():
            if not key.startswith('_'):
                global_config[key] = value
    else:
        for key in _config.__dir__():
            if not key.startswith('_'):
                global_config[key] = getattr(_config, key)


def get_config():
    return global_config


def find_config(config: str):
    for root, dirs, files in os.walk(os.path.join(FASTIE_HOME, 'configs')):
        for file in files:
            if file.startswith(config):
                return os.path.join(root, file)
    return None
