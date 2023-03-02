import os
from argparse import ArgumentParser
from fastNLP import logger

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


def get_config():
    return global_config
