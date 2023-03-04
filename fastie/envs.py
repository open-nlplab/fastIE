"""FastIE 全局变量."""
__all__ = [
    'get_task', 'set_task', 'get_dataset', 'set_dataset', 'get_flag',
    'set_flag'
]
import os
from argparse import ArgumentParser
from fastNLP import logger

parser: ArgumentParser = ArgumentParser(prog='fastie-train',
                                        conflict_handler='resolve')

FASTIE_HOME = f"{os.environ['HOME']}/.fastie"

PARSER_FLAG = 'dataclass'  # "comment"
CONFIG_FLAG = 'dict'  # class

task = None


def get_task():
    return task


def set_task(_task):
    global task
    task = _task


dataset = None


def get_dataset():
    return dataset


def set_dataset(_dataset):
    global dataset
    dataset = _dataset


FLAG = None


def set_flag(_flag: str = 'train'):
    global FLAG
    if _flag not in ['train', 'eval', 'infer', 'interact', 'server']:
        _flag = 'train'
    else:
        FLAG = _flag


def get_flag():
    return FLAG


sample_type = [int, bool, float, str, list, dict, set, tuple, type(None)]
