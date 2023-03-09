"""This module is used to build the task from config."""
__all__ = ['build_task']
from fastie.tasks import NER, RE, EE
from fastie.envs import get_task
from fastie.utils.utils import parse_config

from typing import Union, Optional

from copy import deepcopy


def build_task(_config: Optional[Union[dict, str]] = None):
    """Build the task you want to use from the config you give.

    :param _config: The config you want to use. It can be a dict or a path to a `*.py` config file.
    :return: The task in config.
    """
    task, solution = '', ''
    if _config is not None:
        _config = parse_config(_config)
    if not get_task():
        if _config is None:
            raise ValueError('The task you want to use is not specified.')
        else:
            if isinstance(_config, dict) and 'task' not in _config.keys():
                raise ValueError('The task you want to use is not specified.')
            else:
                task, solution = _config['task'].split('/')
    else:
        task, solution = get_task().split('/')
    if task.lower() == 'ner':
        task_cls = NER.get(solution)
    elif task.lower() == 're':
        task_cls = RE.get(solution)
    elif task.lower() == 'ee':
        task_cls = EE.get(solution)
    if task_cls is None:
        raise ValueError(
            f'The task {task} with solution {solution} is not supported.')
    task_obj = task_cls.from_config(_config)
    return task_obj
