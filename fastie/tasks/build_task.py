# -*- coding: UTF-8 -*- 
from fastie.tasks import NER, RE, EE
from fastie.envs import global_config, set_config
from fastie.node import BaseNodeConfig

from typing import Union, Optional

from copy import deepcopy


def build_task(config: Optional[Union[dict, BaseNodeConfig]] = None):
    """Build the task you want to use from the config you give or the global
    config.

    :param config:
    :return:
    """
    if config is None:
        config = global_config
    origin_global_config = deepcopy(global_config)
    if not hasattr(config, 'task') and 'task' not in config.keys():
        raise ValueError('The task you want to use is not specified.')
    if hasattr(config, 'task'):
        task, solution = getattr(config, 'task').split('/')
    elif 'task' in config.keys():
        task, solution = config['task'].split('/')
    if task.lower() == 'ner':
        task_cls = NER.get(solution)
    elif task.lower() == 're':
        task_cls = RE.get(solution)
    elif task.lower() == 'ee':
        task_cls = EE.get(solution)
    if task_cls is None:
        raise ValueError(
            f'The task {task} with solution {solution} is not supported.')
    task_obj = task_cls()
    set_config(origin_global_config)
    return task_obj
