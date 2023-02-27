import os
import sys
from argparse import ArgumentParser, Namespace, Action
from dataclasses import dataclass, field
from typing import Sequence, Optional

from fastie.controller import CONTROLLER, Interactor
from fastie.dataset import DATASET
from fastie.envs import set_flag, FASTIE_HOME, set_config, global_config, \
    config_flag, parser, find_config
from fastie.exhibition import Exhibition
from fastie.tasks import NER, EE, RE
from fastie.utils import Config
from fastie.node import BaseNodeConfig, BaseNode
from fastie.chain import Chain

chain = Chain([])


def parse_config():
    keys = list(global_config.keys())
    for key in keys:
        if key == 'task':
            obj_cls = None
            task, solution = global_config[key].split('/')
            if task.lower() == 'ner':
                obj_cls = NER.get(solution)
            elif task.lower() == 'ee':
                obj_cls = EE.get(solution)
            elif task.lower() == 're':
                obj_cls = RE.get(solution)
            else:
                print(f'The task type `{task}` you selected does not exist. ')
                print('You can only choose from `ner`, `ee`, or `re`. ')
                exit(0)
            if obj_cls is not None:
                obj = obj_cls()
                _ = chain + obj
            else:
                print(
                    f'The solution `{solution}` you selected does not exist. ')
                print('Here are the optional options: \n')
                sys.argv.append('-t')
                sys.argv.append('-l')
                Exhibition.intercept()
                exit(0)
        elif key == 'dataset':
            obj_cls = DATASET.get(global_config[key])
            if obj_cls is None:
                print(f'The dataset `{global_config[key]}` you selected does '
                      f'not exist. ')
                print('Here are the optional options: \n')
                sys.argv.append('-d')
                sys.argv.append('-l')
                Exhibition.intercept()
                exit(0)
            else:
                obj = obj_cls()
                _ = chain + obj


@dataclass
class CommandNodeConfig(BaseNodeConfig):
    config: str = field(default='',
                        metadata=dict(help='The config file you want to use.',
                                      existence=True,
                                      alias='-c'))
    task: str = field(
        default='',
        metadata=dict(
            help=
            'The task you want to use. Please use / to split the task and the '
            'specific solution.',
            existence=True,
            alias='-t'))
    dataset: str = field(default='',
                         metadata=dict(
                             help='The dataset you want to work with.',
                             existence=['train', 'eval', 'infer'],
                             alias='-d'))


class CommandNode(BaseNode):
    """ fastIE command line basic arguments
    Args:
        :task (str)[train,evaluation,inference]=None: 任务名.
        :dataset (str)[train,evaluation,inference]=None: 数据集名.
    """

    _config = CommandNodeConfig()
    _help = 'fastIE command line basic arguments'

    def __init__(self,
                 solution: Optional[str] = None,
                 dataset: Optional[str] = None):
        BaseNode.__init__(self)
        self.solution: Optional[str] = solution
        self.dataset: Optional[str] = dataset

    @property
    def action(self):
        node = self

        class ParseAction(Action):

            def __call__(self,
                         parser: ArgumentParser,
                         namespace: Namespace,
                         values,
                         option_string: Optional[str] = None):
                if option_string is None:
                    return
                field_dict = node._config.__class__.__dataclass_fields__
                variable_name = ''
                if isinstance(values, list) and len(values) == 1:
                    values = values[0]
                if option_string.replace('--', '') in field_dict.keys():
                    variable_name = option_string.replace('--', '')
                    if variable_name != 'config':
                        setattr(node, variable_name, values)
                    setattr(namespace, variable_name, values)
                else:
                    for key, value in field_dict.items():
                        if isinstance(value.metadata['alias'], Sequence):
                            if option_string in value.metadata['alias']:
                                variable_name = key
                                if variable_name != 'config':
                                    setattr(node, variable_name, values)
                                setattr(namespace, variable_name, values)

                        elif isinstance(value.metadata['alias'], str):
                            if option_string == value.metadata['alias']:
                                variable_name = key
                                if variable_name != 'config':
                                    setattr(node, variable_name, values)
                                setattr(namespace, variable_name, values)
                if variable_name == 'task':
                    if '/' in values:
                        obj_cls = None
                        task, solution = values.split('/')
                        if task.lower() == 'ner':
                            obj_cls = NER.get(solution)
                        elif task.lower() == 'ee':
                            obj_cls = EE.get(solution)
                        elif task.lower() == 're':
                            obj_cls = RE.get(solution)
                        else:
                            print(
                                f'The task type `{task}` you selected does not '
                                f'exist. ')
                            print(
                                'You can only choose from `ner`, `ee`, or `re`. '
                            )
                            exit(0)
                        if obj_cls is not None:
                            obj = obj_cls()
                            _ = chain + obj
                        else:
                            print(
                                f'The solution `{solution}` you selected does '
                                f'not exist. ')
                            print('Here are the optional options: \n')
                            sys.argv.append('-t')
                            sys.argv.append('-l')
                            Exhibition.intercept()
                            exit(0)
                    else:
                        print(
                            f'You must specify both the task category and the '
                            f'specific solution, such as `ner/bert` instead of '
                            f'`{values}`. ')
                        print('Here are the optional options: \n')
                        sys.argv.append('-t')
                        sys.argv.append('-l')
                        Exhibition.intercept()
                        exit(0)
                elif variable_name == 'dataset':
                    obj_cls = DATASET.get(values)
                    if obj_cls is None:
                        print(f'The dataset `{values}` you selected does not '
                              f'exist. ')
                        print('Here are the optional options: \n')
                        sys.argv.append('-d')
                        sys.argv.append('-l')
                        Exhibition.intercept()
                        exit(0)
                    else:
                        obj = obj_cls()
                        _ = chain + obj
                elif variable_name == 'config':
                    if os.path.exists(values):
                        if config_flag == 'class':
                            config = Config.fromfile(values)['Config']()
                        elif config_flag == 'dict':
                            config = Config.fromfile(values)['config']
                        set_config(config)
                        parse_config()
                        set_config(config)
                    elif find_config(values) is not None:
                        if config_flag == 'class':
                            config = Config.fromfile(
                                find_config(values))['Config']()
                        elif config_flag == 'dict':
                            config = Config.fromfile(
                                find_config(values))['config']
                        set_config(config)
                        parse_config()
                        set_config(config)
                    else:
                        print(
                            f'The config file `{values}` you selected does not '
                            f'exist. ')
                        print('Here are the optional options: \n')
                        sys.argv.append('-c')
                        sys.argv.append('-l')
                        Exhibition.intercept()
                        exit(0)

        return ParseAction


def intercept_config():
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-c' or sys.argv[i] == '--config':
            if i < len(sys.argv) - 1:
                if os.path.exists(sys.argv[i + 1]):
                    config = Config.fromfile(sys.argv[i])['Config']()
                    set_config(config)
                elif os.path.exists(os.path.join(FASTIE_HOME,
                                                 sys.argv[i + 1])):
                    config = Config.fromfile(
                        os.path.join(FASTIE_HOME, sys.argv[i]))['Config']()
                    set_config(config)
                break


def interact_handler():
    if not sys.argv[0].endswith('interact'):
        return
    from fastie.dataset.io.sentence import Sentence
    sentence = ''
    while sentence != '!exit':
        sentence = input('Type a sequence, type `!exit` to end interacting:\n')
        if len(sentence) == 0:
            continue
        if sentence == '!exit':
            break
        sentence = Sentence(sentence=sentence)
        _ = chain + sentence
        chain.run()


def main():
    Exhibition.intercept()
    # intercept_config()
    # parse_config()
    if sys.argv[0].endswith('train'):
        _ = chain + CONTROLLER.get('trainer')()
        set_flag('train')
    elif sys.argv[0].endswith('eval'):
        _ = chain + CONTROLLER.get('evaluator')()
        set_flag('eval')
    elif sys.argv[0].endswith('infer'):
        _ = chain + CONTROLLER.get('inference')()
        set_flag('infer')
    elif sys.argv[0].endswith('interact'):
        _ = chain + CONTROLLER.get('interactor')()
        set_flag('interact')
    elif sys.argv[0].endswith('web'):
        set_flag('web')
    node = CommandNode()
    _ = node.parser
    args = parser.parse_known_args()
    args = parser.parse_known_args(args[1])
    interact_handler()
    chain.run()


if __name__ == '__main__':
    main()
