"""FastIE 节点基类，继承该类的子类将具有以下功能：

* 自动将配置类中的配置项注册为 ``argparse`` 解析器的参数，并在解析时自动赋值
* 从 ``dict`` 类型的配置对象或者配置文件实例化
"""
__all__ = ['BaseNodeConfig', 'BaseNode']

import inspect
import re
import zipfile
from argparse import ArgumentParser, Namespace, Action
from dataclasses import dataclass, MISSING
from typing import Union, Sequence, Optional, Dict, Type

from fastie.envs import parser as global_parser, get_flag, PARSER_FLAG
from fastie.utils.utils import parse_config


@dataclass
class BaseNodeConfig:
    """FastIE 节点配置基类."""

    def parse(self, obj: object):
        """将当前对象的属性值赋值给obj.

        :param obj: 任意对象
        :return:
        """
        for key, value in self.__dict__.items():
            setattr(obj, key, value)

    def to_dict(self) -> dict:
        """将当前对象转换为字典.

        :return:
        """
        fields = dict()
        for field_name, field_value in self.__class__.__dict__[
                '__dataclass_fields__'].items():
            fields[field_name] = dict(type=getattr(field_value, 'type'),
                                      default=getattr(field_value, 'default'),
                                      default_factory=getattr(
                                          field_value, 'default_factory'),
                                      metadata=getattr(field_value,
                                                       'metadata'))
        return fields

    @classmethod
    def from_dict(cls, _config: dict):
        """从字典中创建配置.

        :param _config: ``dict`` 类型的配置
        :return: :class:`BaseNodeConfig` 类型的配置
        """
        config = cls()
        for key in config.__dir__():
            if not key.startswith('_') and key in _config.keys():
                setattr(config, key, _config[key])
        return config

    def keys(self):
        """获取当前配置的所有属性名.

        :return: ``list`` 类型的属性名列表
        """
        return [key for key in self.__dir__() if not key.startswith('_')]

    def __getitem__(self, item):
        """通过 ``[]`` 获取属性值.

        :param item: 属性名
        :return: 属性值
        """
        return getattr(self, item)


class BaseNode(object):
    """FastIE 节点基类.

    继承该类的子类将具有以下功能：
        * 自动将配置类中的配置项注册为 ``argparse`` 解析器的参数
        * 从 ``dict`` 类型的配置对象或者配置文件实例化
    """
    _config = BaseNodeConfig()
    _help = 'The base class of all node objects'

    def __init__(self, **kwargs):
        self._parser = global_parser.add_argument_group(
            title=getattr(self, '_help'))
        self._overload_config: dict = {}

    @classmethod
    def from_config(cls, config: Union[BaseNodeConfig, str, dict]):
        """从配置文件或配置对象中创建节点.

        :param config: 可以为 ``*.py`` 文件路径或者 :class:`BaseNodeConfig` 类型的对象
        :return: :class:`BaseNode` 类型的节点
        """
        node = cls()
        if isinstance(config, BaseNodeConfig):
            node._config = config
        else:
            _config = parse_config(config)
            if _config is not None:
                node._overload_config = _config
                node._config = node._config.__class__.from_dict(_config)
        node._config.parse(node)
        return node

    @property
    def parser(self):
        """根据当前节点的配置类构造当前节点的 ``argparse`` 解析器.

        :return: :class:`argparse.ArgumentParser` 类型的解析器
        """

        def inspect_all_bases(cls: type):
            if cls == object:
                return
            if PARSER_FLAG == 'dataclass':
                for key, value in self._config.to_dict().items():
                    if isinstance(value['metadata']['existence'], bool) \
                            and value['metadata']['existence'] == True \
                            or isinstance(value['metadata']['existence'], list) \
                            and get_flag() in value['metadata']['existence'] \
                            or isinstance(value['metadata']['existence'], str) \
                            and get_flag() == value['metadata']['existence']:
                        default_value = None
                        if value['default'] != MISSING:
                            default_value = value['default']
                        if value['default_factory'] != MISSING:
                            default_value = value['default_factory']()
                        arg_flag = [f'--{key}']
                        if 'alias' in value['metadata']:
                            if isinstance(value['metadata']['alias'], str):
                                arg_flag = [
                                    *arg_flag, value['metadata']['alias']
                                ]
                            elif isinstance(value['metadata']['alias'],
                                            Sequence):
                                arg_flag.extend([
                                    item for item in value['metadata']['alias']
                                ])
                        nargs = 1
                        if 'nargs' in value['metadata']:
                            nargs = value['metadata']['nargs']
                        if type(default_value) == bool:
                            self._parser.add_argument(
                                *arg_flag,
                                default=default_value,
                                help=f"{value['metadata']['help']} "
                                f'default: {default_value}',
                                action=self.action,
                                metavar='',
                                nargs='?',
                                const=True,
                                required=False)
                        else:
                            self._parser.add_argument(
                                *arg_flag,
                                default=default_value,
                                type=type(default_value),
                                help=f"{value['metadata']['help']} "
                                f'default: {default_value}',
                                action=self.action,
                                metavar='',
                                nargs=nargs,
                                required=False)
            elif PARSER_FLAG == 'comment':
                for key, value in cls().comments.items():
                    if get_flag() in value['flags']:
                        self._parser.add_argument(
                            f'--{key}',
                            default=value['value'],
                            type=type(value['value']),
                            help=
                            f"{value['description']} 默认值为: {value['value']}",
                            action=self.action,
                            metavar='',
                            required=False)
                for father in cls.__bases__:
                    inspect_all_bases(father)

        inspect_all_bases(self.__class__)
        return self._parser

    @property
    def action(self) -> Type[Action]:
        """根据当前节点的配置类构造当前节点的 ``argparse`` 解析器的 ``action`` 参数.

        :return: :class:`argparse.Action` 类型的 ``action`` 参数
        """
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
                if option_string.replace('--', '') in field_dict.keys():
                    variable_name = option_string.replace('--', '')
                    if 'multi_method' in field_dict[
                            variable_name].metadata.keys():
                        if field_dict[variable_name].metadata[
                                'multi_method'] == 'space-join':
                            values = ' '.join(values)
                        else:
                            # TODO: 当此位置接受多个参数时，需要对多个参数采取的操作
                            pass
                    if isinstance(values, Sequence) and len(values) == 1:
                        values = values[0]
                    setattr(node, variable_name, values)
                    setattr(namespace, variable_name, values)
                else:
                    for key, value in field_dict.items():
                        if isinstance(value.metadata['alias'], Sequence):
                            if option_string in value.metadata['alias']:
                                if 'multi_method' in value.metadata.keys():
                                    if value.metadata[
                                            'multi_method'] == 'space-join':
                                        values = ' '.join(values)
                                    else:
                                        # TODO: 当此位置接受多个参数时，需要对多个参数采取的操作
                                        pass
                                setattr(node, key, values)
                                setattr(namespace, key, values)

                        elif isinstance(value.metadata['alias'], str):
                            if option_string == value.metadata['alias']:
                                if 'multi_method' in value.metadata.keys():
                                    if value.metadata[
                                            'multi_method'] == 'space-join':
                                        values = ' '.join(values)
                                    else:
                                        # TODO: 当此位置接受多个参数时，需要对多个参数采取的操作
                                        pass
                                setattr(node, key, values)
                                setattr(namespace, key, values)

        return ParseAction

    @property
    def comments(self) -> dict:
        """获取当前节点的注释信息.

        .. warning::

            该方法已废弃

        :return:
        """
        comments = {}

        def inspect_all_bases(cls: type):
            if cls == object:
                return
            code_path = inspect.getfile(cls)
            try:
                with open(file=inspect.getfile(cls), mode='r') as file:
                    lines = file.readlines()
            except NotADirectoryError:
                egg_file = code_path.split('.egg')[0] + '.egg'
                sub_path = code_path.split('.egg')[1][1:]
                with zipfile.ZipFile(egg_file, 'r') as zip_file:
                    lines = list(
                        map(lambda x: x.decode(),
                            zip_file.open(sub_path).readlines()))
            for index in range(len(lines)):
                if 'Args:' in lines[index]:
                    break
            for index in range(index + 1, len(lines)):
                if ':' in lines[index]:
                    match_result = re.search(
                        r':(.*?)\((.*?)\)\[(.*?)\]=(.*?):(.*?)$', lines[index])
                    if match_result is None:
                        continue
                    else:
                        key, t, flags, value, description = match_result.groups(
                        )
                        comments[key.strip()] = dict(
                            type=t.strip(),
                            flags=flags.strip().split(','),
                            value=value,
                            description=description)
                if '"""' in lines[index]:
                    break
            for cls in cls.__bases__:
                inspect_all_bases(cls)

        inspect_all_bases(self.__class__)
        return comments

    @property
    def description(self):
        """获取当前节点注释中的描述信息.

        .. warning::
            该方法已废弃

        :return:
        """
        code_path = inspect.getfile(self.__class__)
        try:
            with open(file=inspect.getfile(self.__class__), mode='r') as file:
                lines = file.readlines()
        except NotADirectoryError:
            egg_file = code_path.split('.egg')[0] + '.egg'
            sub_path = code_path.split('.egg')[1][1:]
            with zipfile.ZipFile(egg_file, 'r') as zip_file:
                lines = zip_file.open(sub_path).readlines()
                lines = list(map(lambda x: x.decode(), lines))
        for index in range(len(lines)):
            if '"""' in lines[index]:
                return lines[index].replace('"""', '').strip()

    @property
    def fields(self) -> dict:
        """获取当前节点配置类的所有字段.

        :return: ``dict`` 类型的字段信息
        """
        fields: Dict[str, dict] = dict()
        for key in self.__dir__():
            if isinstance(object.__getattribute__(self, key), BaseNodeConfig):
                config_cls = object.__getattribute__(self, key).__class__
                for field_name, field_value in config_cls.__dict__[
                        '__dataclass_fields__'].items():
                    fields[field_name] = dict(
                        type=getattr(field_value, 'type'),
                        default=getattr(field_value, 'default'),
                        default_factory=getattr(field_value,
                                                'default_factory'),
                        metadata=getattr(field_value, 'metadata'))
        return fields
