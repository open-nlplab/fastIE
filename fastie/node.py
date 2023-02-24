import inspect
import os
import re
import zipfile
from argparse import ArgumentParser, Namespace, Action
from dataclasses import dataclass, MISSING
from functools import reduce
from typing import Callable, Union, Sequence

from fastie.envs import parser as global_parser, get_flag, type_dict, global_config, \
    parser_flag, set_config, config_flag, find_config
from fastie.utils import Config


@dataclass
class BaseNodeConfig:
    def parse(self, obj: object):
        for key, value in self.__dict__.items():
            setattr(obj, key, value)

    def to_dict(self) -> dict:
        fields = dict()
        for field_name, field_value in self.__class__.__dict__["__dataclass_fields__"].items():
            fields[field_name] = dict(type=getattr(field_value, "type"),
                                      default=getattr(field_value, "default"),
                                      default_factory=getattr(field_value, "default_factory"),
                                      metadata=getattr(field_value, "metadata"))
        return fields

    @classmethod
    def from_dict(cls, _config: dict):
        config = cls()
        for key in config.__dir__():
            if not key.startswith("_") and key in _config.keys():
                setattr(config, key, _config[key])
        return config


class BaseNode(object):
    """ 节点基类

    继承该类的对象可以从命令行参数中获取值改变属性
    """
    _config = BaseNodeConfig()
    _help = "The base class of all node objects"
    def __init__(self, **kwargs):
        self._parser = global_parser.add_argument_group(title=getattr(self, "_help"))

    @classmethod
    def from_config(cls, config: Union[BaseNodeConfig, str]):
        node = cls()
        if isinstance(config, str):
            # 用户自己提供的配置文件
            if os.path.exists(config) and os.path.isfile(config) and config.endswith(".py"):
                if config_flag == "dict":
                    config_dict = reduce(lambda x, y: x.update(y),
                                         [value for value in Config.fromfile(config)._cfg_dict.values()
                                          if isinstance(value, dict)])
                    set_config(config_dict)
                    node._config = node._config.__class__.from_dict(config_dict)
                elif config_flag == "class":
                    config_obj = Config.fromfile(config)._cfg_dict.Config()
                    config_dict = {key: getattr(config_obj, key) for key in dir(config_obj) if not key.startswith("_")}
                    set_config(config_dict)
                    node._config = node._config.__class__.from_dict(config_dict)
            elif find_config(config) is not None:
                if config_flag == "dict":
                    config_dict = reduce(lambda x, y: x.update(y),
                                         [value for value in Config.fromfile(find_config(config))._cfg_dict.values()
                                          if isinstance(value, dict)])
                    set_config(config_dict)
                    node._config = node._config.__class__.from_dict(config_dict)
                elif config_flag == "class":
                    config_obj = Config.fromfile(find_config(config))._cfg_dict.Config()
                    config_dict = {key: getattr(config_obj, key) for key in dir(config_obj) if not key.startswith("_")}
                    set_config(config_dict)
                    node._config = node._config.__class__.from_dict(config_dict)
        else:
            node._config = config
            node._config.parse(node)
        return node

    @property
    def parser(self):
        def inspect_all_bases(cls: type):
            if cls == object:
                return
            if parser_flag == "dataclass":
                for key, value in self._config.to_dict().items():
                    if isinstance(value["metadata"]["existence"], bool) and value["metadata"]["existence"] == True \
                            or isinstance(value["metadata"]["existence"], list) and get_flag() in value["metadata"][
                        "existence"] \
                            or isinstance(value["metadata"]["existence"], str) and get_flag() == value["metadata"][
                        "existence"]:
                        default_value = None
                        if value["default"] != MISSING:
                            default_value = value["default"]
                        if value["default_factory"] != MISSING:
                            default_value = value["default_factory"]()
                        arg_flag = [f"--{key}"]
                        if "alias" in value["metadata"]:
                            if isinstance(value["metadata"]["alias"], str):
                                arg_flag = [*arg_flag, value["metadata"]["alias"]]
                            elif isinstance(value["metadata"]["alias"], Sequence):
                                arg_flag = arg_flag.extend([item for item in value["metadata"]["alias"]])
                        nargs = 1
                        if "nargs" in value["metadata"]:
                            nargs = value["metadata"]["nargs"]
                        if type(default_value) == bool:
                            self._parser.add_argument(*arg_flag,
                                                      default=default_value,
                                                      help=f"{value['metadata']['help']} default: {default_value}",
                                                      action=self.action,
                                                      metavar="",
                                                      nargs='?',
                                                      const=True,
                                                      required=False)
                        else:
                            self._parser.add_argument(*arg_flag,
                                                      default=default_value,
                                                      type=type(default_value),
                                                      help=f"{value['metadata']['help']} default: {default_value}",
                                                      action=self.action,
                                                      metavar="",
                                                      nargs=nargs,
                                                      required=False)
            elif parser_flag == "comment":
                for key, value in cls().comments.items():
                    if get_flag() in value["flags"]:
                        self._parser.add_argument(f"--{key}",
                                                  default=value["value"],
                                                  type=type_dict[value["type"]],
                                                  help=f"{value['description']} 默认值为: {value['value']}",
                                                  action=self.action,
                                                  metavar="",
                                                  required=False)
                for father in cls.__bases__:
                    inspect_all_bases(father)

        inspect_all_bases(self.__class__)
        return self._parser

    @property
    def action(self):
        node = self

        class ParseAction(Action):
            def __call__(self, parser: ArgumentParser,
                         namespace: Namespace, values, option_string: str = None):
                field_dict = node._config.__class__.__dataclass_fields__

                if option_string.replace("--", "") in field_dict.keys():
                    variable_name = option_string.replace("--", "")
                    if "multi_method" in field_dict[variable_name].metadata.keys():
                        if field_dict[variable_name].metadata["multi_method"] == "space-join":
                            values = " ".join(values)
                        else:
                            # TODO: 当此位置接受多个参数时，需要对多个参数采取的操作
                            pass


                    setattr(node, variable_name, values)
                    setattr(namespace, variable_name, values)
                else:
                    for key, value in field_dict.items():
                        if isinstance(value.metadata["alias"], Sequence):
                            if option_string in value.metadata["alias"]:
                                if "multi_method" in value.metadata.keys():
                                    if value.metadata["multi_method"] == "space-join":
                                        values = " ".join(values)
                                    else:
                                        # TODO: 当此位置接受多个参数时，需要对多个参数采取的操作
                                        pass
                                setattr(node, key, values)
                                setattr(namespace, key, values)

                        elif isinstance(value.metadata["alias"], str):
                            if option_string == value.metadata["alias"]:
                                if "multi_method" in value.metadata.keys():
                                    if value.metadata["multi_method"] == "space-join":
                                        values = " ".join(values)
                                    else:
                                        # TODO: 当此位置接受多个参数时，需要对多个参数采取的操作
                                        pass
                                setattr(node, key, values)
                                setattr(namespace, key, values)

        return ParseAction

    @property
    def comments(self) -> dict:
        comments = {}

        def inspect_all_bases(cls: type):
            if cls == object:
                return
            code_path = inspect.getfile(cls)
            try:
                with open(file=inspect.getfile(cls), mode="r") as file:
                    lines = file.readlines()
            except NotADirectoryError:
                egg_file = code_path.split(".egg")[0] + ".egg"
                sub_path = code_path.split(".egg")[1][1:]
                with zipfile.ZipFile(egg_file, "r") as zip_file:
                    lines = zip_file.open(sub_path).readlines()
                    lines = list(map(lambda x: x.decode(), lines))
            for index in range(len(lines)):
                if "Args:" in lines[index]:
                    break
            for index in range(index + 1, len(lines)):
                if ":" in lines[index]:
                    if re.search(r":(.*?)\((.*?)\)\[(.*?)\]=(.*?):(.*?)$", lines[index]) is None:
                        continue
                    key, t, flags, value, description = \
                        re.search(r":(.*?)\((.*?)\)\[(.*?)\]=(.*?):(.*?)$", lines[index]).groups()
                    comments[key.strip()] = dict(type=t.strip(),
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
        code_path = inspect.getfile(self.__class__)
        try:
            with open(file=inspect.getfile(self.__class__), mode="r") as file:
                lines = file.readlines()
        except NotADirectoryError:
            egg_file = code_path.split(".egg")[0] + ".egg"
            sub_path = code_path.split(".egg")[1][1:]
            with zipfile.ZipFile(egg_file, "r") as zip_file:
                lines = zip_file.open(sub_path).readlines()
                lines = list(map(lambda x: x.decode(), lines))
        for index in range(len(lines)):
            if '"""' in lines[index]:
                return lines[index].replace('"""', '').strip()

    @property
    def fields(self):
        fields = dict()
        for key in self.__dir__():
            if isinstance(object.__getattribute__(self, key), BaseNodeConfig):
                ConfigClass: type = object.__getattribute__(self, key).__class__
                for field_name, field_value in ConfigClass.__dict__["__dataclass_fields__"].items():
                    fields[field_name] = dict(type=getattr(field_value, "type"),
                                              default=getattr(field_value, "default"),
                                              default_factory=getattr(field_value, "default_factory"),
                                              metadata=getattr(field_value, "metadata")
                                              )
        return fields

    def __getattribute__(self, item: str):
        if item in global_config.keys() and not item.startswith("_"):
            try:
                return global_config[item]
            except AttributeError:
                return object.__getattribute__(self, item)
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if not key.startswith("_") \
                and type(value) in [t for t in type_dict.values()] \
                and not isinstance(value, Callable) \
                and not isinstance(value, BaseNodeConfig):
            global_config[key] = value
        object.__setattr__(self, key, value)
