import sys
import os

from texttable import Texttable

from fastie.dataset import DATASET
from fastie.tasks import NER, EE, RE
from fastie.envs import FASTIE_HOME
from fastie.utils import Config


class Exhibition:
    @property
    def NER(self):
        table = list()
        for key, value in NER.module_dict.items():
            table.append(dict(task="NER",
                              solution=key,
                              description=value().description))
        return table

    @property
    def RE(self):
        table = list()
        for key, value in RE.module_dict.items():
            table.append(dict(task="RE",
                              solution=key,
                              description=value().description))
        return table

    @property
    def EE(self):
        table = list()
        for key, value in EE.module_dict.items():
            table.append(dict(task="EE",
                              solution=key,
                              description=value().description))
        return table

    @property
    def TASK(self):
        return self.NER + self.RE + self.EE

    @property
    def DATASET(self):
        table = list()
        for key, value in DATASET.module_dict.items():
            table.append(dict(dataset=key,
                              description=value().description))
        return table

    @classmethod
    def intercept(cls):
        if "-l" in sys.argv or "--list" in sys.argv:
            sys.argv.append("--task")
            table = Texttable()
            table.set_deco(Texttable.HEADER)
            exhibition = cls()
            for i in range(len(sys.argv)):
                if sys.argv[i].startswith("--task") or sys.argv[i].startswith("-t"):
                    if i < len(sys.argv) - 1 and sys.argv[i + 1].upper().replace("/", "").replace("\\", "") == "NER":
                        table.add_rows([["Task", "Solution", "Description"],
                                        *[[item["task"], item["solution"], item["description"]]
                                          for item in exhibition.NER]])
                    elif i < len(sys.argv) - 1 and sys.argv[i + 1].upper().replace("/", "").replace("\\", "") == "EE":
                        table.add_rows([["Task", "Solution", "Description"],
                                        *[[item["task"], item["solution"], item["description"]]
                                          for item in exhibition.EE]])
                    elif i < len(sys.argv) - 1 and sys.argv[i + 1].upper().replace("/", "").replace("\\", "") == "RE":
                        table.add_rows([["Task", "Solution", "Description"],
                                        *[[item["task"], item["solution"], item["description"]]
                                          for item in exhibition.RE]])
                    else:
                        table.add_rows([["Task", "Solution", "Description"],
                                        *[[item["task"], item["solution"], item["description"]]
                                          for item in exhibition.TASK]])
                    table.add_row(["", "", ""])
                    print(table.draw())
                    exit(0)
                elif sys.argv[i].startswith("--dataset") or sys.argv[i].startswith("-d"):
                    table.add_rows([["Dataset", "Description"],
                                    *[[item["dataset"], item["description"]]
                                      for item in exhibition.DATASET]])
                    table.add_row(["", ""])
                    print(table.draw())
                    exit(0)
                elif sys.argv[i].startswith("--config") or sys.argv[i].startswith("-c"):
                    table.add_row(["Config", "Description"])
                    for root, dirs, files in os.walk(os.path.join(FASTIE_HOME, "configs")):
                        for file in files:
                            if file.endswith(".py"):
                                config = Config.fromfile(os.path.join(root, file))
                                description = f"{file}"
                                if "_help" in config.keys():
                                    description = f": {config['_help']}"
                                table.add_row([f"{file.replace('.py', '')}", description])
                    print(table.draw())
                    exit(0)