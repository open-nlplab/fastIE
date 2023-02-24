from defaultlist import defaultlist

from fastie.controller import BaseController
from fastie.dataset import BaseDataset
from fastie.tasks import BaseTask


class Chain(defaultlist):
    def __init__(self, *args):
        defaultlist.__init__(self, *args)

    def run(self):
        result = None
        for node in self:
            result = node(result)
        return result

    def __add__(self, other):
        if isinstance(other, BaseTask):
            self[1] = other
        elif isinstance(other, BaseDataset):
            self[0] = other
        elif isinstance(other, BaseController):
            self[2] = other
        _ = other.parser
        return self

    def __call__(self, *args, **kwargs):
        self.run()
