import io
import os
import pickle


class Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class Hub:

    @classmethod
    def load(cls, path: str):
        if os.path.exists(path) and os.path.isfile(path):
            with open(path, mode='br') as file:
                return Unpickler(file).load()
        else:
            # 到 s3 上下载
            pass

    @classmethod
    def save(cls, path: str, state_dict: dict):
        with open(path, mode='wb') as file:
            pickle.dump(state_dict, file)
