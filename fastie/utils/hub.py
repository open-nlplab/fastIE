# -*- coding: UTF-8 -*- 
import os
import pickle


class Hub:

    @classmethod
    def load(cls, path: str):
        if os.path.exists(path) and os.path.isfile(path):
            with open(path, mode='br') as file:
                return pickle.load(file)
        else:
            # 到 s3 上下载
            pass

    @classmethod
    def save(cls, path: str, state_dict: dict):
        with open(path, mode='wb') as file:
            pickle.dump(state_dict, file)
