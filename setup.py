from setuptools import setup, find_packages
import os

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()

def find_configs():
    data_files = []
    for root, dirs, files in os.walk('configs'):
        for file in files:
            if file.endswith('.py'):
                data_files.append((os.path.join(os.path.expanduser('~'),
                                                os.path.join('.fastie/configs/', '/'.join(root.split('/')[-2:]))),
                                   ['./' + '/'.join(root.split('/')[-3:]) + '/' + file]))
    return data_files

setup(
    name='fastie',
    version='0.0.1',
    packages=find_packages(),
    install_requires=reqs.strip().split('\n'),
    entry_points={
        'console_scripts': [
            'fastie-train = fastie.command:main',
            'fastie-eval = fastie.command:main',
            'fastie-infer = fastie.command:main'
        ]
    },
    data_files=find_configs()
)
