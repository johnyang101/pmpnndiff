from setuptools import setup

setup(
    name="pmpnndiff",
    packages=[
        'models',
        'data',
        'experiments',
    ],
    package_dir={
        'models': './models',
        'data': './data',
        'experiments': './experiments',
    },
)