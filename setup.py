from setuptools import setup

setup(
    name="pmpnndiff",
    packages=[
        'models',
        'data',
        'experiments',
        'se3_diffusion',
    ],
    package_dir={
        'models': './models',
        'data': './data',
        'experiments': './experiments',
        'se3_diffusion': './se3_diffusion',
    },
)