import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pytorch_utils",
    version="0.0.0",
    author="Basil Kraft",
    author_email="bkraft.work@gmail.com",
    description=("Utitlities for building PyTorch models and pipelines."),
    license="MIT",
    packages=["pytorch_utils"],
    url="https://github.com/bask0/pytorch_utils",
    long_description=read('README.md'),
    classifiers=[
        "Topic :: Utilities",
        "License :: MIT",
    ],
)
