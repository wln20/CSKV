#!/usr/bin/env python

from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="svd-kv",
    version="0.1.0",
    description="Core code for SVD-KV",
    author="Luning Wang",
    author_email="wangluning2@gmail.com",
    # url="https://github.com/XXX.git",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
