#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='amodal',
    version='0.0.0',
    description='Amodal completion',
    author='Jonas Kubilius',
    author_email='qbilius@gmail.com',
    url='https://github.com/qbilius/amodal',
    install_requires=['pytorch-lightning', 'svgwrite', 'cairosvg'],
    packages=find_packages(),
)
