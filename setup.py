#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:34:48 2019

@author: wanglab
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lightsheet",
    version="0.0.1",
    author="Thomas J. Pisano",
    author_email="thomas.john.pisano@gmail.com",
    description="Lightsheet processing package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PrincetonUniversity/lightsheet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General License",
        "Operating System :: OS Independent",
    ],
)
