#!/usr/bin/env python
__doc__ = """
Augmentor - Flip and Rotate Only
"""

from augmentor import *

def get_augmentation(is_train, **kwargs):
    augs = list()
    augs.append(FlipRotate())
    return Compose(augs)
