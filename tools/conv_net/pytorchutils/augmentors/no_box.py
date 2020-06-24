#!/usr/bin/env python
__doc__ = """
Augmentor - No Box Occlusion
"""

from augmentor import *


def get_augmentation(is_train, **kwargs):
    # Mild misalignment
    m1 = Blend(
        [Misalign((0,10), margin=1), SlipMisalign((0,10), margin=1)],
        props=[0.7,0.3]
    )
    # Medium misalignment
    m2 = Blend(
        [Misalign((0,30), margin=1), SlipMisalign((0,30), margin=1)],
        props=[0.7,0.3]
    )
    # Missing section
    missing = Compose([
        MixedMissingSection(maxsec=1, double=True, random=True),
        MixedMissingSection(maxsec=3, double=False, random=True)
    ])

    augs = list()

    # Grayscale
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Missing section & misalignment
    augs.append(Blend([
        Compose([m1,m2]),
        MisalignPlusMissing((5,30), random=True),
        missing
    ]))

    # Out-of-focus
    augs.append(MixedBlurrySection(maxsec=7))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3))

    # Flip & rotate
    augs.append(FlipRotate())

    return Compose(augs)
