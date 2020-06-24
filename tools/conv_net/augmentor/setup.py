from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


requirements = [
    'numpy',
    'scipy',
    'Cython',
]


extensions = [
    Extension(
        'augmentor.warping._warping',
        sources = ['augmentor/warping/*.pyx']
    ),
]


setup(
    # Metadata
    name='augmentor',
    version='0.0.1',
    author='Kisuk Lee',
    author_email='kisuklee@mit.edu',
    url='https://github.com/torms3/Augmentor',
    description='Data augmentation for 3D deep learning',

    # Package info
    packages=find_packages(exclude=('test',)),

    # Externel module
    ext_modules = cythonize(extensions),

    zip_safe=True,
    install_requires=requirements,
)
