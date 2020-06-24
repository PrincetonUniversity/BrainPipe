# from Cython.Build import cythonize
# from distutils.sysconfig import get_python_inc
# from setuptools import setup, Extension
from setuptools import setup

setup(
    name='dataprovider3',
    version='0.0.1',
    description='DataProvider3.',
    url='https://github.com/torms3/DataProvider3',
    author='Kisuk Lee',
    author_email='kisuklee@mit.edu',
    license='MIT',
    # requires=['cython'],
    packages=['dataprovider3',
              'dataprovider3.geometry',
              'dataprovider3.inference'],
    include_package_data=True,
    zip_safe=False
)
