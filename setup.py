#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

try:
    from setuptools import setup
except:
    from distutils.core import setup

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'deprojected_sersic_models', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)


with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['numpy', 'scipy', 'matplotlib', 'astropy', 'dill']

setup_requirements = ['numpy']

setup(
    author="Sedona Price",
    author_email='sedona.price@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: 3-clause BSD',
        'Natural Language :: English',
        "Topic :: Scientific/Engineering",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="A package to calculate mass and kinematic profiles of non-spherical Sersic mass distributions.",
    install_requires=requirements,
    setup_requires=setup_requirements,
    license="3-clause BSD",
    long_description=readme,
    # include_package_data=True,
    name='deprojected_sersic_models',
    packages=['deprojected_sersic_models', 'deprojected_sersic_models.utils',
              'deprojected_sersic_models.plot'],
    version=__version__
)
