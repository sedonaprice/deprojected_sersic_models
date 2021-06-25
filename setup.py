#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

try:
    from setuptools import setup
except:
    from distutils.core import setup


dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'sersic_profile_mass_VC', '__init__.py')).read()
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
    ],
    description="A package to calculate mass and kinematic profiles of flattened Sersic mass distributions.",
    install_requires=requirements,
    setup_requires=setup_requirements,
    license="3-clause BSD",
    long_description=readme,
    include_package_data=True,
    name='sersic_profile_mass_VC',
    packages=['sersic_profile_mass_VC'],
    package_data={'sersic_profile_mass_VC': ['data/SomethingToFix/*.fits']},
    version=__version__
)
