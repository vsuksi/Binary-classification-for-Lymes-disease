#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
    # If `torch` is desired, you can optionally install torch using `pip install torchvision`
    'Click>=6.0', 'numpy>=1.19.0', 'scipy>=1.5.1', 'scikit-learn>=0.23.1', 'matplotlib>=3.2.2','pandas>=1.0.5', 'h5py>=2.10.0', 'xlrd'
]

setup_requirements = [    
    # TODO(CSU-PAL-biology): put setup requirements (distutils extensions, etc.) here
    'pytest-runner', 'docutils==0.12'    
]

test_requirements = [
    # TODO: put package test requirements here
    'pytest'    
]

setup(
    name='calcom',
    version='0.4.0',
    description="A software to calculate and compare classification methods",
    long_description=readme + '\n\n' + history,
    author="Manuchehr Aminian",
    author_email="aminian@colostate.edu",
    url='https://github.com/CSU-PAL-biology/calcom',
    packages=find_packages(), #include=['calcom']
    entry_points={
#        'console_scripts': [
#            'calcom=calcom.cli:main'
#        ]
    },
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='calcom',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
