#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

version = {}
with open("src/baal/version.py") as fp:
    exec(fp.read(), version)


def read(*names, **kwargs):
    return io.open(join(dirname(__file__), *names), encoding=kwargs.get('encoding', 'utf8')).read()


with open('requirements.txt') as f:
    reqs = [line.strip() for line in f]

with open('documentation-requirements.txt') as f:
    documentation_requirements = [line.strip() for line in f]

setup(
    name='baal',
    version=version['__version__'],
    description='Library for bayesian active learning.',
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author='Parmida Atighehchian, Frédéric Branchaud-Charron, Jan Freyberg, Lorne Schell, Rafael Pardinas',
    author_email="""parmida@elementai.com,
                    frederic.branchaud-charron@elementai.com,
                    jan.freyberg@gmail.com,
                    rafael.pardinas@elementai.com,
                    lorne.schell@elementai.com""",
    url='https://github.com/ElementAI/baal',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Utilities',
    ],
    install_requires=reqs,
    extras_require={
        'test': ['pytest', 'pytest-pep8', 'hypothesis', 'coverage', 'coveralls'],
        'documentation': documentation_requirements,
    },
)
