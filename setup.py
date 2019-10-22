"""Setup script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

setuptools.setup(
  name='memotion',
  version='1.0.0',
  description=(
    'Internet Memo Emotion Analysis: Classify emotion and sentiment from ' +
    'multimodal internet memes.'),
  license='GNU General Public License v3.0',
  url='http://https://github.com/TongTongX/CMPUT651Project',
  install_requires=[
    'torch',
  ],
  packages=setuptools.find_packages(),
  classifiers=[
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: GNU General Public License v3.0',
  ],
)