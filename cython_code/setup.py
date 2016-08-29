# -*- coding: utf-8 -*-
"""
generate and compile cython code
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "lib",                                # the extension name
           sources=["lib.pyx", "Neurons.cpp"], # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
      )))