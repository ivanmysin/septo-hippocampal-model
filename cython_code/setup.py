# -*- coding: utf-8 -*-
"""
generate and compile cython code
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "lib2",                                # the extension name
           sources=["lib2.pyx"], # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-std=c++11"],
      )))
