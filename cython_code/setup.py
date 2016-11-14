# -*- coding: utf-8 -*-
"""
generate and compile cython code
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("lib2",
              ["lib2.pyx"],
              language="c++", 
              libraries=["m"],
              extra_compile_args = ["-std=c++11", "-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]

setup( 
  name = "lib2",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)

"""
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules = cythonize(Extension(
    "lib2",                # the extension name
     sources=["lib2.pyx"], # the Cython source and
                           # additional C++ source files
     language="c++",       # generate and compile C++ code
     extra_compile_args=["-std=c++11", "-O3", "-ffast-math", "-march=native", "-fopenmp"],
     extra_link_args=['-fopenmp'],
     libraries=["m"],
      )),
      cmdclass = {"build_ext": build_ext},)
"""


