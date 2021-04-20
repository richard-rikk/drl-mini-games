from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    #Extension("main",  ["main.py"]),
    Extension("cpy.trainer",     ["./games/trainers.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cpy.constants",   ["./games/constants.pyx"]),
]

setup(
    name = 'My Program',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)