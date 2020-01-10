from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("earthmath", ["earthmath.c"])

setup(name = "earthmath", ext_modules=[extension_mod])
