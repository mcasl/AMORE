from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="pyAmore",
    ext_modules=cythonize('**/*.pyx', annotate=True),
)
