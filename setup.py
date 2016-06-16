from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("pyAmore.cython.activation_functions", ["pyAmore/cython/activation_functions.pyx"],
              libraries=["m"]),
    # Everything but primes.pyx is included here.
    Extension("*", ["**/*.pyx"]),
]
setup(
    name="pyAmore",
    ext_modules=cythonize(extensions, annotate=True),
)
