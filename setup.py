from Cython.Build import cythonize
from setuptools import setup

setup(
    packages=["anomalib_app/core", "anomalib_app/ui"],
    ext_modules=cythonize(
        [
            "anomalib_app/core/*.py",
            "anomalib_app/ui/*.py",
        ],
        compiler_directives={"language_level": "3"},
    ),
)
