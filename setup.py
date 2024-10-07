#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2024 cubicibo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import setuptools
import sysconfig
import numpy as np
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError
from setuptools.errors import ExecError
from setuptools.errors import PlatformError

NAME = 'brule'

meta = {}
with open(f"src/{NAME}/__metadata__.py") as f:
    exec(f.read(), meta)

# This chunk of code is stolen from https://github.com/pallets/markupsafe, all credits to them!!
class BuildFailed(Exception):
    pass


class ve_build_ext(build_ext):
    """This class allows C extension building to fail."""

    def run(self):
        try:
            super().run()
        except PlatformError as e:
            raise BuildFailed() from e

    def build_extension(self, ext):
        #extra_compile_args = {
        #    'unix': ['-O2'],
        #}
        #ext.extra_compile_args += extra_compile_args.get(self.compiler.compiler_type, [])

        try:
            super().build_extension(ext)
        except (CCompilerError, ExecError, PlatformError) as e:
            raise BuildFailed() from e
        except ValueError as e:
            import sys
            # this can happen on Windows 64 bit, see Python issue 7511
            if "'path'" in str(sys.exc_info()[1]):  # works with Python 2 and 3
                raise BuildFailed() from e
            raise

def show_message(*lines):
    print("=" * 74)
    for line in lines:
        print(line)
    print("=" * 74)

if __name__ == "__main__":
    try:
        extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    except AttributeError:
        extra_compile_args = []

    brule_codec = setuptools.Extension(
        f"{NAME}._{NAME}",
        sources=[f"src/{NAME}/_brule.cc", f"src/{NAME}/_librle.cc"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
    )

    layout_eng = setuptools.Extension(
        f"{NAME}._layouteng",
        sources=[f"src/{NAME}/_layouteng.cc"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
    )

    modules = [brule_codec, layout_eng]

    def run_setup(modules):
        setuptools.setup(
            name=NAME,
            version=meta['__version__'],
            author=meta['__author__'],
            description="Bitmap RUn LEngth module",
            long_description=open("README.md").read(),
            long_description_content_type="text/markdown",
            license="MIT",
            packages=setuptools.find_packages("src"),
            package_dir={'': 'src'},
            cmdclass={"build_ext": ve_build_ext},
            ext_modules=modules,
            classifiers=[
                'Development Status :: 3 - Alpha',
                'Intended Audience :: Developers',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python :: 3.9',
            ],
            python_requires='>=3.11',
            install_requires=["numpy>=2.0.1", "numba"],
            zip_safe=False,
        )
    ####run_setup
    try:
        run_setup(modules)
    except BuildFailed:
        show_message(
            "WARNING: Failed to compile optimised implementation."
            "Retrying the build with only pure Python implementation.",
        )
        run_setup([])
