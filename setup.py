"""
MIT License

Copyright (c) 2024-26 cubicibo

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

import os
import setuptools
import sysconfig
import numpy as np
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError
from setuptools.errors import ExecError
from setuptools.errors import PlatformError

NAME = 'brule'

# This fallback build system is taken from  https://github.com/pallets/markupsafe (BSD-3 license)
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
        try:
            super().build_extension(ext)
        except (CCompilerError, ExecError, PlatformError) as e:
            raise BuildFailed() from e
        except ValueError as e:
            import sys
            # this can happen on Windows 64 bit, see Python issue 7511
            if "'path'" in str(sys.exc_info()[1]):
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

    include_dirs = [np.get_include()]
    pthread_flag = ["-pthread"] if os.name == 'posix' else []

    brule_codec = setuptools.Extension(
        f"{NAME}._brule",
        sources=[f"src/{NAME}/rle/_brule.cc", f"src/{NAME}/rle/_librle.cc"],
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=extra_compile_args + pthread_flag,
    )

    layout_eng = setuptools.Extension(
        f"{NAME}._layouteng",
        sources=[f"src/{NAME}/layouteng/_layouteng.cc"],
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=extra_compile_args,
    )

    hextree = setuptools.Extension(
        f"{NAME}._hextree",
        sources=[f"src/{NAME}/hextree/_hextree.cc"],
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=extra_compile_args,
    )

    qtzrutc = setuptools.Extension(
        f"{NAME}._qtzrutc",
        sources=[f"src/{NAME}/qtzr/_qtzrutc.cc"],
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=extra_compile_args,
    )

    modules = [brule_codec, layout_eng, hextree, qtzrutc]

    def run_setup(modules):
        setuptools.setup(
            cmdclass={"build_ext": ve_build_ext},
            ext_modules=modules,
        )
    ####

    try:
        run_setup(modules)
    except BuildFailed:
        show_message(
            "WARNING: Failed to compile optimised implementation."
            "Retrying the build with only pure Python implementation.",
        )
        run_setup([])
