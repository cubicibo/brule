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

import numpy as np
from numpy.typing import NDArray

#%%
class HexTree:
    _cimpl = None

    @classmethod
    def _setup(cls) -> None:
        try:
            from . import _hextree
            cls._cimpl = _hextree.quantize
        except (ImportError, ModuleNotFoundError):
            ...

    @classmethod
    def get_capabilities(cls) -> str:
        return ['C'] if cls._cimpl is not None else [None]

    @classmethod
    def _downgrade_impl(cls, impl: str) -> bool:
        return False

    @classmethod
    def quantize(cls, rgba: NDArray[np.uint8], colours: int = 255) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        :param RGBA:    rgba image to quantize
        :param colours: maximum number of palette entries
        :return:        bitmap, palette as numpy arrays.
        """
        assert cls._cimpl is not None, "Cannot use HexTree without its C extension."
        h, w, d = rgba.shape
        assert d == 4, "Expected RGBA"
        assert rgba.dtype in [np.uint8, np.int8], "Expected 8-bit per component"
        assert 16 <= colours <= 256
        #C implementation
        rgbaf = rgba.reshape((-1,))
        if not rgbaf.flags['C_CONTIGUOUS'] and not rgbaf.flags['F_CONTIGUOUS']:
            rgbaf = np.ascontiguousarray(rgbaf)
        bitmap, palette = cls._cimpl((colours, rgbaf, w))
        return bitmap.reshape((h, w)), palette
####

#init module
HexTree._setup()
