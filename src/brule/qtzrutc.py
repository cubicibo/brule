#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2024-2025 cubicibo

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

from typing import Optional

import numpy as np
import cv2
from numpy.typing import NDArray

from anytree import NodeMixin
from anytree.search import findall

class _PyKDMeans:
    def __init__(self, image: NDArray[np.uint8]) -> None:
        self._img = image

    def quantize(self, nk: int = 255) -> NDArray[np.uint8]:
        assert 8 <= nk <= 256
        h, w = self._img.shape[:2]
        # Use PIL to get approximate number of clusters
        flat_img = np.float32(np.asarray(self._img).reshape((-1, 4)))/255.0

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 11, 1.0)
        _, label, center = cv2.kmeans(flat_img, nk, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        return label.reshape(h, w).astype(np.uint8), np.round(np.clip(center, 0.0, 1.0)*255).astype(np.uint8)
#%%
class QtzrUTC:
    _cimpl = None

    @classmethod
    def _setup(cls) -> None:
        try:
            from . import _qtzrutc
            cls._cimpl = _qtzrutc.quantize
        except (ImportError, ModuleNotFoundError):
            ...

    @classmethod
    def get_capabilities(cls) -> str:
        cap  = ['C'] * (cls._cimpl is not None)
        return cap + ['Python']

    @classmethod
    def _downgrade_impl(cls, impl: str) -> bool:
        if impl == 'python':
            cls._cimpl = None
            return True
        elif impl == "c":
            cls._setup()
            return cls._cimpl is not None
        return False

    @classmethod
    def quantize(cls, rgba: NDArray[np.uint8], colours: int = 255) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        :param RGBA:    rgba image to quantize
        :param colours: maximum number of palette entries
        :return:        bitmap, palette as numpy arrays.
        """
        h, w, d = rgba.shape
        assert d == 4, "Expected RGBA"
        assert rgba.dtype in [np.uint8, np.int8], "Expected 8-bit per component"
        assert 8 <= colours <= 256
        #C implementation
        if cls._cimpl is not None:
            rgbaf = rgba.reshape((-1,))
            if not rgbaf.flags['C_CONTIGUOUS'] and not rgbaf.flags['F_CONTIGUOUS']:
                rgbaf = np.ascontiguousarray(rgbaf)
            bitmap, palette = cls._cimpl((colours, rgbaf, w))
            bitmap = bitmap.reshape((h, w))
        else:
            bitmap, palette = _PyKDMeans(rgba).quantize(colours)
        return bitmap, palette
####

#init module
QtzrUTC._setup()
