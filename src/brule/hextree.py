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

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from anytree import NodeMixin
from anytree.search import findall

try:
    from numba import njit, prange
    @njit(fastmath=True, parallel=True)
    def _vect_mask(colours: NDArray[np.uint8]) -> NDArray[np.uint8]:
        indexs = np.zeros((colours.shape[0], 8), np.uint8)
        for depth in prange(8):
            mask = (colours >> (7 - depth)) & 0x01
            mask[:, 0] <<= 3
            mask[:, 1] <<= 2
            mask[:, 2] <<= 1
            indexs[:, depth] = np.sum(mask, axis=1)
        return indexs
except (ImportError, ModuleNotFoundError):
    def _vect_mask(colours: NDArray[np.uint8]) -> NDArray[np.uint8]:
        indexs = np.zeros((colours.shape[0], 8), np.uint8)
        for depth in range(8):
            mask = (colours >> (7 - depth)) & 0x01
            mask[:, 0] <<= 3
            mask[:, 1] <<= 2
            mask[:, 2] <<= 1
            indexs[:, depth] = np.sum(mask, axis=1)
        return indexs

class _RGBA(NodeMixin):
    def __init__(self, rgba: NDArray[np.uint8], count: int, index: Optional[int] = None, parent: Optional['_RGBA'] = None) -> None:
        self.parent = parent
        self.index = index
        self._rgbaf = rgba.astype(float)
        self._count = count

    def fold_children(self) -> None:
        assert not self.is_leaf
        self._rgbaf *= 0
        for child in tuple(self.children):
            self._rgbaf += child._rgbaf * (child._count / self._count)
            child.parent = None

    def finalize(self) -> None:
        self._rgbaf = self._rgbaf.round().astype(np.uint8)

    def get_colour(self) -> NDArray[np.uint8]:
        return self._rgbaf

class _PyHexTree:
    def __init__(self, image: NDArray[np.uint8]) -> None:
        assert image.shape[2] == 4

        self._root = _RGBA(np.array(np.nan), 0)
        self._occurences, self._colours, self._hbd = __class__._preprocess(image)
        self._vect_index = _vect_mask(self._colours)

        for clr, branches, count in zip(self._colours, self._vect_index, self._occurences.values()):
            self.add_leaf(clr, count[0], branches)

    @staticmethod
    def _preprocess(img: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint32]]:
        occs = {}
        clrs = []
        cid = -1

        hbd = np.zeros(img.shape[0]*img.shape[1], np.uint32)
        flattened_img = img.reshape(-1, img.shape[2])
        for pxid, rgba in enumerate(flattened_img):
            brgba = rgba.tobytes()
            try:
                vp = occs[brgba]
                vp[0] += 1
            except KeyError:
                occs[brgba] = vp = [1, cid := cid + 1]
                clrs.append(rgba)
            hbd[pxid] = vp[1]
        return occs, np.stack(clrs, dtype=np.uint8), hbd.reshape(img.shape[:2])

    def add_leaf(self, colour: NDArray[np.uint8], count: int, branches: NDArray[np.uint8]):
        assert len(branches) == 8

        root = self._root
        for branch_id in branches:
            assert 0 <= branch_id < 16
            child = next(filter(lambda x: x.index == branch_id, root.children), None)
            if child is None:
                root = _RGBA(colour, count, branch_id, parent=root)
            else:
                root = child
                root._count += count
        ####

    def reduce(self, max_leafs: int = 255) -> None:
        assert 16 <= max_leafs <= 256
        excess = len(self._root.leaves) - max_leafs
        if excess <= 0:
            return

        f_select = lambda d: findall(self._root, lambda nd: nd.depth == d)
        for depth in iter(range(7, 0, -1)):
            sparents = sorted(f_select(depth), key=lambda n: n._count, reverse=True)

            while len(sparents) and excess > 0:
                parent = sparents.pop()
                excess -= (len(parent.children) - 1)
                parent.fold_children()
        ####
        for leaf in self._root.leaves:
            leaf.finalize()
    ####

    def quantize(self, n_colours: int = 255) -> NDArray[np.uint8]:
        self.reduce(n_colours)
        palette = {leaf.get_colour().tobytes(): k  for k, leaf in enumerate(self._root.leaves)}
        cmap = np.zeros((len(self._colours),), np.uint8)

        for color_idx, (branch, colour) in enumerate(zip(self._vect_index, self._colours)):
            root = self._root
            while len(root.children):
                root = next(filter(lambda child: child.index == branch[root.depth], root.children))
            cmap[color_idx] = palette[root.get_colour().tobytes()]

        return cmap[self._hbd], np.frombuffer(b''.join(palette), np.uint8).reshape((-1, 4))

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
        Encode a 2D map using the RLE defined in 'US 7912305 B1' patent.
        :param RGBA:    rgba image to quantize
        :param colours: maximum number of palette entries
        :return:        bitmap, palette as numpy arrays.
        """
        h, w, d = rgba.shape
        assert d == 4, "Expected RGBA"
        assert rgba.dtype in [np.uint8, np.int8], "Expected 8-bit per component"
        assert 16 <= colours <= 256
        #C implementation
        if cls._cimpl is not None:
            rgbaf = rgba.reshape((-1,))
            if not rgbaf.flags['C_CONTIGUOUS'] and not rgbaf.flags['F_CONTIGUOUS']:
                rgbaf = np.ascontiguousarray(rgbaf)
            bitmap, palette = cls._cimpl((colours, rgbaf, w))
            bitmap = bitmap.reshape((h, w))
        else:
            bitmap, palette = _PyHexTree(rgba).quantize(colours)
        return bitmap, palette
####

#init module
HexTree._setup()
