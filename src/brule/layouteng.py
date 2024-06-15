#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2024 cibo
This file is part of SUPer <https://github.com/cubicibo/SUPer>.

SUPer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SUPer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SUPer.  If not, see <http://www.gnu.org/licenses/>.
"""

from typing import Any, Callable, Optional, Union, TypeVar
from numpy import typing as npt
import numpy as np

_Coordinates = TypeVar('Coordinates')

class _PyLayoutEngine:
    def __init__(self, shape: tuple[int, int]) -> None:
        self._screen = None
        self.init(shape)

    def add(self, xp: int, yp: int, mask: npt.NDArray[np.uint8]) -> None:
        assert self._screen is not None
        self._screen[yp:yp+mask.shape[0], xp:xp+mask.shape[1]] |= (mask > 0)

    @staticmethod
    def _crop(mask: npt.NDArray[np.uint8]) -> tuple[int, int, int, int]:
        rmin, rmax = np.where(np.any(mask, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
        return cmin, rmin, cmax+1, rmax+1

    def find(self) -> tuple[_Coordinates,_Coordinates,_Coordinates, int]:
        cls = self.__class__
        assert self._screen is not None
        assert np.any(self._screen), "No screen overlay in epoch."

        f_area = lambda xyc: (xyc[2]-xyc[0])*(xyc[3]-xyc[1])
        is_vertical = -1

        cmin, rmin, cmax, rmax = cls._crop(self._screen)
        ocbox = (cmin, rmin, cmax, rmax)
        best_score = f_area(ocbox)
        best_wds = (ocbox, ocbox)

        rmin -= min(7, rmin)
        rmax += min(7, self._screen.shape[0]-rmax)
        cmin -= min(7, cmin)
        cmax += min(7, self._screen.shape[1]-cmax)
        cbox = (cmin, rmin, cmax, rmax)

        f_score = lambda wds: sum(map(f_area, wds))

        for xk in range(cmin+8, cmax-8):
            lwd = (cls._crop(self._screen[rmin:rmax, cmin:xk]), cls._crop(self._screen[rmin:rmax, xk:cmax]))
            if best_score > (new_score := f_score(lwd)):
                best_score = new_score
                best_wds = lwd
                is_vertical = False

        for yk in range(rmin+8, rmax-8):
            lwd = (cls._crop(self._screen[rmin:yk, cmin:cmax]), cls._crop(self._screen[yk:rmax, cmin:cmax]))
            if best_score > (new_score := f_score(lwd)):
                best_score = new_score
                best_wds = lwd
                is_vertical = True

        if is_vertical == -1:
            cbox = ocbox
            best_wds = (ocbox, ocbox)

        final_wds = []
        for k, wd in enumerate(best_wds):
            final_wds.append((wd[0]-cbox[0], wd[1]-cbox[1], wd[2]-cbox[0], wd[3]-cbox[1]))

        return cbox, final_wds[0], final_wds[1], is_vertical

    def init(self, shape: Optional[tuple[int, int]]) -> None:
        if isinstance(shape, tuple):
            self._screen = np.zeros(shape[::-1], np.uint8)
        else:
            self._screen = None
####

class LayoutEngine:
    _internals = None
    _has_c_ext = False
    @classmethod
    def _setup(cls):
        try:
            from . import _layouteng
        except (ModuleNotFoundError, ImportError):
            cls._internals = _PyLayoutEngine
        else:
            cls._internals = _layouteng
            cls._has_c_ext = True

    @classmethod
    def get_capabilities(cls) -> list[str]:
        return ['C']*cls._has_c_ext + ['Python']

    def __init__(self, shape: Optional[tuple]) -> None:
        assert 2 == len(shape)
        self._setup_internals(shape)

    def _setup_internals(self, shape: Optional[tuple]) -> None:
        if shape is not None and not isinstance(shape, tuple):
            shape = tuple(shape)
        if isinstance(shape, tuple):
            self._shape = shape
        else:
            assert shape is None

        if self._internals == _PyLayoutEngine:
            self._iinst = _PyLayoutEngine(shape)
        else:
            self._iinst = self._internals

        try:
            self._iinst.init(shape)
        except:
            self._ready = False
        else:
            self._ready = isinstance(shape, tuple)

    def add_to_layout(self, xp: int, yp: int, mask: npt.NDArray[np.uint8]) -> None:
        assert self._ready
        assert 2 == len(mask.shape) and mask.dtype == np.uint8
        assert all(map(lambda x: 0 < x[2]+x[0] <= x[1], zip(mask.shape[::-1], self._shape, (xp, yp))))
        self._iinst.add((xp, yp, mask))

    def get_layout(self) -> tuple[_Coordinates, _Coordinates, _Coordinates, int]:
        assert self._ready
        return self._iinst.find()

    def reset(self) -> None:
        self._setup_internals(self._shape)

    def destroy(self) -> None:
        self._setup_internals(None)

    def __del__(self):
        self.destroy()
####
#init module
LayoutEngine._setup()
