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

from typing import Any, Callable, Optional, Union, TypeVar
from numpy import typing as npt
import numpy as np

_Coordinates = TypeVar('Coordinates')

class _PyLayoutEngine:
    def __init__(self, shape: tuple[int, int]) -> None:
        self._screen = np.zeros(shape[::-1], np.uint8)
        self._last_raw_container = None

    def add(self, xym: tuple[int, int, npt.NDArray[np.uint8]]) -> None:
        xp, yp, mask = xym
        assert self._screen is not None, "destroyed instance"
        self._screen[yp:yp+mask.shape[0], xp:xp+mask.shape[1]] |= (mask > 0)

    @staticmethod
    def _crop(mask: npt.NDArray[np.uint8], yo: int, xo: int) -> tuple[int, int, int, int]:
        rmin, rmax = np.where(np.any(mask, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
        return cmin+xo, rmin+yo, cmax+1+xo, rmax+1+yo

    def get_container(self, iid: int = None) -> tuple[int, int, int, int]:
        assert self._last_raw_container is not None
        return self._last_raw_container

    def find(self, iid: int = None) -> tuple[_Coordinates,_Coordinates,_Coordinates, int]:
        cls = self.__class__
        assert self._screen is not None, "destroyed instance"
        assert np.any(self._screen), "No screen overlay in epoch."

        f_area = lambda xyc: (xyc[2]-xyc[0])*(xyc[3]-xyc[1])
        is_vertical = -1

        cmin, rmin, cmax, rmax = cls._crop(self._screen, 0, 0)
        ocbox = [cmin, rmin, cmax, rmax]

        diffx = max(0, 8 - (ocbox[2]-ocbox[0]))
        if ocbox[0] > diffx:
            ocbox[0] = cmin-diffx
        elif diffx > 0:
            ocbox[2] = cmax+diffx

        diffy = max(0, 8 - (ocbox[3]-ocbox[1]))
        if ocbox[1] > diffy:
            ocbox[1] = rmin-diffy
        elif diffy > 0:
            ocbox[3] = rmax+diffy
        ocbox = tuple(ocbox)
        self._last_raw_container = ocbox

        best_score = f_area(ocbox)
        best_wds = (ocbox, ocbox)

        rmin -= min(7, rmin)
        rmax += min(7, self._screen.shape[0]-rmax)
        cmin -= min(7, cmin)
        cmax += min(7, self._screen.shape[1]-cmax)
        cbox = (cmin, rmin, cmax, rmax)

        f_score = lambda wds: sum(map(f_area, wds))

        for xk in range(cmin+8, cmax-8):
            lwd = (cls._crop(self._screen[rmin:rmax, cmin:xk], rmin, cmin), cls._crop(self._screen[rmin:rmax, xk:cmax], rmin, xk))
            if best_score > (new_score := f_score(lwd)):
                best_score = new_score
                best_wds = lwd
                is_vertical = False

        for yk in range(rmin+8, rmax-8):
            lwd = (cls._crop(self._screen[rmin:yk, cmin:cmax], rmin, cmin), cls._crop(self._screen[yk:rmax, cmin:cmax], yk, cmin))
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

    @classmethod
    def init(cls, shape: tuple[int, int]) -> '_PyLayoutEngine':
        return cls(shape)

    def destroy(self) -> None:
        self._screen = None

    def __del__(self) -> None:
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

    def __init__(self, shape: tuple) -> None:
        assert 2 == len(shape)

        self.__engine = __class__._internals
        self._iinst = self.__engine.init(shape)
        self._shape = shape
        self._ready = True

    def add_to_layout(self, xp: int, yp: int, mask: npt.NDArray[np.uint8]) -> None:
        assert self._ready
        assert 2 == len(mask.shape) and mask.dtype == np.uint8
        assert all(map(lambda x: 0 < x[2]+x[0] <= x[1], zip(mask.shape[::-1], self._shape, (xp, yp))))
        if isinstance(self._iinst, _PyLayoutEngine):
            self._iinst.add((xp, yp, mask))
        else:
            self.__engine.add((self._iinst, xp, yp, mask))

    def get_layout(self) -> tuple[_Coordinates, _Coordinates, _Coordinates, int]:
        assert self._ready
        return self.__engine.find(self._iinst)

    def get_raw_container(self) -> tuple[int, int, int, int]:
        assert self._ready
        return self.__engine.get_container(self._iinst)

    def reset(self) -> None:
        if self._ready:
            self.destroy()
        self._iinst = self.__engine.init(self._shape)
        self._ready = True

    def destroy(self) -> None:
        if self._ready:
            self.__engine.destroy(self._iinst)
        self._ready = False

    def __del__(self) -> None:
        self.destroy()

####
#init module
LayoutEngine._setup()
