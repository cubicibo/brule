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

from typing import Any, Callable, Optional, Union
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

@dataclass
class _CodecFunctions:
    encode: Callable[[npt.NDArray[np.uint8]], Union[bytes, bytearray]]
    decode: Callable[[Union[bytes, bytearray]], npt.NDArray[np.uint8]]

def _try_numba() -> _CodecFunctions:
    #If numba is available, provide compiled functions for the encoder/decoder
    from numba import njit

    @njit(fastmath=True)
    def njit_encode_rle(bitmap: npt.NDArray[np.uint8]) -> list[np.uint8]:
        """
        Encode a 2D map using the RLE defined in 'US 7912305 B1' patent.
        :param bitmap:    Palette mapped image to encode (2d array)
        :return:          Encoded data (vector)
        """
        i, j = 0, 0
        rle_data = [np.uint8(x) for x in range(0)]

        height, width = bitmap.shape
        assert width <= 16383, "Bitmap too large."

        while i < height:
            color = bitmap[i, j]
            prev_j = j
            while (j := j+1) < width and bitmap[i, j] == color: pass

            dist = j - prev_j
            if color == 0:
                if dist > 63:
                    rle_data += [0x00, 0x40 | ((dist >> 8) & 0x3F), dist & 0xFF]
                else:
                    rle_data += [0x00, dist & 0x3F]
            else:
                if dist > 63:
                    rle_data += [0x00, 0xC0 | ((dist >> 8) & 0x3F), dist & 0xFF, color]
                elif dist > 2:
                    rle_data += [0x00, 0x80 | (dist & 0x3F), color]
                else:
                    rle_data += [color] * dist
            if j == width:
                j = 0
                i += 1
                rle_data += [0x00, 0x00]
        return rle_data

    @njit(fastmath=True)
    def njit_decode_rle(rle_data: Union[list[np.uint8], npt.NDArray[np.uint8], bytes], width: int, height: int, check_rle: bool = False) -> npt.NDArray[np.uint8]:
        i, j, k = 0, -1, -2
        # RLE is terminated by new line command ([0x00, 0x00]), we can ignore it.
        len_rle = len(rle_data) - 2
        bitmap = np.zeros((height, width), np.uint8)
        line_length = 0

        while k < len_rle:
            if i % width == 0:
                if line_length > 0:
                    assert 0 == rle_data[k] == rle_data[k+1], "RLE line terminator not found at given width."
                    assert not check_rle or line_length <= width, "Illegal RLE line width."
                    line_length = 0
                i = 0
                j += 1
                k += 2
            k_start = k
            byte = rle_data[k]
            if byte == 0:
                byte = rle_data[(k:=k+1)]
                if byte & 0x40:
                    length = ((byte & 0x3F) << 8) | rle_data[(k:=k+1)]
                else:
                    length = byte & 0x3F
                if byte & 0x80:
                    bitmap[j, i:(i:=i+length)] = rle_data[(k:=k+1)]
                else:
                    bitmap[j, i:(i:=i+length)] = 0
            else:
                bitmap[j, i:(i:=i+1)] = byte
            k+=1
            line_length += (k - k_start)
        return bitmap
    return _CodecFunctions(njit_encode_rle, njit_decode_rle)
####

#%%
class Brule:
    _numba_codec = None
    _cimpl_codec = None
    
    @classmethod
    def _setup(cls) -> None:
        try:
            from . import _brule
            cls._cimpl_codec = _CodecFunctions(_brule.encode, _brule.decode)
        except (ImportError, ModuleNotFoundError): ...
        try:
            cls._numba_codec = _try_numba()
        except (ImportError, ModuleNotFoundError): ...

    @classmethod
    def get_capabilities(cls) -> str:
        cap  = ['C'] * (cls._cimpl_codec is not None)
        cap += ['numba'] * (cls._numba_codec is not None)
        return cap + ['python']

    @classmethod
    def _downgrade_impl(cls, impl: str) -> bool:
        if impl == 'numba':
            assert cls._numba_codec is not None
            cls._cimpl_codec = None
            return True
        if impl == 'python':
            cls._cimpl_codec = None
            cls._numba_codec = None
            return True
        return False

    @classmethod
    def encode(cls, bitmap: npt.NDArray[np.uint8]) -> bytes:
        """
        Encode a 2D map using the RLE defined in 'US 7912305 B1' patent.
        :param bitmap:    Palette mapped image to encode (2d array)
        :return:          Encoded bytes (vector)
        """
        if cls._cimpl_codec is not None:
            if not bitmap.flags['C_CONTIGUOUS']:
                bitmap = np.ascontiguousarray(bitmap)
            return cls._cimpl_codec.encode(bitmap)
        if cls._numba_codec is not None:
            return cls._numba_codec.encode(bitmap)

        rle_data = []
        i, j = 0, 0

        height, width = bitmap.shape
        assert width <= 16383, "Bitmap too large."

        while i < height:
            color = bitmap[i, j]
            prev_j = j
            while (j := j+1) < width and bitmap[i, j] == color: pass

            dist = j - prev_j
            if color == 0:
                if dist > 63:
                    rle_data += [0x00, 0x40 | ((dist >> 8) & 0x3F), dist & 0xFF]
                else:
                    rle_data += [0x00, dist & 0x3F]
            else:
                if dist > 63:
                    rle_data += [0x00, 0xC0 | ((dist >> 8) & 0x3F), dist & 0xFF, color]
                elif dist > 2:
                    rle_data += [0x00, 0x80 | (dist & 0x3F), color]
                else:
                    rle_data += [color] * dist
            if j == width:
                j = 0
                i += 1
                rle_data += [0x00, 0x00]
        return bytes(rle_data)

    @classmethod
    def decode(cls, data: Union[bytes, bytearray], width: Optional[int] = None, height: Optional[int] = None, check_rle: bool = False) -> npt.NDArray[np.uint8]:
        """
        Decode a RLE object, as defined in 'US 7912305 B1' patent.
        :param data:  Bytes to decode
        :return: numpy bitmap
        """
        if isinstance(data, list):
            data = bytearray(data)

        if cls._cimpl_codec is not None and check_rle is False:
            return cls._cimpl_codec.decode(data)
        if cls._numba_codec is not None and width and height:
            return cls._numba_codec.decode(np.frombuffer(data, dtype=np.uint8), width, height, check_rle)

        if check_rle:
            assert width is not None and width > 0

        k = 0
        len_data = len(data)
        bitmap = []
        line = []
        rle_line_length = 0

        while k < len_data:
            k_start = k
            byte = data[k]
            if byte == 0:
                byte = data[(k:=k+1)]
                if byte == 0:
                    assert not check_rle or rle_line_length <= width
                    bitmap.append(line)
                    line = []
                    rle_line_length = 0
                    k += 1
                    continue
                if byte & 0x40:
                    length = ((byte & 0x3F) << 8) | data[(k:=k+1)]
                else:
                    length = byte & 0x3F
                if byte & 0x80:
                    line += [data[(k:=k+1)]]*length
                else:
                    line += [0]*length
            else:
                line.append(byte)
            k+=1
            rle_line_length += (k - k_start)
        return np.asarray(bitmap, np.uint8)
####

#init module
Brule._setup()
