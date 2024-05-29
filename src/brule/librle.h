/**
 MIT License

 Copyright (c) 2024, cubicibo

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
*/

#pragma once

#define EXPORT_DYLIB (0)

#if defined(_WIN32) && EXPORT_DYLIB
# ifdef LIBRLE_EXPORTS
#  define LRB_EXPORT_API __declspec(dllexport)
# else
#  define LRB_EXPORT_API __declspec(dllimport)
# endif
#else
# define LRB_EXPORT_API
#endif

#define LRB_VERSION ((int)1)
#define LRB_VERSION_STR "0.0.1"
#define LRB_AUTHOR "cubicibo"

#if EXPORT_DYLIB && (defined(__GNUC__) || defined (__llvm__))
#define LRB_NONNULL __attribute__((nonnull))
#define LRB_FRESULT __attribute__((warn_unused_result))
#else
#define LRB_NONNULL
#define LRB_FRESULT
#endif

typedef struct lrb_rle_result {
	unsigned int length;
	unsigned char* data;
} lrb_rle_result;

typedef struct lrb_bitmap_result {
	unsigned int width;
	unsigned int height;
	unsigned char* data;
} lrb_bitmap_result;

typedef enum lrb_error {
	LRB_OK = 0,
	LRB_ENOMEM = 100,
	LRB_INVALID_PTR,
	LRB_USED_PTR,
	LRB_INVALID_DATA,
	LRB_INVALID_VALUE,
	LRB_INVALID_DIMENSION,
	LRB_UNSUPPORTED,
} lrb_error;

// Encode bitmap to RLE 
LRB_EXPORT_API LRB_FRESULT lrb_error lrb_encode_bitmap(const void* bitmap, const unsigned int width, const unsigned int height, lrb_rle_result* rle_res) LRB_NONNULL;

// Decode RLE to bitmap 
LRB_EXPORT_API LRB_FRESULT lrb_error lrb_decode_rle(const void* data, const unsigned int length, lrb_bitmap_result* bitmap_res) LRB_NONNULL;

// Destroy a bitmap (decode) result
LRB_EXPORT_API LRB_FRESULT lrb_error lrb_destroy_bitmap(lrb_bitmap_result* bitmap_res) LRB_NONNULL;

// Destroy a rle (encode) result
LRB_EXPORT_API LRB_FRESULT lrb_error lrb_destroy_rle(lrb_rle_result* rle_res) LRB_NONNULL;

// lib version
LRB_EXPORT_API int lrb_version(void);
