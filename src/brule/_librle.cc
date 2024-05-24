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

#include <stdlib.h>
#include <string.h>
#include "librle.h"

lrb_error lrb_encode_bitmap(const void* bitmap, const unsigned int width, const unsigned int height, lrb_rle_result* rle_res)
{
	if (!rle_res)
		return LRB_INVALID_PTR;

	if (!bitmap)
		return LRB_INVALID_PTR;

	if (width < 8 || height < 8 || width > 4096 || height > 4096)
		return LRB_INVALID_DIMENSION;

	const unsigned int step = 16384, area = width * height;
	const unsigned char* cbit = (const unsigned char*)bitmap;

	unsigned char* tmpptr;
	unsigned char color;
	unsigned long line_index, j, start_point, distance, allocated_size;

	allocated_size = step;
	if (rle_res->data)
		return LRB_INVALID_PTR;
	rle_res->data = (unsigned char*)calloc(allocated_size, sizeof(unsigned char));
	if (!rle_res->data)
		return LRB_ENOMEM;
	rle_res->length = 0;

	for (line_index = 0; line_index < area; line_index += width) {
		j = 0;
		do {
			start_point = j;
			color = cbit[line_index + j];
			while ((++j < width) && (color == cbit[j + line_index]));

			distance = j - start_point;
			if (0 == color) {
				rle_res->data[rle_res->length++] = 0;
				if (distance > 63) {
					rle_res->data[rle_res->length++] = 0x40 | ((unsigned char)(distance >> 8) & 0x3F);
					rle_res->data[rle_res->length++] = (unsigned char)distance & 0xFF;
				}
				else {
					rle_res->data[rle_res->length++] = (unsigned char)distance & 0x3F;
				}
			}
			else {
				if (distance > 63) {
					rle_res->data[rle_res->length++] = 0;
					rle_res->data[rle_res->length++] = 0xC0 | ((unsigned char)(distance >> 8) & 0x3F);
					rle_res->data[rle_res->length++] = distance & 0xFF;
					rle_res->data[rle_res->length++] = color;
				}
				else if (distance > 2) {
					rle_res->data[rle_res->length++] = 0;
					rle_res->data[rle_res->length++] = 0x80 | ((unsigned char)distance & 0x3F);
					rle_res->data[rle_res->length++] = color;
				}
				else {
					rle_res->data[rle_res->length++] = color;
					if (distance == 2)
						rle_res->data[rle_res->length++] = color;
				}
			}
			if (allocated_size < rle_res->length + 6) {
				allocated_size += step;
				tmpptr = (unsigned char*)realloc(rle_res->data, allocated_size * sizeof(unsigned char));
				if (!tmpptr || allocated_size > 8 << 20) {
					free(rle_res->data);
					memset(rle_res, 0, sizeof(lrb_rle_result));
					return LRB_ENOMEM;
				}
				rle_res->data = tmpptr;
			}
		} while (j < width);
		rle_res->data[rle_res->length++] = 0;
		rle_res->data[rle_res->length++] = 0;
	}
	return LRB_OK;
}

lrb_error lrb_decode_rle(const void* data, const unsigned int length, lrb_bitmap_result* bitmap_res)
{
	if (!data || !bitmap_res)
		return LRB_INVALID_PTR;
	if (length < 2)
		return LRB_INVALID_VALUE;

	const unsigned long step = 16384;
	unsigned long allocated_size = step;
	if (bitmap_res->data)
		return LRB_INVALID_PTR;

	bitmap_res->data = (unsigned char*)calloc(allocated_size, sizeof(unsigned char));
	if (!bitmap_res->data)
		return LRB_ENOMEM;

	const unsigned char* rle = (const unsigned char*)data;
	unsigned char* tmpptr;
	unsigned int i = 0, line_width = 0, repeat_len, rle_cmd;
	unsigned long j = 0, line_index = 0;
	unsigned char color;

	do {
		if (rle[i]) {
			color = rle[i];
			repeat_len = 1;
		}
		else {
			rle_cmd = rle[++i];
			if (!rle_cmd) {
				if (!line_width) {
					line_width = j;
				}
				else if (j != line_width) {
					free(bitmap_res->data);
					memset(bitmap_res, 0, sizeof(lrb_bitmap_result));
					return LRB_INVALID_DATA;
				}
				repeat_len = j = 0;
				line_index += line_width;
			}
			else {
				repeat_len = (rle_cmd & 0x40) ? (((rle_cmd & 0x3F) << 8) | rle[++i]) : (rle_cmd & 0x3F);
				color = (rle_cmd & 0x80) ? rle[++i] : 0;
			}
		}
		if (line_index + j + repeat_len >= allocated_size) {
			allocated_size += step;
			tmpptr = (unsigned char*)realloc(bitmap_res->data, allocated_size * sizeof(unsigned char));
			if (!tmpptr || allocated_size > 8 << 20) {
				free(bitmap_res->data);
				memset(bitmap_res, 0, sizeof(lrb_bitmap_result));
				return LRB_ENOMEM;
			}
			bitmap_res->data = tmpptr;
		}
		if (repeat_len)
			memset(&bitmap_res->data[line_index + j], color, repeat_len);
		j += repeat_len;
	} while (++i < length);
	if (i > length) {
		free(bitmap_res->data);
		memset(bitmap_res, 0, sizeof(lrb_bitmap_result));
		return LRB_INVALID_DATA;
	}
	bitmap_res->height = line_index / line_width;
	bitmap_res->width = line_width;
	return LRB_OK;
}

lrb_error lrb_destroy_bitmap(lrb_bitmap_result* bitmap)
{
	if (!bitmap)
		return LRB_INVALID_PTR;

	if (bitmap->data)
		free(bitmap->data);
	memset(bitmap, 0, sizeof(lrb_bitmap_result));
	return LRB_OK;
}

lrb_error lrb_destroy_rle(lrb_rle_result* rle)
{
	if (!rle)
		return LRB_INVALID_PTR;

	if (rle->data)
		free(rle->data);
	memset(rle, 0, sizeof(lrb_rle_result));
	return LRB_OK;
}

int lrb_version(void)
{
	return LRB_VERSION;
}
