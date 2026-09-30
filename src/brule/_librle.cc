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
#include <pthread.h>

#include "librle.h"

#define UMAX(a, b) (a < b ? b : a)
#define UMIN(a, b) (a < b ? a : b)

#define MAX_NUM_THREADS 3
#define ALLOC_SIZE_STEP 24576

typedef struct SLICE_INPUT_s
{
    const unsigned char* bitmap_slice;
    unsigned int width;
    unsigned int height;
    lrb_rle_result_t rle_res;
} SLICE_INPUT_t;

void* lrb_encode_slice(void *c)
{
    if (!c)
        return NULL;

    SLICE_INPUT_t *in_out = (SLICE_INPUT_t *)c;
    const unsigned int width = in_out->width;
    const unsigned int height = in_out->height;
    const unsigned char* cbit = (const unsigned char*)in_out->bitmap_slice;

    const unsigned int area = width * height;

    lrb_rle_result_t rle_res;
    unsigned long allocated_size = ALLOC_SIZE_STEP;
    rle_res.data = (unsigned char*)malloc(ALLOC_SIZE_STEP * sizeof(unsigned char));
    if (!rle_res.data)
        return NULL;
    rle_res.length = 0;

    for (unsigned long line_index = 0; line_index < area; line_index += width) {
        unsigned long j = 0;
        do {
            unsigned long start_point = j;
            unsigned long color = cbit[line_index + j];
            while ((++j < width) && (color == cbit[line_index + j]));

            unsigned long distance = j - start_point;
            if (!distance || start_point + distance > width) {
                free(rle_res.data);
                memset(&in_out->rle_res, 0, sizeof(lrb_rle_result_t));
                return NULL;
            }
            if (0 == color) {
                rle_res.data[rle_res.length++] = 0;
                if (distance > 63) {
                    rle_res.data[rle_res.length++] = (0x40 | ((distance >> 8) & 0x3F));
                    rle_res.data[rle_res.length++] = (distance & 0xFF);
                } else {
                    rle_res.data[rle_res.length++] = (distance & 0x3F);
                }
            } else {
                if (distance > 63) {
                    rle_res.data[rle_res.length++] = 0;
                    rle_res.data[rle_res.length++] = (0xC0 | ((distance >> 8) & 0x3F));
                    rle_res.data[rle_res.length++] = (distance & 0xFF);
                    rle_res.data[rle_res.length++] = color;
                } else if (distance > 2) {
                    rle_res.data[rle_res.length++] = 0;
                    rle_res.data[rle_res.length++] = (0x80 | (distance & 0x3F));
                    rle_res.data[rle_res.length++] = color;
                } else {
                    rle_res.data[rle_res.length++] = color;
                    if (distance == 2)
                        rle_res.data[rle_res.length++] = color;
                }
            }
            if (allocated_size < rle_res.length + 8) {
                allocated_size += ALLOC_SIZE_STEP;
                unsigned char* tmpptr = (unsigned char*)realloc(rle_res.data, allocated_size * sizeof(unsigned char));
                if (!tmpptr || allocated_size > 8 << 20) {
                    free(rle_res.data);
                    memset(&in_out->rle_res, 0, sizeof(lrb_rle_result_t));
                    return NULL;
                }
                rle_res.data = tmpptr;
            }
        } while (j < width);
        rle_res.data[rle_res.length++] = 0;
        rle_res.data[rle_res.length++] = 0;
    }
    memcpy(&in_out->rle_res, &rle_res, sizeof(rle_res));
    return c;
}


lrb_error lrb_encode_bitmap(const unsigned char* bitmap, const unsigned int width, const unsigned int height, lrb_rle_result_t* rle_res)
{
    if (!rle_res)
        return LRB_INVALID_PTR;

    if (!bitmap)
        return LRB_INVALID_PTR;

    if (width < 8 || height < 8 || width > 4096 || height > 4096)
        return LRB_INVALID_DIMENSION;

    if (rle_res->data)
        return LRB_INVALID_PTR;

    const uint32_t num_slices = UMIN(height/100, MAX_NUM_THREADS);

    if (num_slices == 1)
    {
        SLICE_INPUT_t in_out = {.bitmap_slice = bitmap, .width = width, .height = height};
        lrb_error e = lrb_encode_slice(&in_out) ? LRB_OK : LRB_THREAD_FAIL;
        memcpy(rle_res, &in_out.rle_res, sizeof(lrb_rle_result_t));
        return e;
    }

    pthread_t threads[MAX_NUM_THREADS] = {0};
    SLICE_INPUT_t *slice, *slices[MAX_NUM_THREADS];

    const uint32_t stride_height = height / num_slices;
    uint32_t orphaned_lines = height - stride_height*num_slices;
    uint32_t offset = 0;
    for (uint32_t s = 0; s < num_slices; ++s)
    {
        slices[s] = slice = (SLICE_INPUT_t*)malloc(sizeof(SLICE_INPUT_t));
        if (slice) {
            slice->width = width;
            slice->height = stride_height;
            if (orphaned_lines > 0) {
                --orphaned_lines;
                slice->height = stride_height + 1;
            }
            else {
                slice->height = stride_height;
            }
            slice->bitmap_slice = &bitmap[offset];
            offset += slice->height*width;
            pthread_create(&threads[s], NULL, *lrb_encode_slice, (void *) slice);
        }
    }

    lrb_error gbl_err = LRB_OK;

    uint32_t total_length = 0, total_written = 0;
    rle_res->length = 0;
    for (uint32_t s = 0; s < num_slices; ++s)
    {
        SLICE_INPUT_t *slice = slices[s];
        if (threads[s])
        {
            void *p;
            pthread_join(threads[s], &p);

            if (p && gbl_err == LRB_OK)
            {
                if (rle_res->length + slice->rle_res.length > total_length) {
                    total_length += slice->rle_res.length * num_slices; /* possibly enough to do it once */
                    rle_res->data = (unsigned char*)realloc(rle_res->data, total_length);
                }
                memcpy(&rle_res->data[total_written], slice->rle_res.data, slice->rle_res.length);
                rle_res->length += slice->rle_res.length;
            }
            else if (gbl_err == LRB_OK)
                gbl_err = LRB_THREAD_FAIL;
        }
        if (slice) {
            free(slice->rle_res.data);
            free(slice);
        }
    }
    return gbl_err;
}


lrb_error lrb_decode_rle(const unsigned char* data, const unsigned int length, lrb_bitmap_result_t* bitmap_res)
{
    if (!data || !bitmap_res)
        return LRB_INVALID_PTR;
    if (length < 2)
        return LRB_INVALID_VALUE;

    unsigned long allocated_size = ALLOC_SIZE_STEP;
    if (bitmap_res->data)
        return LRB_INVALID_PTR;

    bitmap_res->data = (unsigned char*)malloc(allocated_size * sizeof(unsigned char));
    if (!bitmap_res->data)
        return LRB_ENOMEM;

    const unsigned char* rle = data;
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
                    memset(bitmap_res, 0, sizeof(lrb_bitmap_result_t));
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
            allocated_size += ALLOC_SIZE_STEP;
            tmpptr = (unsigned char*)realloc(bitmap_res->data, allocated_size * sizeof(unsigned char));
            if (!tmpptr || allocated_size > 8 << 20) {
                free(bitmap_res->data);
                memset(bitmap_res, 0, sizeof(lrb_bitmap_result_t));
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
        memset(bitmap_res, 0, sizeof(lrb_bitmap_result_t));
        return LRB_INVALID_DATA;
    }
    bitmap_res->height = line_index / line_width;
    bitmap_res->width = line_width;
    return LRB_OK;
}

lrb_error lrb_destroy_bitmap(lrb_bitmap_result_t* bitmap)
{
    if (!bitmap)
        return LRB_INVALID_PTR;

    if (bitmap->data)
        free(bitmap->data);
    memset(bitmap, 0, sizeof(lrb_bitmap_result_t));
    return LRB_OK;
}

lrb_error lrb_destroy_rle(lrb_rle_result_t* rle)
{
    if (!rle)
        return LRB_INVALID_PTR;

    if (rle->data)
        free(rle->data);
    memset(rle, 0, sizeof(lrb_rle_result_t));
    return LRB_OK;
}

int lrb_version(void)
{
    return LRB_VERSION;
}
