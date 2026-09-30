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
#include <stdint.h>

#include "librle.h"

#define UMAX(a, b) (a < b ? b : a)
#define UMIN(a, b) (a < b ? a : b)

//#define WIN32_ALLOW_THREADS
#define MAX_NUM_THREADS 3
#define ALLOC_SIZE_STEP 24576

typedef struct SLICE_IO_s
{
    const unsigned char* bitmap_slice;
    unsigned int width;
    unsigned int height;
    lrb_rle_result rle_res;
} SLICE_IO_t;

static void* lrb_encode_slice(void *c);

#if defined(_WIN32) && defined(WIN32_ALLOW_THREADS)
#include <windows.h>
#include <process.h>

# define HAS_THREADING
static unsigned __stdcall lrb_encode_slice_win(void *arg)
{
    return lrb_encode_slice(arg) != NULL;
}

typedef HANDLE THREAD_t;
# define thread_init(t, ar) (t = (THREAD_t)(_beginthreadex(NULL, 0, lrb_encode_slice_win, (void*)ar, 0, NULL)))
# define thread_join(t, r) r = WaitForSingleObject(t, INFINITE)
static inline lrb_error thread_validate(THREAD_t *t, uint32_t rv)
{
    int success = 0;
    DWORD exit_code = 0;
    if (rv != WAIT_OBJECT_0)
        CloseHandle(*t);
    else {
        success = GetExitCodeThread(*t, &exit_code);
        CloseHandle(*t);
    }
    return success && exit_code == 0 ? LRB_OK : LRB_THREAD_FAIL;
}

/* inaccurate posix detection */
#elif (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#include <pthread.h>

# define HAS_THREADING
typedef pthread_t THREAD_t;
# define thread_init(t, ar) (pthread_create(&t, NULL, lrb_encode_slice, (void*)ar) == 0)
# define thread_join(t, r) pthread_join(t, NULL)
static inline lrb_error thread_validate(THREAD_t *t, uint32_t rv) { return LRB_OK; }

#endif /// ifelse threading

static void* lrb_encode_slice(void *c)
{
    if (!c)
        return NULL;

    SLICE_IO_t *in_out = (SLICE_IO_t *)c;
    const uint32_t width = in_out->width;
    const uint32_t height = in_out->height;
    const unsigned char* cbit = (const unsigned char*)in_out->bitmap_slice;

    const uint32_t area = width * height;

    lrb_rle_result rle_res;
    uint32_t allocated_size = ALLOC_SIZE_STEP;
    rle_res.data = (unsigned char*)malloc(ALLOC_SIZE_STEP * sizeof(unsigned char));
    if (!rle_res.data)
        return NULL;
    rle_res.length = 0;

    for (uint32_t line_index = 0; line_index < area; line_index += width) {
        uint32_t j = 0;
        do {
            uint32_t start_point = j;
            uint8_t color = cbit[line_index + j];
            while ((++j < width) && (color == cbit[line_index + j]));

            uint32_t distance = j - start_point;
            if (!distance || start_point + distance > width) {
                free(rle_res.data);
                memset(&in_out->rle_res, 0, sizeof(lrb_rle_result));
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
                if (allocated_size > 8u << 20)
                {
                    free(rle_res.data);
                    memset(&in_out->rle_res, 0, sizeof(lrb_rle_result));
                    return NULL;
                }
                unsigned char* tmpptr = (unsigned char*)realloc(rle_res.data, allocated_size * sizeof(unsigned char));
                if (!tmpptr) {
                    free(rle_res.data);
                    memset(&in_out->rle_res, 0, sizeof(lrb_rle_result));
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

lrb_error lrb_encode_bitmap(const void* bitmap, const unsigned int width, const unsigned int height, lrb_rle_result* rle_res)
{
    if (!rle_res)
        return LRB_INVALID_PTR;

    if (!bitmap)
        return LRB_INVALID_PTR;

    if (width < 8 || height < 8 || width > 4096 || height > 4096)
        return LRB_INVALID_DIMENSION;

    if (rle_res->data)
        return LRB_INVALID_PTR;

#ifdef HAS_THREADING
    // don't create threads for small widths
    const uint32_t num_slices = width >= 250 ? UMIN(height/200, MAX_NUM_THREADS) : 1u;

    if (num_slices <= 1)
#endif
    {
        SLICE_IO_t in_out;
        in_out.bitmap_slice = (unsigned char*)bitmap;
        in_out.width = width;
        in_out.height = height;

        lrb_encode_slice(&in_out);
        memcpy(rle_res, &in_out.rle_res, sizeof(lrb_rle_result));
        return in_out.rle_res.length ? LRB_OK : LRB_INVALID_VALUE;
    }

#ifdef HAS_THREADING
    THREAD_t threads[MAX_NUM_THREADS] = {0};
    SLICE_IO_t *slice, *slices[MAX_NUM_THREADS] = {0};

    const uint32_t stride_height = height / num_slices;
    uint32_t orphaned_lines = height - stride_height*num_slices;
    uint32_t offset = 0;
    for (uint32_t s = 0; s < num_slices; ++s)
    {
        slices[s] = slice = (SLICE_IO_t*)calloc(1, sizeof(SLICE_IO_t));
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
            slice->bitmap_slice = &((unsigned char*)bitmap)[offset];
            offset += slice->height*width;
            if (!thread_init(threads[s], slice))
            {
                free(slice);
                slices[s] = NULL;
                break;
            }
        }
    }

    lrb_error gbl_err = LRB_OK;

    uint32_t total_length = 0;
    rle_res->length = 0;
    for (uint32_t s = 0; s < num_slices; ++s)
    {
        SLICE_IO_t *slice = slices[s];
        if (threads[s])
        {
            uint32_t r;
            thread_join(threads[s], r);

            if (LRB_OK == thread_validate(&threads[s], r) && gbl_err == LRB_OK && slice)
            {
                if (rle_res->length + slice->rle_res.length > total_length) {
                    total_length += slice->rle_res.length * num_slices; /* possibly enough to do it once */
                    unsigned char *ptmp = (unsigned char*)realloc(rle_res->data, total_length);
                    if (!ptmp) {
                        gbl_err = LRB_ENOMEM;
                        goto slice_clear;
                    }
                    rle_res->data = ptmp;
                }
                memcpy(&rle_res->data[rle_res->length], slice->rle_res.data, slice->rle_res.length);
                rle_res->length += slice->rle_res.length;
            }
            else if (gbl_err == LRB_OK)
                gbl_err = slice ? LRB_THREAD_FAIL : LRB_ENOMEM;
        }
slice_clear:
        if (slice) {
            free(slice->rle_res.data);
            free(slice);
        }
    }
    return gbl_err;
#endif // HAS_THREADING
}

lrb_error lrb_decode_rle(const void* data, const unsigned int length, lrb_bitmap_result* bitmap_res)
{
    if (!data || !bitmap_res)
        return LRB_INVALID_PTR;
    if (length < 2)
        return LRB_INVALID_VALUE;

    unsigned long allocated_size = ALLOC_SIZE_STEP;
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
            allocated_size += ALLOC_SIZE_STEP;
            if (allocated_size > 8u << 20)
            {
                free(bitmap_res->data);
                memset(bitmap_res, 0, sizeof(lrb_bitmap_result));
                return LRB_ENOMEM;
            }
            tmpptr = (unsigned char*)realloc(bitmap_res->data, allocated_size * sizeof(unsigned char));
            if (!tmpptr) {
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
