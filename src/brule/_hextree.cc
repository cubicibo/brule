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
#include <stdint.h>
#include <string.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

//None defined: use Atkinson dithering
//#define USE_JJN_DITHERING
//#define USE_SIERRA_DITHERING

#define MAX_DEPTH 8
#define MAX_WIDTH 16
#define DTYPE_ERROR float

#define ERR_THRESH_SKIP (2)
#define MAX_ERROR_COMPONENT (16)
#define N_ACTIVE_ROWS_DIFFUSION (3)
#define ERROR_ROW(y, rl) ((y) % N_ACTIVE_ROWS_DIFFUSION)*(rl)
#define ERROR_ROW_LOC(y, rl, x) (ERROR_ROW(y, rl) + (x << 2))
#define ADDERROR(d, e)  (d)[0] += (e)[0];\
                        (d)[1] += (e)[1];\
                        (d)[2] += (e)[2];\
                        (d)[3] += (e)[3]
#define ADDERRORF(d, e, f) (d)[0] += (e)[0]*(DTYPE_ERROR)f;\
                           (d)[1] += (e)[1]*(DTYPE_ERROR)f;\
                           (d)[2] += (e)[2]*(DTYPE_ERROR)f;\
                           (d)[3] += (e)[3]*(DTYPE_ERROR)f

#define CLIP(v) (v > 255 ? 255 : (v < 0 ? 0 : v))
#define CLIPD(d) CLIP(lrint(d))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define ABS(a) (a < 0 ? -1*a : a)
#define SIGN(a) (a < 0 ? -1 : 1)
#define XBIT2ID(val, off, shift) (((val[shift] >> (MAX_DEPTH - 1 - off)) & 0x01) << shift)

static void inline unpackRgba(const uint8_t *value, DTYPE_ERROR *prgba)
{
    prgba[3] = (DTYPE_ERROR)(value[3]);
    if (value[3] > 0) {
        prgba[2] = (DTYPE_ERROR)(value[2]);
        prgba[1] = (DTYPE_ERROR)(value[1]);
        prgba[0] = (DTYPE_ERROR)(value[0]);
    } else {
        prgba[2] = prgba[1] = prgba[0] = 0;
    }
}

static void inline clipPackPack(const DTYPE_ERROR *rgba, int32_t *irgba, uint8_t *v)
{
    irgba[3] = v[3] = (uint8_t)CLIPD(rgba[3]);
    if (irgba[3] > 0) {
        irgba[2] = v[2] = (uint8_t)CLIPD(rgba[2]);
        irgba[1] = v[1] = (uint8_t)CLIPD(rgba[1]);
        irgba[0] = v[0] = (uint8_t)CLIPD(rgba[0]);
    } else {
        irgba[2] = v[2] = 0;
        irgba[1] = v[1] = 0;
        irgba[0] = v[0] = 0;
    }
}

PyDoc_STRVAR(hextree_quantize_doc, "quantize(bitmap: tuple[NDArray[uint8], int]) -> tuple[NDArray[uint8], NDArray[uint8]]\
\
Quantize input RGBA and return (bitmap, palette).");

typedef struct rgba_leaf_s {
    uint32_t depth;
    uint32_t count;
    DTYPE_ERROR rgba[4];
    void *ref;
} rgba_leaf_t;

typedef struct hexnode_s {
    rgba_leaf_t *priv;
    struct hexnode_s *children[MAX_WIDTH];
} hexnode_t;

typedef struct ctx_s {
    hexnode_t *root;
    rgba_leaf_t **leafs;
    uint32_t nLeafs;
    uint32_t nLeafsMax;
    uint32_t nEndLeafs;
    int32_t paletteLUT[256][4];
    int16_t lutLen;
} ctx_t;

static uint8_t inline indexFrom(const uint8_t *v, const uint8_t depth)
{
    return (XBIT2ID(v, depth, 0) | XBIT2ID(v, depth, 1) | XBIT2ID(v, depth, 2) | XBIT2ID(v, depth, 3)) * (v[3] > 0);
}

static void flush(hexnode_t *current)
{    
    for (uint8_t c = 0; c < MAX_WIDTH; c++) {
        if (current->children[c])
            flush(current->children[c]);
    }
    if (current->priv)
        free(current->priv);
    free(current);
}

static void destroy(void *ctx)
{
    ctx_t *pctx = (ctx_t*)ctx;
    for (uint32_t k = 0; k < pctx->nLeafs; k++) {
        if (pctx->leafs[k]) {
            if (pctx->leafs[k]->ref)
                ((hexnode_t*)pctx->leafs[k]->ref)->priv = NULL;
            free(pctx->leafs[k]);
        }
    }
    free(pctx->leafs);
    pctx->nLeafsMax = pctx->nLeafs = 0;
    flush(pctx->root);
    free(ctx);
}

static void* init(void)
{
    ctx_t *ctx = (ctx_t*)calloc(1, sizeof(ctx_t));
    ctx->root = (hexnode_t*)calloc(1, sizeof(hexnode_t));
    return (void*)ctx;
}

static int insert(void *vctx, const uint8_t *value)
{
    ctx_t *ctx = (ctx_t*)vctx;

    hexnode_t *child;
    hexnode_t *node = ctx->root;

    for (uint8_t d = 0, c; d < MAX_DEPTH; d++) {
        c = indexFrom(value, d);
        if (!node->children[c])
        {
            child = (hexnode_t*)calloc(1, sizeof(hexnode_t));
            if (!child)
                return -1;

            if (ctx->nLeafs >= ctx->nLeafsMax) {
                rgba_leaf_t **newleafs = (rgba_leaf_t**)realloc(ctx->leafs, sizeof(rgba_leaf_t*)*(ctx->nLeafsMax+1024));
                if (!newleafs) {
                    free(child);
                    return -1;
                }
                ctx->nLeafsMax += 1024;
                ctx->leafs = newleafs;
            }

            child->priv = (rgba_leaf_t*)calloc(1, sizeof(rgba_leaf_t));
            if (!child->priv) {
                free(child);
                return -1;
            }
            child->priv->depth = d;
            child->priv->ref = (void*)child;

            unpackRgba(value, child->priv->rgba);

            ctx->leafs[ctx->nLeafs++] = (rgba_leaf_t*)child->priv;
            node->children[c] = child;
            ctx->nEndLeafs += (d == MAX_DEPTH-1);
        }
        node = node->children[c];
        ++node->priv->count;
    }
    return 0;
}

static int32_t inline fetchColour(hexnode_t *cur, const uint8_t *value)
{
    for (uint8_t d = 0, c; d < MAX_DEPTH; d++) {
        c = indexFrom(value, d);
        if (cur->children[c])
            cur = cur->children[c];
        else
            break;
    }
    return (int32_t)cur->priv->count;
}

static int32_t inline testFetchColourLeaf(hexnode_t *cur, uint8_t *value)
{
    for (uint8_t d = 0, c; d < MAX_DEPTH; d++) {
        c = indexFrom(value, d);
        if (cur->children[c])
            cur = cur->children[c];
        else
            break;
    }
    //is leaf?
    if (!memcmp(&cur->children[1], cur->children, sizeof(hexnode_t*)*(MAX_WIDTH-1)))
        return (int32_t)cur->priv->count;
    return -1;
}

#if defined(USE_JJN_DITHERING)
# define DEN_DITHERING (48)
static inline void diffuseJJN(DTYPE_ERROR *errors, DTYPE_ERROR *rgbaErr, uint32_t x, uint32_t y,
                              uint32_t lineId, uint32_t rowLen, uint32_t width, size_t len, int direction)
{
    DTYPE_ERROR *pErrors;
    if (x < width - 2 && x > 1) {
        pErrors = &errors[ERROR_ROW_LOC(lineId, rowLen, x)];
        ADDERRORF(pErrors + 4*direction, rgbaErr, 7);
        ADDERRORF(pErrors + 8*direction, rgbaErr, 5);
        if (y < len - width) {
            pErrors = &errors[ERROR_ROW_LOC(lineId+1, rowLen, x)];
            ADDERRORF(pErrors - 8, rgbaErr, 3);
            ADDERRORF(pErrors - 4, rgbaErr, 5);
            ADDERRORF(pErrors,     rgbaErr, 7);
            ADDERRORF(pErrors + 4, rgbaErr, 5);
            ADDERRORF(pErrors + 8, rgbaErr, 3);
        }
        if (y < len - 2*width) {
            pErrors = &errors[ERROR_ROW_LOC(lineId+2, rowLen, x)];
            ADDERRORF(pErrors - 8, rgbaErr, 1);
            ADDERRORF(pErrors - 4, rgbaErr, 3);
            ADDERRORF(pErrors,     rgbaErr, 5);
            ADDERRORF(pErrors + 4, rgbaErr, 3);
            ADDERRORF(pErrors + 8, rgbaErr, 1);
        }
    }
}

#elif defined(USE_SIERRA_DITHERING)
# define DEN_DITHERING (32)
static inline void diffuseSierra(DTYPE_ERROR *errors, DTYPE_ERROR *rgbaErr, uint32_t x, uint32_t y,
                                uint32_t lineId, uint32_t rowLen, uint32_t width, size_t len, int direction)
{
    DTYPE_ERROR *pErrors;
    if (x < width - 2 && x > 1) {
        pErrors = &errors[ERROR_ROW_LOC(lineId, rowLen, x)];
        ADDERRORF(pErrors + 4*direction, rgbaErr, 5);
        ADDERRORF(pErrors + 8*direction, rgbaErr, 3);
    }
    if (y < len - width && x < width - 2 && x > 1) {
        pErrors = &errors[ERROR_ROW_LOC(lineId+1, rowLen, x)];
        ADDERRORF(pErrors - 8, rgbaErr, 2);
        ADDERRORF(pErrors - 4, rgbaErr, 4);
        ADDERRORF(pErrors,     rgbaErr, 5);
        ADDERRORF(pErrors + 4, rgbaErr, 4);
        ADDERRORF(pErrors + 8, rgbaErr, 2);
    }
    if (y < len - 2*width && x > 0 && x < width - 1) {
        pErrors = &errors[ERROR_ROW_LOC(lineId+2, rowLen, x)];
        ADDERRORF(pErrors - 4, rgbaErr, 2);
        ADDERRORF(pErrors,     rgbaErr, 3);
        ADDERRORF(pErrors + 4, rgbaErr, 2);
    }
}

#else
//Implementation is in the generateDitheredBitmap function.
# define DEN_DITHERING (8)
#endif

#define DITHDIV(x) ((x) / (DTYPE_ERROR)DEN_DITHERING)

static inline uint32_t colorDist(const int32_t *v1, const int32_t *v2)
{
    int32_t diff = (v1[0] - v2[0]);
    uint32_t dist = diff*diff;

    diff = (v1[1] - v2[1]);
    dist += diff*diff;

    diff = (v1[2] - v2[2]);
    dist += diff*diff;
    return dist;
}

static inline int16_t findClosestInPalette(const ctx_t *ctx, const int32_t *irgba/*, int16_t penalizedEntry*/)
{
    uint32_t minDist = (uint32_t)(-1);
    int16_t bestFitId = -1;

    for (int16_t paletteEntryId = 0; paletteEntryId < ctx->lutLen; ++paletteEntryId) {
        uint32_t dist = colorDist(ctx->paletteLUT[paletteEntryId], irgba);

        int32_t diff = (ctx->paletteLUT[paletteEntryId][3] - irgba[3]);
        //dist = (uint32_t)((diff*diff)/(DTYPE_ERROR)1.5 + ((DTYPE_ERROR)dist*((irgba[3]*ctx->paletteLUT[paletteEntryId][3]/(DTYPE_ERROR)1953810))));
        dist = dist + 3*(uint32_t)(diff*diff);
        /*
        if (paletteEntryId == penalizedEntry)
            dist += (dist >> 4); //Penalize by 10%
        */
        if (dist < minDist) {
            minDist = dist;
            bestFitId = paletteEntryId;
        }
    }
    return bestFitId;
}

static int generateDitheredBitmap( const void *vctx, uint8_t **bitmap, const uint8_t *rgba,
                                   const size_t len, const int16_t plen, const uint32_t width )
{
    const ctx_t *ctx = (const ctx_t*)vctx;
    const uint32_t rowLen = width << 2;
    const uint32_t errArrLen = (width << 2) * N_ACTIVE_ROWS_DIFFUSION;

    *bitmap = (uint8_t*)malloc(len*sizeof(uint8_t));
    if (NULL == *bitmap)
        return -1;

    DTYPE_ERROR *pErrors, *errors = (DTYPE_ERROR*)calloc(errArrLen, sizeof(DTYPE_ERROR));
    if (NULL == errors)
        return -1;

    DTYPE_ERROR sErr, rgbaOrg[4], rgbaErr[4];
    int32_t paletteEntryId, irgba[4];
    uint8_t crgba[4];
    const uint8_t *rgbaPixel;

    int direction = -1;
    
    for (uint32_t y = 0, lineId = 0; y < len; y += width, ++lineId) {
        uint32_t x;
        if (direction < 0) {
            direction = 1;
            x = 0;
        } else {
            direction = -1;
            x = width - 1;
        }
        for (uint32_t cnt = 0; cnt < width; ++cnt, x += direction) {
            pErrors = &errors[ERROR_ROW_LOC(lineId, rowLen, x)];
            rgbaPixel = &rgba[(y + x) << 2];

            sErr = 0;
            for (uint8_t k = 0; k < 4; k++) {
                pErrors[k] = SIGN(pErrors[k])*MIN((DTYPE_ERROR)MAX_ERROR_COMPONENT, ABS(pErrors[k]));
                sErr += pErrors[k]*pErrors[k];
            }

            unpackRgba(rgbaPixel, rgbaOrg);
            if (sErr > (DTYPE_ERROR)ERR_THRESH_SKIP) {
                //Add the residual error to the image pixel
                ADDERROR(rgbaOrg, pErrors);
                clipPackPack(rgbaOrg, irgba, crgba);
                //Test if we have a match in the tree, returns -1 if none
                paletteEntryId = testFetchColourLeaf(ctx->root, crgba);
            } else {
                //no error => colour is a leaf
                paletteEntryId = fetchColour(ctx->root, rgbaPixel);
            }
            //reset error accumulator of pixel
            memset(pErrors, 0, sizeof(rgbaErr));

            if (paletteEntryId < 0) {
                paletteEntryId = findClosestInPalette(ctx, irgba/*, fetchColour(ctx->root, rgbaPixel)*/);
            } else if (paletteEntryId >= plen) {
                free(errors);
                return -1;
            }

            for (uint8_t k = 0; k < 4; k++)
                rgbaErr[k] = DITHDIV(rgbaOrg[k] - (DTYPE_ERROR)ctx->paletteLUT[paletteEntryId][k]);

            //simplified edges conditions
#if defined(USE_JJN_DITHERING)
            diffuseJJN(errors, rgbaErr, x, y, lineId, rowLen, width, len, direction);
#elif defined(USE_SIERRA_DITHERING)
            diffuseSierra(errors, rgbaErr, x, y, lineId, rowLen, width, len, direction);
#else //Atkinson

            if ((direction > 0 && x < width - 2) || (direction < 0 && x > 1)) {
                ADDERROR(pErrors + 4*direction, rgbaErr);
                ADDERROR(pErrors + 8*direction, rgbaErr);
            }
            if ((y < len - width) && (x < width - 1) && (x > 0)) {
                pErrors = &errors[ERROR_ROW_LOC(lineId+1, rowLen, x)];
                ADDERROR(pErrors - 4, rgbaErr);
                ADDERROR(pErrors,     rgbaErr);
                ADDERROR(pErrors + 4, rgbaErr);
            }
            if (y < len - 2*width) {
                pErrors = &errors[ERROR_ROW_LOC(lineId+2, rowLen, x)];
                ADDERROR(pErrors, rgbaErr);
            }
#endif
            (*bitmap)[y + x] = paletteEntryId;
        }
    }
    free(errors);
    return 0;
}

static int generateBitmap(const void* vctx, uint8_t** bitmap, const uint8_t* rgba, const size_t len, const uint32_t plen)
{
    const ctx_t *ctx = (const ctx_t*)vctx;

    *bitmap = (uint8_t*)calloc(len, sizeof(uint8_t));
    if (NULL == *bitmap)
        return -1;

    uint32_t value;
    for (size_t k = 0; k < len; k++) {
        value = (uint32_t)fetchColour(ctx->root, &rgba[k<<2]);

        //All pixels are unmodified, all tree look-ups shall always lend on a leaf (value < plen)
        if (value >= plen)
            return -1;
        (*bitmap)[k] = (uint8_t)value;
    }
    return 0;
}

static int cmpleafs(const void* e1, const void* e2)
{
    const rgba_leaf_t *l1 = *(rgba_leaf_t**)(void**)e1;
    const rgba_leaf_t *l2 = *(rgba_leaf_t**)(void**)e2;

    int r = (int)l2->depth - (int)l1->depth;
    if (r != 0)
        return r;
    return (int)l1->count - (int)l2->count;
}

static int32_t reduceTo(void *ctx, unsigned int maxLeafs, uint8_t **palette)
{
    ctx_t *pctx = (ctx_t*)ctx;
    hexnode_t *ref;
    rgba_leaf_t* leaf;

    uint32_t leafCnt = pctx->nEndLeafs;

    if (maxLeafs < MAX_WIDTH || !pctx->leafs)
        return -1;

    qsort(pctx->leafs, pctx->nLeafs, sizeof(rgba_leaf_t*), cmpleafs);

    for (uint32_t k = 0; k < pctx->nLeafs && maxLeafs < leafCnt; k++) {
        leaf = ((rgba_leaf_t*)pctx->leafs[k]);
        if (leaf->depth == MAX_DEPTH-1)
            continue;
        ref = (hexnode_t*)leaf->ref;
        if (!ref || !ref->priv)
            continue;

        memset(leaf->rgba, 0, sizeof(DTYPE_ERROR)*4);
        for (uint8_t cc = 0; cc < MAX_WIDTH; cc++) {
            if (ref->children[cc]) {
                DTYPE_ERROR cf = (DTYPE_ERROR)ref->children[cc]->priv->count / (DTYPE_ERROR)leaf->count;
                leaf->rgba[0] += ref->children[cc]->priv->rgba[0] * cf;
                leaf->rgba[1] += ref->children[cc]->priv->rgba[1] * cf;
                leaf->rgba[2] += ref->children[cc]->priv->rgba[2] * cf;
                leaf->rgba[3] += ref->children[cc]->priv->rgba[3] * cf;
                ref->children[cc]->priv->ref = NULL;
                free(ref->children[cc]);
                ref->children[cc] = NULL;
                --leafCnt;
            }
        }
        ++leafCnt; //parent is now a leaf
    }

    *palette = (uint8_t*)calloc(leafCnt << 2, sizeof(uint8_t));
    if (!palette)
        return -1;


    uint8_t *pal = *palette;
    for (uint32_t k = 0, plen = 0; plen != leafCnt; k++) {
        leaf = (rgba_leaf_t*)pctx->leafs[k];
        ref = (hexnode_t*)leaf->ref;

        if (leaf && ref) {
            clipPackPack(leaf->rgba, pctx->paletteLUT[plen], &pal[plen << 2]);
            leaf->count = plen;
            ++plen;
        }
    }
    pctx->lutLen = leafCnt;
    return leafCnt;
}

PyObject *hextree_quantize(PyObject *self, PyObject *arg)
{
    if (Py_None == arg || !PyTuple_Check(arg))
        return NULL;

    size_t nArgs = PyTuple_Size(arg);
    if (nArgs < 2 || nArgs > 3)
        return NULL;

    uint32_t width = 0;
    if (nArgs >= 3) {
        PyObject *pwi = PyTuple_GetItem(arg, 2);
        if (PyLong_Check(pwi))
            width = (uint32_t)PyLong_AsLong(pwi);
    }

    PyObject *pnc = PyTuple_GetItem(arg, 0);
    PyObject *pai = PyTuple_GetItem(arg, 1);
    if (!PyLong_Check(pnc) || !PyArray_Check(pai))
        return NULL;

    int nc = (int)PyLong_AsLong(pnc);
    if (nc < 16 || nc > 256)
        return NULL;

    PyArrayObject *arr = (PyArrayObject*)pai;
    size_t len = PyArray_DIM(arr, 0);
    size_t rawLen = len;

    //Not flat or not mod4
    if (PyArray_NDIM(arr) != 1 || (len & 0b11))
        return NULL;
    len >>= 2;

    if (0 == (PyArray_FLAGS(arr) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
        return NULL;

    if (0 == (NPY_ARRAY_ALIGNED & PyArray_FLAGS(arr)))
        return NULL;

    const uint8_t *rgba = (const uint8_t*)PyArray_BYTES(arr);

    if (width > len || (width > 0 && len % width) || width > 8192)
        return NULL;

    int32_t err = 0, plen = 0;

    //catalog
    void *ctx = init();
    for (size_t k = 0; k < rawLen && 0 == err; k += 4)
        err = insert(ctx, &rgba[k]);

    //returned objects
    uint8_t *palette = NULL;
    uint8_t *bitmap = NULL;
    PyObject *rtup = NULL;
    PyArrayObject *arr_bitmap = NULL, *arr_pal = NULL;
    PyArray_Descr *desc_pal = NULL, *desc_bitmap = NULL;

    if (0 == err) {
        //generate palette
        plen = reduceTo(ctx, nc, &palette);
        if (plen > 0 && plen <= 256) {
            if (width)
                err = generateDitheredBitmap(ctx, &bitmap, rgba, len, (int16_t)plen, width);
            else
                err = generateBitmap(ctx, &bitmap, rgba, len, plen);
        } else {
            err = 1;
        }
    }
    //generate bitmap
    if (plen > 0 && 0 == err) {
        npy_intp bitmap_dim = len;
        desc_bitmap = PyArray_DescrFromType(NPY_UBYTE);
        arr_bitmap = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, desc_bitmap, 1, &bitmap_dim, NULL, bitmap, 0, NULL);
        PyArray_ENABLEFLAGS(arr_bitmap, NPY_ARRAY_OWNDATA);

        if (arr_bitmap) {
            npy_intp palette_dim[2] = {plen, 4};
            desc_pal = PyArray_DescrFromType(NPY_UBYTE);
            arr_pal = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, desc_pal, 2, palette_dim, NULL, palette, 0, NULL);
            PyArray_ENABLEFLAGS(arr_pal, NPY_ARRAY_OWNDATA);

            if (arr_pal)
                rtup = Py_BuildValue("NN", arr_bitmap, arr_pal);
        }
    }
    destroy(ctx);

    if (NULL == rtup || plen <= 0 || err != 0) {
        //palette and bitmap are owned by non-existing objects
        if (palette && NULL == arr_pal) {
            Py_XDECREF(desc_pal);
            free(palette);
        }
        if (bitmap && NULL == arr_bitmap) {
            Py_XDECREF(desc_bitmap);
            free(bitmap);
        }

        Py_XDECREF(arr_pal);
        Py_XDECREF(arr_bitmap);
        Py_RETURN_NONE;
    }
    return rtup;
}

/*
 * List of functions to add to brule in exec_brule().
 */
static PyMethodDef hextree_functions[] = {
    { "quantize", (PyCFunction)hextree_quantize, METH_O, hextree_quantize_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for hextree.
 */
PyDoc_STRVAR(hextree_doc, "HexTree 8-bit quanizer.");

static PyModuleDef hextree_def = {
    PyModuleDef_HEAD_INIT,
    "hextree",
    hextree_doc,
    0,              /* m_size */
    hextree_functions,/* m_methods */
    NULL,           /* m_slots */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit__hextree() {
    import_array();
    return PyModuleDef_Init(&hextree_def);
}