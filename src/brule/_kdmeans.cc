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

#define MAX_DEPTH 8
#define MAX_WIDTH 16
#define N_COMPONENTS 4
#define MAX_ENTRIES 256

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define CLIP(v) (v > 255 ? 255 : MAX(v, 0))
#define CLIPD(d) CLIP(lrint(d))
#define ABS(a) (a < 0 ? -1*a : a)
#define SIGN(a) (a < 0 ? -1 : 1)
#define XBIT2ID(val, off, shift) (((val[shift] >> (MAX_DEPTH - 1 - off)) & 0x01) << shift)
#define DTYPE_ERROR float

static void inline clipPackPack(const DTYPE_ERROR *rgba, int32_t *irgba, uint8_t *v)
{
    irgba[0] = v[0] = (uint8_t)CLIPD(rgba[0]);
    irgba[1] = v[1] = (uint8_t)CLIPD(rgba[1]);
    irgba[2] = v[2] = (uint8_t)CLIPD(rgba[2]);
    irgba[3] = v[3] = (uint8_t)CLIPD(rgba[3]);
}

static void inline unpackRgbaFp(const uint8_t *v, DTYPE_ERROR *prgba)
{
    prgba[0] = (DTYPE_ERROR)v[0];
    prgba[1] = (DTYPE_ERROR)v[1];
    prgba[2] = (DTYPE_ERROR)v[2];
    prgba[3] = (DTYPE_ERROR)v[3];
}

static void inline unpackRgba(const uint8_t *value, int32_t *prgba)
{
    prgba[0] = (int32_t)value[0];
    prgba[1] = (int32_t)value[1];
    prgba[2] = (int32_t)value[2];
    prgba[3] = (int32_t)value[3];
}

static void inline packRgba(uint8_t *v, const int32_t *prgba)
{
    v[0] = prgba[0];
    v[1] = prgba[1];
    v[2] = prgba[2];
    v[3] = prgba[3];
}


PyDoc_STRVAR(kdmeans_quantize_doc, "quantize(data: tuple[ndarray[uint8], int]) -> tuple[ndarray[uint8], ndarray[uint8]]\
\
Quantize input RGBA and return (bitmap, palette).");

typedef struct rgba_leaf_s {
    int32_t cc[N_COMPONENTS];
    uint32_t count;
    int16_t aci;
} leaf_data_t;

typedef struct hexnode_s {
    leaf_data_t *priv;
    struct hexnode_s *children[MAX_WIDTH];
} hexnode_t;

typedef struct ctx_s {
    hexnode_t *root;
    leaf_data_t **leafs;
    uint32_t nLeafs;
    uint32_t nLeafsMax;
    int32_t centroids[MAX_ENTRIES][N_COMPONENTS];
    uint16_t nc;
} ctx_t;

#if (N_COMPONENTS == 4)
static uint8_t inline indexFrom(const uint8_t *v, const uint8_t depth)
{
    return XBIT2ID(v, depth, 0) | XBIT2ID(v, depth, 1) | XBIT2ID(v, depth, 2) | XBIT2ID(v, depth, 3);
}
#else
# error "No indexer for CC != 4."
#endif

static void flush(hexnode_t *current)
{    
    for (uint8_t c = 0; c < MAX_WIDTH; c++) {
        if (current->children[c])
            flush(current->children[c]);
    }
    free(current);
}

static void destroy(ctx_t *ctx)
{
    for (uint32_t leafId = 0; leafId < ctx->nLeafs; leafId++)
        free(ctx->leafs[leafId]);
    free(ctx->leafs);
    ctx->nLeafsMax = ctx->nLeafs = 0;
    flush(ctx->root);
    free(ctx);
}

static ctx_t* init(void)
{
    ctx_t *ctx = (ctx_t*)calloc(1, sizeof(ctx_t));
    ctx->root = (hexnode_t*)calloc(1, sizeof(hexnode_t));
    return ctx;
}

static int insert(ctx_t *ctx, const uint8_t *value)
{
    hexnode_t *child;
    hexnode_t *node = ctx->root;

    for (uint8_t d = 0, c; d < MAX_DEPTH; d++) {
        c = indexFrom(value, d);
        if (!node->children[c])
        {
            child = (hexnode_t*)calloc(1, sizeof(hexnode_t));
            if (!child)
                return -1;

            if (d == MAX_DEPTH-1) {
                if (ctx->nLeafs >= ctx->nLeafsMax) {
                    leaf_data_t **newleafs = (leaf_data_t**)realloc(ctx->leafs, sizeof(leaf_data_t*)*(ctx->nLeafsMax+1024));
                    if (!newleafs) {
                        free(child);
                        return -1;
                    }
                    ctx->nLeafsMax += 1024;
                    ctx->leafs = newleafs;
                }
                child->priv = (leaf_data_t*)calloc(1, sizeof(leaf_data_t));
                if (!child->priv) {
                    free(child);
                    return -1;
                }
                ctx->leafs[ctx->nLeafs++] = child->priv;
                unpackRgba(value, child->priv->cc);
            }
            node->children[c] = child;
        }
        node = node->children[c];
    }
    ++node->priv->count;
    return 0;
}

static void init_centroids(ctx_t *ctx, const size_t len)
{
    const int32_t strideSample = (int32_t)lrint(ctx->nLeafs/(float)ctx->nc);
    uint32_t nextSample = 0;
    int cid = 0;
    uint32_t maxLen = len/10;

    const int32_t avgCount = (int32_t)lrint((double)len/(double)ctx->nLeafs);
    for (uint32_t k = 0; k < ctx->nLeafs; ++k) {
        //Avoid large bias on single colour
        if (ctx->leafs[k]->count > maxLen)
            ctx->leafs[k]->count = maxLen;
        if (k == nextSample) {
            memcpy(ctx->centroids[cid], ctx->leafs[k]->cc, sizeof(int32_t)*N_COMPONENTS);
            nextSample += ((++cid >= ctx->nc) ? 0 : strideSample);
        }

        //Flatten the histogram to improve high frequencies (and hence, dithering)
        ctx->leafs[k]->count += (avgCount - (int32_t)ctx->leafs[k]->count)/4;
    }
}

static uint32_t inline computeCentroidComponentAndDiff(int32_t *centroidComponent, const uint32_t cValue, const uint32_t count)
{
    int32_t cCoord = (int32_t)CLIPD(cValue / (double)count);
    uint32_t adiff = (*centroidComponent - cCoord);
    if (0 != adiff) {
        *centroidComponent = cCoord;
        return adiff*adiff;
    }
    return 0;
}

static inline uint32_t colorDist(const int32_t *v1, const int32_t *v2)
{
    int32_t diff = (v1[0] - v2[0]);
    uint32_t dist = diff*diff;

    diff = (v1[1] - v2[1]);
    dist += diff*diff;

    diff = (v1[2] - v2[2]);
    dist += diff*diff;

    diff = (v1[3] - v2[3]);
    //1953810 = 255*255*3
    return (uint32_t)((diff*diff)/(DTYPE_ERROR)1.5 + ((DTYPE_ERROR)dist*((v1[3]*v2[3]/(DTYPE_ERROR)1953810))));
}

static uint32_t kmeans_assign_update(ctx_t *ctx)
{
    uint32_t cdist, bdist, utemp;
    int32_t stemp;
    int16_t bcid;
    uint32_t newCentroids[MAX_ENTRIES][N_COMPONENTS] = {{0}};
    uint32_t countPerCentroid[MAX_ENTRIES] = {0};

    for (uint32_t k = 0; k < ctx->nLeafs; ++k) {
        bdist = (uint32_t)(-1);
        for (int16_t cid = 0; cid < ctx->nc; ++cid) {
            cdist = colorDist(ctx->leafs[k]->cc, ctx->centroids[cid]);
            if (bdist > cdist) {
                bcid = cid;
                bdist = cdist;
            }
        }
        ctx->leafs[k]->aci = bcid;
        utemp = ctx->leafs[k]->count;

        newCentroids[bcid][0] += ctx->leafs[k]->cc[0]*utemp;
        newCentroids[bcid][1] += ctx->leafs[k]->cc[1]*utemp;
        newCentroids[bcid][2] += ctx->leafs[k]->cc[2]*utemp;
        newCentroids[bcid][3] += ctx->leafs[k]->cc[3]*utemp;
        countPerCentroid[bcid] += utemp;
    }

    uint64_t tdiff = 0; //absolute distance shift
    for (uint16_t k = 0; k < ctx->nc; ++k) {
        tdiff += computeCentroidComponentAndDiff(&ctx->centroids[k][0], newCentroids[k][0], countPerCentroid[k]);
        tdiff += computeCentroidComponentAndDiff(&ctx->centroids[k][1], newCentroids[k][1], countPerCentroid[k]);
        tdiff += computeCentroidComponentAndDiff(&ctx->centroids[k][2], newCentroids[k][2], countPerCentroid[k]);
        tdiff += computeCentroidComponentAndDiff(&ctx->centroids[k][3], newCentroids[k][3], countPerCentroid[k]);
    }
    return (uint32_t)(tdiff/ctx->nc);
}

static uint8_t inline fetchPaletteId(hexnode_t *cur, const uint8_t *value)
{
    //unsafe, use only if value is guaranteed to be in array
    for (uint8_t d = 0; d < MAX_DEPTH; d++)
        cur = cur->children[indexFrom(value, d)];
    return (uint8_t)cur->priv->aci;
}

static int16_t inline testFetchPaletteId(hexnode_t *cur, const uint8_t *value)
{
    //unsafe, use only if value is guaranteed to be in array
    for (uint8_t d = 0, c; d < MAX_DEPTH; d++) {
        c = indexFrom(value, d);
        if (cur->children[c])
            cur = cur->children[c];
        else
            return -1;
    }
    return (uint8_t)cur->priv->aci;
}

static inline int16_t findClosestInPalette(const ctx_t *ctx, const int32_t *irgba)
{
    uint32_t minDist = 4*256*256;
    int16_t bestFitId = -1;

    for (int16_t cid = 0; cid < ctx->nc; ++cid) {
        uint32_t dist = colorDist(ctx->centroids[cid], irgba);

        if (dist < minDist) {
            minDist = dist;
            bestFitId = cid;
        }
    }
    return bestFitId;
}

#define ERR_THRESH_SKIP (2)
#define MAX_ERROR_COMPONENT (24)
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
#define DEN_DITHERING (8)
#define DITHDIV(x) ((x) / ((DTYPE_ERROR)DEN_DITHERING))

static int generateDitheredBitmap(const ctx_t *ctx, uint8_t **bitmap, const uint8_t *rgba, const size_t len, const uint32_t width)
{
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
    const uint8_t *pix;
    
    for (uint32_t y = 0, lineId = 0; y < len; y += width, ++lineId) {
        for (uint32_t x = 0; x < width; ++x) {
            pix = &rgba[(y+x) << 2];
            pErrors = &errors[ERROR_ROW_LOC(lineId, rowLen, x)];

            sErr = 0;
            for (uint8_t k = 0; k < 4; k++) {
                pErrors[k] = SIGN(pErrors[k])*MIN((DTYPE_ERROR)MAX_ERROR_COMPONENT, ABS(pErrors[k]));
                sErr += pErrors[k]*pErrors[k];
            }

            unpackRgbaFp(pix, rgbaOrg);
            if (sErr > (DTYPE_ERROR)ERR_THRESH_SKIP) {
                //Add the residual error to the image pixel
                ADDERROR(rgbaOrg, pErrors);
                clipPackPack(rgbaOrg, irgba, crgba);
            } else {
                crgba[0] = pix[0];
                crgba[1] = pix[1];
                crgba[2] = pix[2];
                crgba[3] = pix[3];
            }
            //reset error accumulator of pixel
            memset(pErrors, 0, sizeof(rgbaErr));

            //fetch closest palette entry
            paletteEntryId = testFetchPaletteId(ctx->root, crgba);
            if (paletteEntryId < 0) {
                paletteEntryId = findClosestInPalette(ctx, irgba);
            }

            for (uint8_t k = 0; k < 4; k++)
                rgbaErr[k] = DITHDIV(rgbaOrg[k] - (DTYPE_ERROR)ctx->centroids[paletteEntryId][k]);

            //simplified edges conditions
            if (x < width - 2) {
                ADDERROR(pErrors + 4, rgbaErr);
                ADDERROR(pErrors + 8, rgbaErr);
            }
            if (y < len - width && x < width - 1 && x > 0) {
                pErrors = &errors[ERROR_ROW_LOC(lineId+1, rowLen, x) - 4];
                ADDERROR(pErrors,     rgbaErr);
                ADDERROR(pErrors + 4, rgbaErr);
                ADDERROR(pErrors + 8, rgbaErr);
            }
            if (y < len - 2*width) {
                pErrors = &errors[ERROR_ROW_LOC(lineId+2, rowLen, x)];
                ADDERROR(pErrors, rgbaErr);
            }
            (*bitmap)[y + x] = (uint8_t)paletteEntryId;
        }
    }
    free(errors);
    return 0;
}

static int generateBitmap(const ctx_t *ctx, uint8_t** bitmap, const uint8_t* rgba, const size_t len)
{
    hexnode_t *cur = ctx->root;

    *bitmap = (uint8_t*)calloc(len, sizeof(uint8_t));
    if (NULL == *bitmap)
        return -1;

    for (size_t k = 0; k < len; k++)
        (*bitmap)[k] = fetchPaletteId(ctx->root, &rgba[k<<2]);
    return 0;
}

PyObject *kdmeans_quantize(PyObject *self, PyObject *arg)
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

    uint16_t nc = (uint16_t)PyLong_AsLong(pnc);
    if (nc < 8 || nc > MAX_ENTRIES)
        return NULL;

    PyArrayObject *arr = (PyArrayObject*)pai;
    size_t len = PyArray_DIM(arr, 0);

    //Not flat or not mod4
    if (PyArray_NDIM(arr) != 1 || (len & 0b11))
        return NULL;
    size_t rawLen = len;
    len >>= 2;

    //8 MiB is an arbitrary limitation to not overflow
    if (width > len || (width > 0 && len % width) || width > 8192 || len > (8 << 20))
        return NULL;

    if (0 == (PyArray_FLAGS(arr) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)))
        return NULL;

    if (0 == (NPY_ARRAY_ALIGNED & PyArray_FLAGS(arr)))
        return NULL;

    const uint8_t *rgba = (const uint8_t*)PyArray_BYTES(arr);
    int err = 0;

    //catalog
    ctx_t *ctx = init();
    for (size_t k = 0; k < rawLen && !err; k += 4)
        err = insert(ctx, &rgba[k]);

    ctx->nc = (uint16_t)MIN(nc, ctx->nLeafs);

    //returned objects
    uint8_t *palette = (uint8_t*)calloc(N_COMPONENTS*ctx->nc, sizeof(uint8_t));
    uint8_t *bitmap = NULL;
    PyObject *rtup = NULL;
    PyArrayObject *arr_bitmap = NULL, *arr_pal = NULL;
    PyArray_Descr *desc_pal = NULL, *desc_bitmap = NULL;

    if (0 == err && palette) {
        if (ctx->nc < ctx->nLeafs) {
            init_centroids(ctx, len);

            uint32_t errorDist, prevError = 0;
            for (uint8_t k = 0; k < 4; ++k) {
                errorDist = kmeans_assign_update(ctx);
                if (errorDist == 0 || errorDist == prevError)
                    break;
                prevError = errorDist;
            }
        } else {
            for (uint8_t k = 0; k < ctx->nLeafs; k++) {
                memcpy(ctx->centroids[k], ctx->leafs[k]->cc, sizeof(int32_t)*N_COMPONENTS);
                ctx->leafs[k]->aci = k;
            }
        }
        if (width)
            err = generateDitheredBitmap(ctx, &bitmap, rgba, len, width);
        else
            err = generateBitmap(ctx, &bitmap, rgba, len);
        if (0 == err) {
            for (uint16_t k = 0; k < ctx->nc; ++k) {
                packRgba(&palette[k<<2], ctx->centroids[k]);
            }
        }
    }
    //generate bitmap
    if (0 == err) {
        npy_intp bitmap_dim = len;
        desc_bitmap = PyArray_DescrFromType(NPY_UBYTE);
        arr_bitmap = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, desc_bitmap, 1, &bitmap_dim, NULL, bitmap, 0, NULL);
        PyArray_ENABLEFLAGS(arr_bitmap, NPY_ARRAY_OWNDATA);

        if (arr_bitmap) {
            npy_intp palette_dim[2] = {ctx->nc, 4};
            desc_pal = PyArray_DescrFromType(NPY_UBYTE);
            arr_pal = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, desc_pal, 2, palette_dim, NULL, palette, 0, NULL);
            PyArray_ENABLEFLAGS(arr_pal, NPY_ARRAY_OWNDATA);

            if (arr_pal)
                rtup = Py_BuildValue("NN", arr_bitmap, arr_pal);
        }
    }
    destroy(ctx);

    if (NULL == rtup || err != 0) {
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
static PyMethodDef kdmeans_functions[] = {
    { "quantize", (PyCFunction)kdmeans_quantize, METH_O, kdmeans_quantize_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for kdmeans.
 */
PyDoc_STRVAR(kdmeans_doc, "kd-means 8-bit quanizer.");

static PyModuleDef kdmeans_def = {
    PyModuleDef_HEAD_INIT,
    "kdmeans",
    kdmeans_doc,
    0,              /* m_size */
    kdmeans_functions,/* m_methods */
    NULL,           /* m_slots */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit__kdmeans() {
    import_array();
    return PyModuleDef_Init(&kdmeans_def);
}