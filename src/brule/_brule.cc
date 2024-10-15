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
#include <stdio.h>
#include "librle.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

//static PyObject* RLEError;

PyDoc_STRVAR(brule_encode_doc, "encode(bitmap: NDArray[char]) -> bytes\
\
2-D bitmap encoder, output is raw RLE data and length.");

PyDoc_STRVAR(brule_decode_doc, "decode(data: bytes, length: int) -> NDArray[char]\
\
Run length decoder to fetch back the bitmap.");

PyObject *brule_encode(PyObject *self, PyObject *arg)
{
    PyObject* result = NULL;
    PyArrayObject* arr = (PyArrayObject*)arg;

    npy_intp width;
    npy_intp height;
    if (PyArray_NDIM(arr) != 2) {
        //PyErr_SetString(RLEError, "Expected 2-D bitmap shape.");
        return NULL;
    }

    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        //PyErr_SetString(RLEError, "Array must be C contiguous.");
        return NULL;
    }

    char* bitmap = PyArray_BYTES(arr);
    if (!bitmap) {
        //PyErr_SetString(RLEError, "Array cannot be null.");
        return NULL;
    } else {
        height = PyArray_DIM(arr, 0);
        width = PyArray_DIM(arr, 1);
        if (height < 8 || width < 8) {
            //PyErr_SetString(RLEError, "Invalid array shape.");
            return NULL;
        }
    }

    lrb_rle_result res = {0};
    if (!lrb_encode_bitmap(bitmap, (unsigned int)width, (unsigned int)height, &res)) {
        result = Py_BuildValue("y#", (const char*)res.data, (Py_ssize_t)res.length);

        lrb_destroy_rle(&res);
        return result;
    } else {
        lrb_destroy_rle(&res);
        //PyErr_SetString(RLEError, "Error in RL encoder.");
        return NULL;
    }
}

PyObject* brule_decode(PyObject* self, PyObject* arg)
{
    lrb_bitmap_result res = {0};
    Py_ssize_t sz = 0;
    char *data = NULL;

    if (PyBytes_Check(arg)) {
        sz = PyBytes_Size(arg);
        data = PyBytes_AsString(arg);
    } else if (PyByteArray_Check(arg)) {
        sz = PyByteArray_Size(arg);
        data = PyByteArray_AsString(arg);
    } else {
        //PyErr_SetString(RLEError, "Input must be bytes or bytearray.");
        return NULL;
    }

    if (!data || sz <= 0) {
        //PyErr_SetString(RLEError, "Zero size or no data.");
        return NULL;
    }

    if (!lrb_decode_rle((void*)data, (unsigned int)sz, &res)) {
        if (res.width && res.height) {
            npy_intp dims[2] = {res.height, res.width};
            PyArray_Descr *desc = PyArray_DescrFromType(NPY_BYTE);
            PyObject *arr_obj = PyArray_NewFromDescr(&PyArray_Type, desc, 2, dims, NULL, res.data, 0, NULL);
            PyArray_ENABLEFLAGS((PyArrayObject*)arr_obj, NPY_ARRAY_OWNDATA);

            res.data = NULL; //do not own anymore, numpy will free it when no longer needed.
            return arr_obj;
        } else {
            lrb_destroy_bitmap(&res);
            //PyErr_SetString(RLEError, "Incoherrent width or height.");
            return NULL;
        }
    } else {
        lrb_destroy_bitmap(&res);
        //PyErr_SetString(RLEError, "Error in run-length decoder. Maybe invalid data.");
        return NULL;
    }
    Py_RETURN_NONE;
}


/*
 * List of functions to add to brule in exec_brule().
 */
static PyMethodDef brule_functions[] = {
    { "encode", (PyCFunction)brule_encode, METH_O, brule_encode_doc },
    { "decode", (PyCFunction)brule_decode, METH_O, brule_decode_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for brule.
 */
PyDoc_STRVAR(brule_doc, "Bitmap RUn LEngth module");


static PyModuleDef brule_def = {
    PyModuleDef_HEAD_INIT,
    "brule",
    brule_doc,
    0,              /* m_size */
    brule_functions,           /* m_methods */
    NULL,//brule_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit__brule() {
    import_array();
    return PyModuleDef_Init(&brule_def);
}
