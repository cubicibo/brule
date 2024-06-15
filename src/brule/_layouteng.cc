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
//#include <stdio.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#define MIN_MARGIN_BOX (8)
#define NUM_WINDOWS_MAX (2)

typedef struct container_s {
    int x1, y1;
    int x2, y2;
} container_t;

typedef struct shape_s {
    int width, height;
} shape_t;

static container_t container;  // container within video frame
static container_t last_windows[NUM_WINDOWS_MAX];
static uint8_t *screen;
static shape_t shape;
static uint8_t is_vertical_layout;
static uint8_t has_layout_changed;

PyDoc_STRVAR(layouteng_init_doc, "init(shape: Optional[tuple]) -> None\
\
Initialise the layout engine.");

PyDoc_STRVAR(layouteng_add_doc, "add(xp: int, yp: int, mask: NDArray[uint8]) -> None\
\
Add a frame to the layout.");

PyDoc_STRVAR(layouteng_find_doc, "find() -> (bbox, bbox_w0, bbox_w1, bool)\
\
Find and return the new optimal layout.");

#define CONTAINER_TO_PYTUP(c, xp, yp) Py_BuildValue("(iiii)", c.x1-xp, c.y1-yp, c.x2-xp, c.y2-yp);
#define BOX_AREA(box) (uint32_t)((box.x2-box.x1)*(box.y2-box.y1))
#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

//Adjust (z1) for a potential left shift, and take the largest coordinate out of the two.
#define ADJUST_DIM(X_OR_Y) \
    if (c1->X_OR_Y##1 > c2->X_OR_Y##1) { \
        c1->X_OR_Y##1 = c2->X_OR_Y##1;   \
    }                                    \
    c1->X_OR_Y##2 = MAX(c1->X_OR_Y##2, c2->X_OR_Y##2);

// Add 7 pixels to all dimensions for potential "repulsive" directional padding
#define DIRECTIONAL_PAD(DIM, COORD) \
    direction = DIM >= c1->COORD ? 1 : -1;         \
    if ((diff = (DIM - c1->COORD)*direction) > 0) {  \
        c1->COORD += direction*MIN(diff, MIN_MARGIN_BOX-1);  \
    }


static void or_containers(container_t *c1, const container_t *c2)
{
    ADJUST_DIM(x);
    ADJUST_DIM(y);
}

static void pad_container(const container_t *c2, container_t *c1)
{
    memcpy(c1, c2, sizeof(container_t));
    int diff, direction;
    DIRECTIONAL_PAD(0, y1);
    DIRECTIONAL_PAD(0, x1);

    DIRECTIONAL_PAD(shape.height, y2);
    DIRECTIONAL_PAD(shape.width, x2);
}

static void cut_vertical(const int margin, container_t *max_cont)
{
    int pixelExist;
    int xk, yk;

    pixelExist = 0;
    for (xk = max_cont->x1; xk < max_cont->x2 - margin && !pixelExist; xk++) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*shape.width + xk] > 0);
        }
    }
    max_cont->x1 = MAX(max_cont->x1, xk - (pixelExist & 0x1));

    pixelExist = 0;
    for (xk = max_cont->x2 - 1; xk >= max_cont->x1 + margin; xk--) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*shape.width + xk] > 0);
        }
        if (pixelExist)
            break;
    }
    max_cont->x2 = MIN(max_cont->x2, xk + (pixelExist > 0) + 1);

    pixelExist = 0;
    for (yk = max_cont->y2 - 1; yk >= max_cont->y1 + margin; yk--) {
        if(screen[yk*shape.width + max_cont->x1] || memcmp(&screen[yk*shape.width + max_cont->x1],
                                                           &screen[yk*shape.width + max_cont->x1 + 1],
                                                           max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y2 = MIN(max_cont->y2, yk + 1);

    for (yk = max_cont->y1; yk < max_cont->y2 - margin; yk++) {
        if(screen[yk*shape.width + max_cont->x1] || memcmp(&screen[yk*shape.width + max_cont->x1],
                                                           &screen[yk*shape.width + max_cont->x1 + 1],
                                                           max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y1 = MAX(max_cont->y1, yk);
}

static void cut_horizontal(const int margin, container_t *max_cont)
{
    int pixelExist;
    int xk, yk;

    for (yk = max_cont->y1; yk < max_cont->y2 - margin; yk++) {
        if (screen[yk*shape.width + max_cont->x1] || memcmp(&screen[yk*shape.width + max_cont->x1],
                                                            &screen[yk*shape.width + max_cont->x1 + 1],
                                                            max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y1 = MAX(max_cont->y1, yk);

    for (yk = max_cont->y2 - 1; yk >= max_cont->y1 + margin; yk--) {
        if (screen[yk*shape.width+max_cont->x1] || memcmp(&screen[yk*shape.width + max_cont->x1],
                                                          &screen[yk*shape.width + max_cont->x1 + 1],
                                                          max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y2 = MIN(max_cont->y2, yk + 1);

    pixelExist = 0;
    for (xk = max_cont->x2 - 1; xk >= max_cont->x1 + margin; xk--) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*shape.width + xk] > 0);
        }
        if (pixelExist)
            break;
    }
    max_cont->x2 = MIN(max_cont->x2, xk + (pixelExist & 0x1) + 1);

    pixelExist = 0;
    for (xk = max_cont->x1; xk < max_cont->x2 - margin && !pixelExist; xk++) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*shape.width + xk] > 0);
        }
    }
    max_cont->x1 = MAX(max_cont->x1, xk - (pixelExist & 0x1));
}

static void brute_force_windows(container_t *best, const container_t *cbox)
{
    const int margin = 8;
    int xk, yk;
    uint8_t is_vertical = (uint8_t)(-1);
    uint32_t best_score = BOX_AREA(container);
    uint32_t surface;
    container_t eval[NUM_WINDOWS_MAX];

    for (yk = cbox->y1 + margin; yk <= cbox->y2 - margin; yk++) {
        memcpy(eval, cbox, sizeof(container_t));
        memcpy(&eval[1], cbox, sizeof(container_t));
        eval[0].y2 = yk;
        eval[1].y1 = yk;

        cut_vertical(margin, &eval[0]);
        cut_vertical(margin, &eval[1]);

        surface = BOX_AREA(eval[0]) + BOX_AREA(eval[1]);
        if (surface < best_score) {
            best_score = surface;
            is_vertical = 1;
            memcpy(best, eval, NUM_WINDOWS_MAX*sizeof(container_t));
        }
    }

    for (xk = cbox->x1 + margin; xk <= cbox->x2 - margin; xk++) {
        memcpy(eval, cbox, sizeof(container_t));
        memcpy(&eval[1], cbox, sizeof(container_t));
        eval[0].x2 = xk;
        eval[1].x1 = xk;

        cut_horizontal(margin, &eval[0]);
        cut_horizontal(margin, &eval[1]);

        surface = BOX_AREA(eval[0]) + BOX_AREA(eval[1]);
        if (surface < best_score) {
            best_score = surface;
            is_vertical = 0;
            memcpy(best, eval, NUM_WINDOWS_MAX*sizeof(container_t));
        }
    }

    //No split or not worthwile?
    if (best_score >= BOX_AREA(container)) {
        memcpy(best, &container, sizeof(container_t));
        memcpy(&best[1], &container, sizeof(container_t));
        is_vertical = (uint8_t)(-1);
    }
    is_vertical_layout = is_vertical;
}

PyObject *layouteng_add(PyObject *self, PyObject *arg)
{
    if (Py_None == arg || !PyTuple_Check(arg) || 3 != PyTuple_Size(arg)) {
        return NULL;
    }

    PyObject *xpp = PyTuple_GetItem(arg, 0);
    PyObject *ypp = PyTuple_GetItem(arg, 1);
    if (!PyLong_Check(xpp) || !PyLong_Check(ypp))
        return NULL;
    int xp = (int)PyLong_AsLong(xpp);
    int yp = (int)PyLong_AsLong(ypp);

    PyArrayObject *arr = (PyArrayObject*)PyTuple_GetItem(arg, 2);
    if (PyArray_NDIM(arr) != 2) {
        return NULL;
    }

    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        return NULL;
    }

    char* bitmap = PyArray_BYTES(arr);
    npy_intp width;
    npy_intp height;
    if (!bitmap) {
        return NULL;
    } else {
        height = PyArray_DIM(arr, 0);
        width = PyArray_DIM(arr, 1);
        if (height <= 0 || width <= 0 || xp + width > shape.width || yp + height > shape.height) {
            return NULL;
        }
    }

    uint16_t x, y;
    int layout_changed = 0;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (bitmap[y*width + x]) {
                layout_changed = layout_changed || (!screen[(y+yp)*shape.width + xp + x]);
                screen[(y+yp)*shape.width + xp + x] = 1;
            }
        }
    }
    has_layout_changed = layout_changed;
    Py_RETURN_NONE;  
}

PyObject *layouteng_find(PyObject *self, PyObject *arg)
{
    container_t new_windows[2];
    container_t extended_container;
    container_t current_cont = {0};

    int x,y;
    for (y = 0; y < shape.height && 0 == screen[y*shape.width] && 0 == memcmp(&screen[y*shape.width], &screen[y*shape.width+1], shape.width-1); y++);
    current_cont.y1 = MIN(y, shape.height);

    for (y = (uint16_t)(shape.height - 1); y > current_cont.y1 && screen[y*shape.width] == 0 && 0 == memcmp(&screen[y*shape.width], &screen[y*shape.width+1], shape.width-1); y--);
    current_cont.y2 = MIN(y + 1, shape.height);

    current_cont.x1 = shape.width;
    for (y = current_cont.y1; y < current_cont.y2; y++) {
        for (x = 0; x < shape.width; x++) {
            if (screen[y*shape.width + x]) {
                current_cont.x2 = MAX(current_cont.x2, x);
                current_cont.x1 = MIN(current_cont.x1, x);
            }
        }
    }
    current_cont.x2 = MIN(current_cont.x2+1, shape.width);

    if (current_cont.x1 >= current_cont.x2 || current_cont.y1 >= current_cont.y2) {
        return NULL;
    }

    if (has_layout_changed) {
        or_containers(&container, &current_cont);

        //Generate a larger container with 7 pixels added in all direction (whenever possible)
        pad_container((const container_t *)&container, &extended_container);

        brute_force_windows(new_windows, &extended_container);
        memcpy(last_windows, new_windows, NUM_WINDOWS_MAX*sizeof(container_t));
    } else {
        if (last_windows[0].x2 == 0)
            return NULL;
        pad_container((const container_t *)&container, &extended_container);
        memcpy(new_windows, last_windows, NUM_WINDOWS_MAX*sizeof(container_t));
    }

    PyObject *py_cont, *py_w1, *py_w2;
    if ((uint8_t)is_vertical_layout > 1) {
        py_cont = CONTAINER_TO_PYTUP(container, 0, 0);
        py_w1 = CONTAINER_TO_PYTUP(new_windows[0], container.x1, container.y1);
        py_w2 = CONTAINER_TO_PYTUP(new_windows[1], container.x1, container.y1);
    }
    else {
        py_cont = CONTAINER_TO_PYTUP(extended_container, 0, 0);
        py_w1 = CONTAINER_TO_PYTUP(new_windows[0], extended_container.x1, extended_container.y1);
        py_w2 = CONTAINER_TO_PYTUP(new_windows[1], extended_container.x1, extended_container.y1);
    }

    if (py_cont && py_w1 && py_w2) {
        PyObject *r_tup = Py_BuildValue("(NNNb)", py_cont, py_w1, py_w2, (char)is_vertical_layout);
        if (r_tup)
            return r_tup;
    }
    if (py_cont)
        Py_DECREF(py_cont);
    if (py_w1)
        Py_DECREF(py_w1);
    if (py_w2)
        Py_DECREF(py_w2);
    return NULL;
}

PyObject* layouteng_init(PyObject* self, PyObject* arg)
{
    if (Py_None == arg || !PyTuple_Check(arg) || 2 != PyTuple_Size(arg)) {
        if (screen)
            free(screen);
        screen = NULL;
        if (Py_None != arg)
            return NULL;
        Py_RETURN_NONE;  
    }

    shape.width = (uint16_t)PyLong_AsLong(PyTuple_GetItem(arg, 0));
    shape.height = (uint16_t)PyLong_AsLong(PyTuple_GetItem(arg, 1));
    
    if (shape.height > 1088 || shape.height < 16)
        return NULL;
    if (shape.width > 1928 || shape.width < 16)
        return NULL;

    is_vertical_layout = (uint8_t)(-1);
    has_layout_changed = 0;
    container.x1 = shape.width;
    container.x2 = 0;
    container.y1 = shape.height;
    container.y2 = 0;
    memset(last_windows, 0, NUM_WINDOWS_MAX*sizeof(container_t));

    if (screen)
        free(screen);
    screen = (uint8_t*)calloc(shape.width*shape.height, sizeof(uint8_t));
    if (!screen)
        return NULL;

    Py_RETURN_NONE;
}

/*
 * List of functions to add to brule in exec_brule().
 */
static PyMethodDef layouteng_functions[] = {
    { "init", (PyCFunction)layouteng_init, METH_O, layouteng_init_doc },
    { "add", (PyCFunction)layouteng_add, METH_O, layouteng_add_doc },
    { "find", (PyCFunction)layouteng_find, METH_NOARGS, layouteng_find_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Documentation for layouteng.
 */
PyDoc_STRVAR(layouteng_doc, "Multi-windows layout finder.");

static PyModuleDef layouteng_def = {
    PyModuleDef_HEAD_INIT,
    "layouteng",
    layouteng_doc,
    0,              /* m_size */
    layouteng_functions,/* m_methods */
    NULL,           /* m_slots */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit__layouteng() {
    import_array();
    return PyModuleDef_Init(&layouteng_def);
}
