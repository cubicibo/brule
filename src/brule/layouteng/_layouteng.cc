/**
 MIT License

 Copyright (c) 2024-2025, cubicibo

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

typedef struct layout_s {
    container_t container;
    container_t last_windows[NUM_WINDOWS_MAX];
    uint8_t *screen;
    shape_t shape;
    uint8_t is_vertical_layout;
    uint8_t has_layout_changed;
} layout_t;

typedef struct module_s {
    unsigned int instances_count;
    uint32_t valid_instances;
    layout_t refs[32];
} module_t;

static module_t brule_le;

PyDoc_STRVAR(layouteng_init_doc, "init(shape: Optional[tuple]) -> None\
\
Initialise the layout engine.");

PyDoc_STRVAR(layouteng_destroy_doc, "destroy(shape: Optional[tuple]) -> None\
\
Destroy the instance of the layout engine.");

PyDoc_STRVAR(layouteng_add_doc, "add(xp: int, yp: int, mask: NDArray[uint8]) -> None\
\
Add a frame to the layout.");

PyDoc_STRVAR(layouteng_find_doc, "find() -> (bbox, bbox_w0, bbox_w1, bool)\
\
Find and return the new optimal layout.");

PyDoc_STRVAR(layouteng_get_doc, "get_container() -> (x1, y1, x2, y2)\
\
Find and return the raw container without any additional padding.");

#define CONTAINER_TO_PYTUP(c, xp, yp) Py_BuildValue("(iiii)", c.x1-xp, c.y1-yp, c.x2-xp, c.y2-yp);
#define BOX_AREA(box) (uint32_t)((box.x2-box.x1)*(box.y2-box.y1))
#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)
#define DIM(C, X) (C.X##2 - C.X##1)

//Adjust (z1) for a potential left shift, and take the largest coordinate out of the two.
#define ADJUST_DIM(X_OR_Y) \
    if (c1->X_OR_Y##1 > c2->X_OR_Y##1) { \
        c1->X_OR_Y##1 = c2->X_OR_Y##1;   \
    }                                    \
    c1->X_OR_Y##2 = MAX(c1->X_OR_Y##2, c2->X_OR_Y##2);

// Add 7 pixels to all dimensions for potential "repulsive" directional padding
#define DIRECTIONAL_PAD(DIM, COORD, DIFF, DIRECTION) \
    DIRECTION = DIM >= c1->COORD ? 1 : -1;         \
    if ((DIFF = (DIM - c1->COORD)*DIRECTION) > 0) {  \
        c1->COORD += DIRECTION*MIN(DIFF, MIN_MARGIN_BOX-1);  \
    }


static inline void or_containers(container_t *c1, const container_t *c2)
{
    ADJUST_DIM(x);
    ADJUST_DIM(y);
}

static inline void pad_container(shape_t *shape, const container_t *c2, container_t *c1)
{
    memcpy(c1, c2, sizeof(container_t));
    int diff, direction;

    DIRECTIONAL_PAD(0, y1, diff, direction);
    DIRECTIONAL_PAD(0, x1, diff, direction);

    DIRECTIONAL_PAD(shape->height, y2, diff, direction);
    DIRECTIONAL_PAD(shape->width,  x2, diff, direction);
}

static inline layout_t* check_instance(PyObject *arg)
{
    if (!PyLong_Check(arg))
        return NULL;

    const uint32_t iid = (uint32_t)PyLong_AsLong(arg);
    if ((brule_le.valid_instances >> iid) & 0b1)
        return &brule_le.refs[iid];

    return NULL;
}

static void cut_vertical(layout_t *le, const int margin, container_t *max_cont)
{
    int pixelExist;
    int xk, yk;

    const uint8_t *screen = (const uint8_t*)le->screen;

    pixelExist = 0;
    for (xk = max_cont->x1; xk < max_cont->x2 - margin && !pixelExist; xk++) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*le->shape.width + xk] > 0);
        }
    }
    max_cont->x1 = MAX(max_cont->x1, xk - (pixelExist & 0x1));

    pixelExist = 0;
    for (xk = max_cont->x2 - 1; xk >= max_cont->x1 + margin; xk--) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*le->shape.width + xk] > 0);
        }
        if (pixelExist)
            break;
    }
    max_cont->x2 = MIN(max_cont->x2, MAX(max_cont->x1 + margin, xk + (pixelExist > 0)));

    for (yk = max_cont->y2 - 1; yk >= max_cont->y1 + margin; yk--) {
        if(screen[yk*le->shape.width + max_cont->x1] || memcmp(&screen[yk*le->shape.width + max_cont->x1],
                                                               &screen[yk*le->shape.width + max_cont->x1 + 1],
                                                               max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y2 = MIN(max_cont->y2, yk + 1);

    for (yk = max_cont->y1; yk < max_cont->y2 - margin; yk++) {
        if(screen[yk*le->shape.width + max_cont->x1] || memcmp(&screen[yk*le->shape.width + max_cont->x1],
                                                               &screen[yk*le->shape.width + max_cont->x1 + 1],
                                                               max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y1 = MAX(max_cont->y1, yk);
}

static void cut_horizontal(layout_t *le, const int margin, container_t *max_cont)
{
    int pixelExist;
    int xk, yk;

    const uint8_t *screen = (const uint8_t*)le->screen;

    for (yk = max_cont->y1; yk < max_cont->y2 - margin; yk++) {
        if (screen[yk*le->shape.width + max_cont->x1] || memcmp(&screen[yk*le->shape.width + max_cont->x1],
                                                                &screen[yk*le->shape.width + max_cont->x1 + 1],
                                                                max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y1 = MAX(max_cont->y1, yk);

    for (yk = max_cont->y2 - 1; yk >= max_cont->y1 + margin; yk--) {
        if (screen[yk*le->shape.width+max_cont->x1] || memcmp(&screen[yk*le->shape.width + max_cont->x1],
                                                              &screen[yk*le->shape.width + max_cont->x1 + 1],
                                                              max_cont->x2 - max_cont->x1 - 1)) {
            break;
        }
    }
    max_cont->y2 = MIN(max_cont->y2, yk + 1);

    pixelExist = 0;
    for (xk = max_cont->x2 - 1; xk >= max_cont->x1 + margin; xk--) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*le->shape.width + xk] > 0);
        }
        if (pixelExist)
            break;
    }
    max_cont->x2 = MIN(max_cont->x2, MAX(max_cont->x1 + margin, xk + (pixelExist > 0)));

    pixelExist = 0;
    for (xk = max_cont->x1; xk < max_cont->x2 - margin && !pixelExist; xk++) {
        for (yk = max_cont->y1; yk < max_cont->y2; yk++) {
            pixelExist = pixelExist || (screen[yk*le->shape.width + xk] > 0);
        }
    }
    max_cont->x1 = MAX(max_cont->x1, xk - (pixelExist & 0x1));
}

static void brute_force_windows(layout_t *le, container_t *best, const container_t *cbox)
{
    const int margin = 8;
    int xk, yk;
    uint8_t is_vertical = (uint8_t)(-1);
    uint32_t best_score = BOX_AREA(le->container);
    uint32_t surface;
    container_t eval[NUM_WINDOWS_MAX];

    for (yk = cbox->y1 + margin; yk <= cbox->y2 - margin; yk++) {
        memcpy(eval, cbox, sizeof(container_t));
        memcpy(&eval[1], cbox, sizeof(container_t));
        eval[0].y2 = yk;
        eval[1].y1 = yk;

        cut_vertical(le, margin, &eval[0]);
        cut_vertical(le, margin, &eval[1]);

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

        cut_horizontal(le, margin, &eval[0]);
        cut_horizontal(le, margin, &eval[1]);

        surface = BOX_AREA(eval[0]) + BOX_AREA(eval[1]);
        if (surface < best_score) {
            best_score = surface;
            is_vertical = 0;
            memcpy(best, eval, NUM_WINDOWS_MAX*sizeof(container_t));
        }
    }

    //No split or not worthwile?
    if (best_score >= BOX_AREA(le->container)) {
        memcpy(best, &le->container, sizeof(container_t));
        memcpy(&best[1], &le->container, sizeof(container_t));
        is_vertical = (uint8_t)(-1);
    }
    le->is_vertical_layout = is_vertical;
}

static int get_current_container(layout_t *le, container_t *current_cont)
{
    int x,y,md;
    uint8_t *screen = le->screen;

    for (y = 0; y < le->shape.height && 0 == screen[y*le->shape.width] && 0 == memcmp(&screen[y*le->shape.width], &screen[y*le->shape.width+1], le->shape.width-1); y++);
    current_cont->y1 = MIN(y, le->shape.height);

    for (y = (uint16_t)(le->shape.height - 1); y > current_cont->y1 && screen[y*le->shape.width] == 0 && 0 == memcmp(&screen[y*le->shape.width], &screen[y*le->shape.width+1], le->shape.width-1); y--);
    current_cont->y2 = MIN(y + 1, le->shape.height);

    current_cont->x1 = le->shape.width;
    for (y = current_cont->y1; y < current_cont->y2; y++) {
        for (x = 0; x < le->shape.width; x++) {
            if (screen[y*le->shape.width + x]) {
                current_cont->x2 = MAX(current_cont->x2, x);
                current_cont->x1 = MIN(current_cont->x1, x);
            }
        }
    }
    current_cont->x2 = MIN(current_cont->x2+1, le->shape.width);

    if (!(current_cont->x1 >= current_cont->x2 || current_cont->y1 >= current_cont->y2)) {
        x = MIN_MARGIN_BOX - (current_cont->x2 - current_cont->x1);

        if (x > 0) {
            md = MIN(x, current_cont->x1);
            current_cont->x1 -= md;
            current_cont->x2 += (x - md);
        }

        x = MIN_MARGIN_BOX - (current_cont->y2 - current_cont->y1);
        if (x > 0) {
            md = MIN(x, current_cont->y1);
            current_cont->y1 -= md;
            current_cont->y2 += (x - md);
        }

        return true;
    }
    return false;
}

PyObject *layouteng_add(PyObject *self, PyObject *arg)
{    
    if (Py_None == arg || !PyTuple_Check(arg) || 4 != PyTuple_Size(arg)) {
        return NULL;
    }

    layout_t *le = check_instance(PyTuple_GetItem(arg, 0));
    if (!le)
        return NULL;

    PyObject *xpp = PyTuple_GetItem(arg, 1);
    PyObject *ypp = PyTuple_GetItem(arg, 2);
    if (!PyLong_Check(xpp) || !PyLong_Check(ypp))
        return NULL;
    int xp = (int)PyLong_AsLong(xpp);
    int yp = (int)PyLong_AsLong(ypp);

    PyArrayObject *arr = (PyArrayObject*)PyTuple_GetItem(arg, 3);
    if (PyArray_NDIM(arr) != 2) {
        return NULL;
    }

    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        return NULL;
    }

    char* bitmap = PyArray_BYTES(arr);
    npy_intp width, height;

    if (!bitmap) {
        return NULL;
    } else {
        height = PyArray_DIM(arr, 0);
        width = PyArray_DIM(arr, 1);
        if (height <= 0 || width <= 0 || xp + width > le->shape.width || yp + height > le->shape.height) {
            return NULL;
        }
    }

    uint16_t x, y;
    uint8_t *screen = le->screen;
    int layout_changed = 0;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (bitmap[y*width + x]) {
                layout_changed = layout_changed || (!screen[(y+yp)*le->shape.width + xp + x]);
                screen[(y+yp)*le->shape.width + xp + x] = 1;
            }
        }
    }
    le->has_layout_changed = le->has_layout_changed || layout_changed;
    Py_RETURN_NONE;
}

PyObject *layouteng_find(PyObject *self, PyObject *arg)
{
    layout_t *le = check_instance(arg);
    if (!le)
        return NULL;

    container_t new_windows[NUM_WINDOWS_MAX];
    container_t extended_container;
    container_t current_cont = {0};

    if (le->has_layout_changed) {
        if (!get_current_container(le, &current_cont)) {
            return NULL;
        }
        or_containers(&le->container, &current_cont);

        if (DIM(le->container, x) < MIN_MARGIN_BOX || DIM(le->container, y) < MIN_MARGIN_BOX)
            return NULL;

        //Generate a larger container with 7 pixels added in all direction (whenever possible)
        pad_container(&le->shape, (const container_t *)&le->container, &extended_container);

        brute_force_windows(le, new_windows, &extended_container);
        memcpy(le->last_windows, new_windows, NUM_WINDOWS_MAX*sizeof(container_t));
        le->has_layout_changed = 0;
    } else {
        if (le->last_windows[0].x2 == 0)
            return NULL;
        pad_container(&le->shape, (const container_t *)&le->container, &extended_container);
    }

    PyObject *py_cont, *py_w1, *py_w2;
    if ((uint8_t)le->is_vertical_layout > 1) {
        py_cont = CONTAINER_TO_PYTUP(le->container, 0, 0);
        py_w1 = CONTAINER_TO_PYTUP(le->last_windows[0], le->container.x1, le->container.y1);
        py_w2 = CONTAINER_TO_PYTUP(le->last_windows[1], le->container.x1, le->container.y1);
    } else {
        py_cont = CONTAINER_TO_PYTUP(extended_container, 0, 0);
        py_w1 = CONTAINER_TO_PYTUP(le->last_windows[0], extended_container.x1, extended_container.y1);
        py_w2 = CONTAINER_TO_PYTUP(le->last_windows[1], extended_container.x1, extended_container.y1);
    }

    if (py_cont && py_w1 && py_w2) {
        PyObject *r_tup = Py_BuildValue("(NNNb)", py_cont, py_w1, py_w2, (char)le->is_vertical_layout);
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
    if (!PyTuple_Check(arg) || 2 != PyTuple_Size(arg))
        return NULL;

    uint32_t iid;
    for (iid = 0; iid < 32 && ((brule_le.valid_instances >> iid) & 0b1); ++iid);

    if (iid == 32)
        return NULL; //we can't store more than 32 instances of the layout engine.

    layout_t *le = &brule_le.refs[iid];

    le->shape.width = (uint16_t)PyLong_AsLong(PyTuple_GetItem(arg, 0));
    le->shape.height = (uint16_t)PyLong_AsLong(PyTuple_GetItem(arg, 1));

    if (le->shape.height > 1088 || le->shape.height < MIN_MARGIN_BOX)
        return NULL;
    if (le->shape.width > 1928 || le->shape.width < MIN_MARGIN_BOX)
        return NULL;

    le->is_vertical_layout = (uint8_t)(-1);
    le->has_layout_changed = 0;
    le->container.x1 = le->shape.width;
    le->container.x2 = 0;
    le->container.y1 = le->shape.height;
    le->container.y2 = 0;
    memset(le->last_windows, 0, NUM_WINDOWS_MAX*sizeof(container_t));

    le->screen = (uint8_t*)calloc(le->shape.width * le->shape.height, sizeof(uint8_t));
    if (!le->screen)
        return NULL;

    brule_le.valid_instances |= (1 << iid);
    brule_le.instances_count++;

    return PyLong_FromLong((long)iid);
}

PyObject* layouteng_destroy(PyObject* self, PyObject* arg)
{
    if (Py_None == arg || !PyLong_Check(arg))
        return NULL;

    uint32_t instance_id = (uint32_t)PyLong_AsLong(arg);
    uint32_t instance_bit = (1u << instance_id);

    if (instance_id >= 32 || !(brule_le.valid_instances & instance_bit) || !brule_le.instances_count)
        return NULL;

    brule_le.valid_instances &= ~instance_bit;
    brule_le.instances_count--;

    free(brule_le.refs[instance_id].screen);
    brule_le.refs[instance_id].screen = NULL;

    Py_RETURN_NONE;
}


PyObject* layouteng_get_container(PyObject* self, PyObject* arg)
{
    layout_t *le = check_instance(arg);
    if (!le)
        Py_RETURN_NONE;
    return CONTAINER_TO_PYTUP(le->container, 0, 0);
}

/*
 * List of functions to add to brule in exec_brule().
 */
static PyMethodDef layouteng_functions[] = {
    { "init", (PyCFunction)layouteng_init, METH_O, layouteng_init_doc },
    { "destroy", (PyCFunction)layouteng_destroy, METH_O, layouteng_destroy_doc },
    { "add", (PyCFunction)layouteng_add, METH_O, layouteng_add_doc },
    { "get_container", (PyCFunction)layouteng_get_container, METH_O, layouteng_get_doc },
    { "find", (PyCFunction)layouteng_find, METH_O, layouteng_find_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

void layouteng_free(void *m)
{
    uint32_t mask = brule_le.valid_instances;

    for (int k = 0; k < 32 && brule_le.instances_count; ++k)
    {
        if (mask & (1 << k))
        {
            free(brule_le.refs[k].screen);
            --brule_le.instances_count;
        }
    }
    brule_le.instances_count = brule_le.valid_instances = 0;
}

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
    layouteng_free, /* m_free */
};

PyMODINIT_FUNC PyInit__layouteng() {
    import_array();
    memset(&brule_le, 0, sizeof(brule_le));
    return PyModuleDef_Init(&layouteng_def);
}
