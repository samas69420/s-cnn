#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <pthread.h>

PyObject* pool2d(PyObject* self, PyObject* args){

    PyObject* spiking_frame3d;
    PyObject* done_mask3d;
    long window_size;

    if(!PyArg_ParseTuple(args, "OOl", &spiking_frame3d, &done_mask3d, &window_size)) {
        return NULL;
    }

    // get the number of dimensions of the numpy arrays

    int ndimframe = PyArray_NDIM((PyArrayObject*) spiking_frame3d);
    int ndimmask = PyArray_NDIM((PyArrayObject*) done_mask3d);

    if (ndimframe != 3 || ndimmask != 3) {
        PyErr_SetString(PyExc_TypeError, "spiking frame and and mask should be both 3d arrays");
        return NULL;
    }

    Py_ssize_t dimframe[3] = {0};
    Py_ssize_t dimmask[3] = {0};

    for (int i = 0; i < 3; ++i) {
        dimframe[i] = PyArray_DIMS((PyArrayObject *)spiking_frame3d)[i];
        dimmask[i] = PyArray_DIMS((PyArrayObject *)done_mask3d)[i];
    }

    Py_ssize_t input_shape[3] = {dimframe[0],dimframe[1],dimframe[2]};

    if (input_shape[1] % window_size != 0 || input_shape[1] % window_size != 0) {
        PyErr_SetString(PyExc_TypeError, "input spiking frame shape and window size should be compatible");
        return NULL;
    }

    Py_ssize_t output_shape[3] = {0};
    output_shape[0] = input_shape[0];
    output_shape[1] = input_shape[1] / window_size;
    output_shape[2] = input_shape[2] / window_size;

    if (output_shape[0] != dimmask[0] || output_shape[1] != dimmask[1] || output_shape[2] != dimmask[2]) {
        PyErr_SetString(PyExc_TypeError, "output frame and mask should have the same shape");
        return NULL;
    }

    double*** data_spiking_frame;
    PyArray_AsCArray((PyObject**)&spiking_frame3d, &data_spiking_frame, input_shape, 3, PyArray_DescrFromType(NPY_DOUBLE)); 

    double*** data_done_mask;
    PyArray_AsCArray((PyObject**)&done_mask3d, &data_done_mask, output_shape, 3, PyArray_DescrFromType(NPY_DOUBLE)); 

    PyObject* result = PyArray_Zeros(3, output_shape, PyArray_DescrFromType(NPY_DOUBLE), 0);
    double*** data_result;
    PyArray_AsCArray((PyObject**)&result, &data_result, output_shape, 3, PyArray_DescrFromType(NPY_DOUBLE)); 

    // do the pooling

    for(int k = 0; k < input_shape[0]; k++){
        for(int i = 0; i < input_shape[1]; i += window_size){
            for(int j = 0; j < input_shape[2]; j += window_size){
                int _sum = 0;
                for(int _i = i;  _i < i + window_size; _i++){
                    for(int _j = j;  _j < j + window_size; _j++){
                        _sum += (long int)data_spiking_frame[k][_i][_j];
                    }
                }
                if(_sum >= 1 && data_done_mask[k][i/window_size][j/window_size] == 1){
                    data_result[k][i/window_size][j/window_size] = 1;
                    data_done_mask[k][i/window_size][j/window_size] = 0;
                }
            }
        }
    }


    PyArray_Free((PyObject*)spiking_frame3d , data_spiking_frame); 
    PyArray_Free((PyObject*)done_mask3d, data_done_mask); 
    PyArray_Free((PyObject*)result, data_result); 

    return result;
}

PyMethodDef methods[] = {
    {"pool2d", pool2d, METH_VARARGS, "pooling between 3d arrays but along 2 dimensions assuming a squared 2d window"},
    {NULL, NULL, 0, NULL}
};

struct PyModuleDef fastpool = {
    PyModuleDef_HEAD_INIT,
    "fastpool",
    "Module for custom pooling implementation - author: samas69420",
    -1, 
    methods
};

PyMODINIT_FUNC PyInit_fastpool() {
    printf("using fast pooling module\n");
    import_array();
    return PyModule_Create(&fastpool);
}

//$ gcc -shared fastpool.c -I/home/samas/.pyenv/versions/torchenv/lib/python3.11/site-packages/numpy/core/include -I/usr/include/python3.11 -o fastpool.so -fPIC -O3 -fopt-info-vec 
