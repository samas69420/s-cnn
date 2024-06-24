#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

PyObject* conv2d(PyObject* self, PyObject* args){

    PyObject* arr3d;
    PyObject* kernel3d;

    if(!PyArg_ParseTuple(args, "OO", &arr3d, &kernel3d)) {
        return NULL;
    }

    // get the number of dimensions of the numpy arrays

    int ndimarr = PyArray_NDIM((PyArrayObject*) arr3d);
    int ndimkernel = PyArray_NDIM((PyArrayObject*) kernel3d);

    if (ndimarr != 3 || ndimkernel != 3) {
        PyErr_SetString(PyExc_TypeError, "array and kernel should be both 3d arrays");
        return NULL;
    }

    // get actual shapes

    Py_ssize_t dimarr[3] = {0};
    Py_ssize_t dimkernel[3] = {0};

    for (int i = 0; i < 3; ++i) {
        dimarr[i] = PyArray_DIMS((PyArrayObject *)arr3d)[i];
        dimkernel[i] = PyArray_DIMS((PyArrayObject *)kernel3d)[i];
    }

    if (dimarr[0] != dimkernel[0]) {
        PyErr_SetString(PyExc_ValueError, "array and kernel must have the same dimension for axis 0");
        return NULL;
    }
    if (dimarr[1] < dimkernel[1] || dimarr[2] < dimkernel[2]) {
        PyErr_SetString(PyExc_ValueError, "the width and height of the array must be greater than or equal to those of the kernel");
        return NULL;
    }

    // setup the variables for computation

    int dimresult1 = dimarr[1] - dimkernel[1] + 1;
    int dimresult2 = dimarr[2] - dimkernel[2] + 1;
    npy_intp output_shape[] = {dimresult1,dimresult2};

    PyObject* result = PyArray_Zeros(2, output_shape, PyArray_DescrFromType(NPY_DOUBLE), 0);

    double*** data_arr3d;
    double*** data_kernel3d;
    PyArray_AsCArray((PyObject**)&arr3d, &data_arr3d, dimarr,3, PyArray_DescrFromType(NPY_DOUBLE)); // refcount 3->4
    PyArray_AsCArray((PyObject**)&kernel3d, &data_kernel3d, dimkernel,3, PyArray_DescrFromType(NPY_DOUBLE)); // refcount 3->1

    double* data_result = (double*) PyArray_DATA((PyArrayObject*)result);

    // actual convolution operation

    for (int i=0; i < dimresult1; i++){
        for(int j=0; j < dimresult2; j++){
            for (int _k=0; _k < dimarr[0]; _k++){
                for (int _i=i; _i < i+dimkernel[1]; _i++){
                    for (int _j=j; _j < j+dimkernel[2]; _j++)
                        data_result[i*dimresult1 + j] += data_arr3d[_k][_i][_j]*data_kernel3d[_k][_i-i][_j-j];
                }
            }
        }
    }

    PyArray_Free((PyObject*)arr3d, data_arr3d); 
    PyArray_Free((PyObject*)kernel3d, data_kernel3d); 

    return result;
}

PyMethodDef methods[] = {
    {"conv2d", conv2d, METH_VARARGS, "2d convolution between 3d arrays assumming valid mode and stride 1 for both axes"},
    {NULL, NULL, 0, NULL}
};

struct PyModuleDef fastconv = {
    PyModuleDef_HEAD_INIT,
    "fastconv",
    "Module for custom convolution implementation - author: samas69420",
    -1, 
    methods
};

PyMODINIT_FUNC PyInit_fastconv() {
    printf("using fast convolution module\n");
    import_array();
    return PyModule_Create(&fastconv);
}

//$ gcc -shared fastconv.c -I/home/samas/.pyenv/versions/torchenv/lib/python3.11/site-packages/numpy/core/include -I/usr/include/python3.11 -o fastconv.so -fPIC -O3 -fopt-info-vec -mavx2 -mfma
