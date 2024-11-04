#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <utility>
#include <stdexcept>

namespace py = pybind11;


enum PyStringKind {
    PyString_BYTE_KIND,
    PyString_UNICODE1_KIND,
    PyString_UNICODE2_KIND,
    PyString_UNICODE4_KIND
};


class PyStringWrap {
public:
    explicit PyStringWrap(const py::handle &handle)
    : _pyobj(handle.ptr()) {
        if (py::isinstance<py::bytes>(handle)) {
            Py_ssize_t size;
            char *ptr;
            PyBytes_AsStringAndSize(_pyobj, &ptr, &size);
            _len = (size_t)size;
            _ptr.reset(ptr, &no_delete);
            _kind = PyString_BYTE_KIND;
        }
        else if (py::isinstance<py::bytearray>(handle)) {
            _len = PyByteArray_GET_SIZE(_pyobj);
            _ptr.reset(PyByteArray_AS_STRING(_pyobj), &no_delete);
            _kind = PyString_BYTE_KIND;
        }
        else if (py::isinstance<py::str>(handle)) {
            _len = (size_t)PyUnicode_GET_LENGTH(_pyobj);
            _ptr.reset(PyUnicode_DATA(_pyobj), &no_delete);
            switch (PyUnicode_KIND(_pyobj)) {
                case PyUnicode_1BYTE_KIND:
                    _kind = PyString_UNICODE1_KIND;
                    break;
                case PyUnicode_2BYTE_KIND:
                    _kind = PyString_UNICODE2_KIND;
                    break;
                case PyUnicode_4BYTE_KIND:
                    _kind = PyString_UNICODE4_KIND;
                    break;
                default:
                    throw std::runtime_error("unexpected unicode kind");
            }
        }
        else
            throw py::value_error("can only initialize PyStringWrap objects from bytes, bytearray or str objects.");
        inc_ref();
    }

    PyStringWrap(const PyStringWrap &other)
    : _pyobj(other._pyobj), _len(other._len), _ptr(other._ptr), _kind(other._kind) {
        inc_ref();
    }

    PyStringWrap(PyStringWrap &&other) noexcept
    : _pyobj(other._pyobj), _len(other._len), _ptr(std::move(other._ptr)), _kind(other._kind) {
        other._pyobj = nullptr;
    }

    ~PyStringWrap() {
        if (_pyobj)
            dec_ref();
    }

    PyStringWrap& operator=(const PyStringWrap &other) {
        if (this != &other) {
            dec_ref();
            _pyobj = other._pyobj;
            _len = other._len;
            _ptr = other._ptr;
            _kind = other._kind;
            inc_ref();
        }
        return *this;
    }

    PyStringWrap& operator=(PyStringWrap &&other) noexcept {
        if (this != &other) {
            dec_ref();
            _pyobj = other._pyobj;
            other._pyobj = nullptr;
            _len = other._len;
            _ptr = std::move(other._ptr);
            _kind = other._kind;
        }
        return *this;
    }

    void toUnicode4() {
        switch (_kind) {
            case PyString_BYTE_KIND:
                throw std::runtime_error("cannot convert bytes/bytearray to unicode");
            case PyString_UNICODE4_KIND:
                break;
            case PyString_UNICODE1_KIND:
            case PyString_UNICODE2_KIND:
                _ptr.reset(PyUnicode_AsUCS4Copy(_pyobj), [](void *p) { PyMem_Free(p); });
                if (!_ptr)
                    throw std::bad_alloc();
                _kind = PyString_UNICODE4_KIND;
                break;
            default:
                throw std::runtime_error("unexpected unicode kind");
        }
    }

    inline auto len() const { return _len; }
    inline auto ptr() const { return _ptr.get(); }
    inline auto kind() const { return _kind; }

private:
    PyObject *_pyobj;
    size_t _len;
    std::shared_ptr<const void> _ptr;
    PyStringKind _kind;

    void inc_ref() const {
        Py_INCREF(_pyobj);
    }

    void dec_ref() const {
        Py_DECREF(_pyobj);
    }

    static void no_delete(const void *p) {}
};
