#ifdef PYMODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "all.h"

namespace py = pybind11;

// 演算での利用を想定し、enumをクラスではなく単純なintとして見せる (Color.BLACK == 0)
#define ENUM_TO_INT(TYPE) namespace pybind11 {\
	namespace detail {\
		template <> struct type_caster<TYPE> {\
		public:\
			PYBIND11_TYPE_CASTER(TYPE, _(#TYPE));\
			bool load(handle src, bool) {\
				/* Extract PyObject from handle */\
				PyObject *source = src.ptr();\
				/* Try converting into a Python integer value */\
				PyObject *tmp = PyNumber_Long(source);\
				if (!tmp)\
					return false;\
				/* Now try to convert into a C++ int */\
				value = static_cast<TYPE>(PyLong_AsLong(tmp));\
				Py_DECREF(tmp);\
				/* Ensure return code was OK (to avoid out-of-range errors etc) */\
				return !(value == -1 && !PyErr_Occurred());\
			}\
\
			static handle cast(TYPE src, return_value_policy /* policy */, handle /* parent */) {\
				return PyLong_FromLong(static_cast<long>(src));\
			}\
		};\
	}\
}

ENUM_TO_INT(Color);
ENUM_TO_INT(File);
ENUM_TO_INT(Rank);
ENUM_TO_INT(Square);
ENUM_TO_INT(Move);
#endif
