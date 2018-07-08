#ifdef PYMODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

PYBIND11_MODULE(neneshogi_cpp, m) {
	m.doc() = "neneshogi native module";
}
#endif
