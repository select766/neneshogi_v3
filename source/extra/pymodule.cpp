#ifdef PYMODULE
#include "pymodule.h"
#include "../engine/user-engine/dnn_converter.h"

PYBIND11_MODULE(neneshogi_cpp, m) {
	m.doc() = "neneshogi native module";
	py::enum_<Color>(m, "Color")
		.value("BLACK", BLACK)
		.value("WHITE", WHITE)
		.value("COLOR_NB", COLOR_NB);
	py::class_<DNNConverter>(m, "DNNConverter")
		.def(py::init<int, int>())
		.def("set_packed_sfen", &DNNConverter::set_packed_sfen)
		.def("set_sfen", &DNNConverter::set_sfen)
		.def("board_shape", &DNNConverter::board_shape)
		.def("move_shape", &DNNConverter::move_shape)
		.def("get_board_array", &DNNConverter::get_board_array)
		.def("get_move_index", &DNNConverter::get_move_index)
		.def("reverse_move_index", &DNNConverter::reverse_move_index)
		;
}
#endif
