#ifdef PYMODULE
#include "pymodule.h"
#include "../engine/user-engine/dnn_converter_py.h"
#include "../engine/user-engine/print_py.h"

PYBIND11_MODULE(neneshogi_cpp, m) {
	m.doc() = "neneshogi native module";
	py::enum_<Color>(m, "Color")
		.value("BLACK", BLACK)
		.value("WHITE", WHITE)
		.value("COLOR_NB", COLOR_NB);
	py::class_<DNNConverterPy>(m, "DNNConverter")
		.def(py::init<int, int>())
		.def("set_packed_sfen", &DNNConverterPy::set_packed_sfen)
		.def("set_sfen", &DNNConverterPy::set_sfen)
		.def("get_sfen", &DNNConverterPy::get_sfen)
		.def("board_shape", &DNNConverterPy::board_shape)
		.def("move_shape", &DNNConverterPy::move_shape)
		.def("get_board_array", &DNNConverterPy::get_board_array)
		.def("get_move_index", &DNNConverterPy::get_move_index)
		.def("reverse_move_index", &DNNConverterPy::reverse_move_index)
		.def("move_from_usi", &DNNConverterPy::move_from_usi)
		.def("move_to_usi", &DNNConverterPy::move_to_usi)
		;
	py::class_<PrintPy>(m, "Print")
		.def_static("move", &PrintPy::move)
		.def_static("piece", &PrintPy::piece)
		.def_static("board", &PrintPy::board)
		;
}

// dummy
void user_test(Position& pos_, istringstream& is)
{
}
void USI::extra_option(USI::OptionsMap & o)
{
}
void Search::init()
{
}
void Search::clear()
{
}
void MainThread::think()
{
}
void Thread::search()
{
}
#endif
