#ifdef PYMODULE
#include "dnn_converter_py.h"
static bool pymodule_initialized = false;
DNNConverterPy::DNNConverterPy(int format_board, int format_move) : cvt(format_board, format_move)
{
	if (!pymodule_initialized) {
		// main�֐��Ɠ����̏�����������
		Bitboards::init();
		Position::init();
		Search::init();
		Threads.set(1);
		Eval::init();
		pymodule_initialized = true;
	}
}

py::tuple DNNConverterPy::board_shape() const
{
	auto bs = cvt.board_shape();
	return py::make_tuple(bs[0], bs[1], bs[2]);
}

py::tuple DNNConverterPy::move_shape() const
{
	auto ms = cvt.move_shape();
	return py::make_tuple(ms[0], ms[1], ms[2]);
}


bool DNNConverterPy::set_packed_sfen(const char* packed_sfen)
{
	const PackedSfen* sfen = reinterpret_cast<const PackedSfen*>(packed_sfen);
	return pos.set_from_packed_sfen(*sfen, &init_state, Threads.main()) == 0;
	// gamePly = 0�ƂȂ�̂Œ���
}

void DNNConverterPy::set_sfen(std::string sfen)
{
	pos.set(sfen, &init_state, Threads.main());
}

py::array_t<float> DNNConverterPy::get_board_array() const {
	auto bs = cvt.board_shape();
	float *buf = new float[bs[0] * bs[1] * bs[2]]();
	cvt.get_board_array(pos, buf);
	auto ary = py::array_t<float>(
		py::buffer_info(
			buf,
			sizeof(float),
			py::format_descriptor<float>::format(),
			3,
			bs,
			{ sizeof(float) * bs[1] * bs[2],  sizeof(float) * bs[2],  sizeof(float) }
		)
		);
	delete[] buf;
	return ary;
}
int DNNConverterPy::get_move_index(Move move) const
{
	return cvt.get_move_index(pos, move);
}
Move DNNConverterPy::reverse_move_index(int move_index) const
{
	return cvt.reverse_move_index(pos, move_index);
}
#endif
