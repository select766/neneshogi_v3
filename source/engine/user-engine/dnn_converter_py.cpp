#ifdef PYMODULE
#include "dnn_converter_py.h"
static bool pymodule_initialized = false;
DNNConverterPy::DNNConverterPy(int format_board, int format_move) : cvt(format_board, format_move)
{
	if (!pymodule_initialized) {
		// mainŠÖ”‚Æ“¯“™‚Ì‰Šú‰»‚ğ‚·‚é
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
	return py::make_tuple(85, 9, 9);
}

py::tuple DNNConverterPy::move_shape() const
{
	return py::make_tuple(139, 9, 9);
}


bool DNNConverterPy::set_packed_sfen(const char* packed_sfen)
{
	const PackedSfen* sfen = reinterpret_cast<const PackedSfen*>(packed_sfen);
	return pos.set_from_packed_sfen(*sfen, &init_state, Threads.main()) == 0;
	// gamePly = 0‚Æ‚È‚é‚Ì‚Å’ˆÓ
}

void DNNConverterPy::set_sfen(std::string sfen)
{
	pos.set(sfen, &init_state, Threads.main());
}

py::array_t<float> DNNConverterPy::get_board_array() const {
	float buf[85 * 9 * 9] = {};
	cvt.get_board_array(pos, buf);
	return py::array_t<float>(
		py::buffer_info(
			buf,
			sizeof(float),
			py::format_descriptor<float>::format(),
			3,
			{ 85, 9, 9 },
			{ sizeof(float) * 9 * 9,  sizeof(float) * 9,  sizeof(float) }
		)
		);
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
