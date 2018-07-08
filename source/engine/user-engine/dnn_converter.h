#include "../../extra/all.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class DNNConverter {
	Position pos;
	StateInfo init_state;
	int format_board, format_move;
public:
	DNNConverter(int format_board, int format_move);
	py::tuple board_shape() const;
	py::tuple move_shape() const;
	bool set_packed_sfen(const char* packed_sfen);
	void set_sfen(std::string sfen);
	py::array_t<float> get_board_array() const;
	int get_move_index(Move move) const;
	Move reverse_move_index(int move_index) const;
private:
	int get_move_index_inner(const Position& pos, Move move) const;
	Move reverse_move_index_inner(const Position& pos, int move_index) const;
};
