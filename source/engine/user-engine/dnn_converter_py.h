#pragma once
#ifdef PYMODULE
#include "../../extra/all.h"
#include "dnn_converter.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class DNNConverterPy {
	DNNConverter cvt;
	StateInfo init_state;
public:
	Position pos;
	DNNConverterPy(int format_board, int format_move);
	// C4316
	void *operator new(size_t size) {
		return _mm_malloc(size, alignof(DNNConverterPy));
	}

	void operator delete(void *p) {
		return _mm_free(p);
	}
	py::tuple board_shape() const;
	py::tuple move_shape() const;
	bool set_packed_sfen(const char* packed_sfen);
	void set_sfen(std::string sfen);
	std::string get_sfen() const;
	py::array_t<float> get_board_array() const;
	int get_move_index(Move move) const;
	Move reverse_move_index(int move_index) const;
	Move move_from_usi(const std::string move_usi);
	std::string move_to_usi(Move move) const;
};
#endif
