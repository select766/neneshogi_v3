#pragma once
#include "../../extra/all.h"

class DNNConverter {
	int format_board, format_move;
	int get_move_index_0(const Position& pos, Move move) const;
	int get_move_index_1(const Position& pos, Move move) const;
	Move reverse_move_index_0(const Position& pos, int move_index) const;
	Move reverse_move_index_1(const Position& pos, int move_index) const;
public:
	DNNConverter(int format_board, int format_move);
	vector<int> board_shape() const;
	vector<int> move_shape() const;
	void DNNConverter::get_board_array(const Position & pos, float *buf) const;
	int get_move_index(const Position& pos, Move move) const;
	Move reverse_move_index(const Position& pos, int move_index) const;
};
