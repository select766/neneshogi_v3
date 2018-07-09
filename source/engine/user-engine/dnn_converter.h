#include "../../extra/all.h"

class DNNConverter {
	int format_board, format_move;
public:
	DNNConverter(int format_board, int format_move);
	vector<int> board_shape() const;
	vector<int> move_shape() const;
	void DNNConverter::get_board_array(const Position & pos, float *buf) const;
	int get_move_index(const Position& pos, Move move) const;
	Move reverse_move_index(const Position& pos, int move_index) const;
};
