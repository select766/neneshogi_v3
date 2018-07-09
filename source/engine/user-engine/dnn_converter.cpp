#include "dnn_converter.h"

DNNConverter::DNNConverter(int _format_board, int _format_move) : format_board(format_board), format_move(format_move)
{
}

vector<int> DNNConverter::board_shape() const
{
	return vector<int>{85, 9, 9};
}

vector<int> DNNConverter::move_shape() const
{
	return vector<int>{139, 9, 9};
}


static void fill_channel(float* buf, int ch, float value)
{
	for (Square i = SQ_ZERO; i < SQ_NB; i++) {
		buf[ch * SQ_NB + i] = value;
	}
}

static void fill_channel_range(float* buf, int ch_begin, int ch_end, float value)
{
	while (ch_begin < ch_end)
	{
		fill_channel(buf, ch_begin++, value);
	}
}

void DNNConverter::get_board_array(const Position & pos, float *buf) const
{	/*
	* Ponanza (SDT5)の資料を参考に作成
	* 盤上の駒14チャンネル *二人
	* 持ち駒は、枚数分のチャンネル(金なら4)を用意して1で埋めていく(歩は最大8枚)
	* 歩*8,香車*4,桂馬*4,銀*4,角*2,飛車*2,金*4=28*二人
	* 王手かどうか 1次元
	* 後手番の際は、盤面・駒の所属を反転して先手番の状態にする。
	* 手数は現在入れていない。Position.set_from_packed_sfenに要素がないため。
	*/
	if (pos.side_to_move() == BLACK) {
		for (Square i = SQ_ZERO; i < SQ_NB; i++) {
			Piece p = pos.piece_on(i);
			if (p == PIECE_ZERO) {
				continue;
			}
			int ch;
			if (color_of(p) == BLACK) {
				ch = p - B_PAWN;
			}
			else {
				ch = p - W_PAWN + 14;
			}
			buf[ch * SQ_NB + i] = 1;
		}
	}
	else {
		for (Square i = SQ_ZERO; i < SQ_NB; i++) {
			Piece p = pos.piece_on(i);
			if (p == PIECE_ZERO) {
				continue;
			}
			int ch;
			// 先手後手入れ替え+座標回転
			if (color_of(p) == BLACK) {
				ch = p - B_PAWN + 14;
			}
			else {
				ch = p - W_PAWN;
			}
			buf[ch * SQ_NB + Inv(i)] = 1;
		}

	}

	int ch_ofs = 28;
	Hand hands[2] = { pos.hand_of(pos.side_to_move()), pos.hand_of(~pos.side_to_move()) };
	for (int i = 0; i < 2; i++) {
		Hand hand = hands[i];
		//歩は最大8枚
		fill_channel_range(buf, ch_ofs, (std::min)(ch_ofs + hand_count(hand, PAWN), 8), 1.0);
		ch_ofs += 8;
		fill_channel_range(buf, ch_ofs, ch_ofs + hand_count(hand, LANCE), 1.0);
		ch_ofs += 4;
		fill_channel_range(buf, ch_ofs, ch_ofs + hand_count(hand, KNIGHT), 1.0);
		ch_ofs += 4;
		fill_channel_range(buf, ch_ofs, ch_ofs + hand_count(hand, SILVER), 1.0);
		ch_ofs += 4;
		fill_channel_range(buf, ch_ofs, ch_ofs + hand_count(hand, BISHOP), 1.0);
		ch_ofs += 2;
		fill_channel_range(buf, ch_ofs, ch_ofs + hand_count(hand, ROOK), 1.0);
		ch_ofs += 2;
		fill_channel_range(buf, ch_ofs, ch_ofs + hand_count(hand, GOLD), 1.0);
		ch_ofs += 4;
	}

	fill_channel(buf, 84, (float)pos.in_check());
}


int DNNConverter::get_move_index(const Position & pos, Move move) const
{	/*
	AlphaZeroの論文を参考に作成
	9x9は移動元。
	139次元のうち、64次元は「クイーン」の動き(8方向*最大8マス)。
	2次元は桂馬の動き。66次元は前述の64+2次元と動きは同じで、成る場合。
	7次元は駒を打つ場合で、この場合の座標は打つ先。
	クイーンの動きは(筋,段)=(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)
	* 後手番の際は、盤面・駒の所属を反転して先手番の状態にする。
	*/
	Move &m = move;
	Color side_to_move = pos.side_to_move();
	Square _move_to = move_to(m);
	if (side_to_move == WHITE) {
		_move_to = Inv(_move_to);
	}

	if (is_drop(m))
	{
		return (move_dropped_piece(m) - PAWN + 132) * (int)SQ_NB + _move_to;
	}
	else
	{
		Square _move_from = move_from(m);
		if (side_to_move == WHITE) {
			_move_from = Inv(_move_from);
		}

		int file_diff = file_of(_move_to) - file_of(_move_from);
		int rank_diff = rank_of(_move_to) - rank_of(_move_from);
		int ch;
		if (file_diff == -1 && rank_diff == -2)
		{
			ch = 64;
		}
		else if (file_diff == 1 && rank_diff == -2)
		{
			ch = 65;
		}
		else if (file_diff < 0)
		{
			if (rank_diff < 0)
			{
				ch = -1 + -file_diff;
			}
			else if (rank_diff == 0)
			{
				ch = 7 + -file_diff;
			}
			else
			{
				ch = 15 + -file_diff;
			}
		}
		else if (file_diff == 0)
		{
			if (rank_diff < 0)
			{
				ch = 23 + -rank_diff;
			}
			else
			{
				ch = 31 + rank_diff;
			}
		}
		else
		{
			// fild_diff > 0
			if (rank_diff < 0)
			{
				ch = 39 + file_diff;
			}
			else if (rank_diff == 0)
			{
				ch = 47 + file_diff;
			}
			else
			{
				ch = 55 + file_diff;
			}
		}

		if (is_promote(m))
		{
			ch += 66;
		}

		return ch * (int)SQ_NB + _move_from;
	}
}

Move DNNConverter::reverse_move_index(const Position & pos, int move_index) const
{
	int ch = move_index / (int)SQ_NB;
	Square _move_from = (Square)(move_index % (int)SQ_NB);
	Color side_to_move = pos.side_to_move();
	if (ch >= 132)
	{
		// drop
		Piece pt = (Piece)(ch - 132 + 1);
		if (side_to_move == WHITE) {
			_move_from = Inv(_move_from);
		}
		return make_move_drop(pt, _move_from);
	}
	else
	{
		// move
		bool is_promote = ch >= 66;
		if (is_promote)
		{
			ch -= 66;
		}

		int from_file = file_of(_move_from);
		int from_rank = rank_of(_move_from);

		int to_file, to_rank;
		if (ch == 64)
		{
			to_file = from_file - 1;
			to_rank = from_rank - 2;
		}
		else if (ch == 65)
		{
			to_file = from_file + 1;
			to_rank = from_rank - 2;
		}
		else
		{
			int dirs[][2] = { { -1,-1 },{ -1,0 },{ -1,1 },{ 0,-1 },{ 0,1 },{ 1,-1 },{ 1,0 },{ 1,1 } };
			int dir_index = ch / 8;
			int move_length = ch % 8 + 1;
			to_file = from_file + dirs[dir_index][0] * move_length;
			to_rank = from_rank + dirs[dir_index][1] * move_length;
		}
		Square _move_to = (File)to_file | (Rank)to_rank;
		if (side_to_move == WHITE) {
			_move_from = Inv(_move_from);
			_move_to = Inv(_move_to);
		}

		if (is_promote)
		{
			return make_move_promote(_move_from, _move_to);
		}
		else
		{
			return make_move(_move_from, _move_to);
		}
	}
}
