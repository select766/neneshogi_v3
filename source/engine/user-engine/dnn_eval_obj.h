#pragma once
#include "../../shogi.h"

#ifdef USER_ENGINE_POLICY
class dnn_table_index
{
public:
	int dummy;
};
#endif

#ifdef USER_ENGINE_MCTS
const int MAX_SEARCH_PATH_LENGTH = 64;
class dnn_table_index
{
public:
	int path_length;
	int path_indices[MAX_SEARCH_PATH_LENGTH];//path_indices[path_length-1]は新規末端ノード
	uint16_t path_child_indices[MAX_SEARCH_PATH_LENGTH];//path_child_indices[path_length-1]は無効
};
#endif

class dnn_move_index
{
public:
	uint16_t move;
	uint16_t index;
};

class dnn_move_prob
{
public:
	uint16_t move;
	uint16_t prob_scaled;
};

class dnn_eval_obj
{
public:
	dnn_table_index index;
	float input_array[85 * 9 * 9];//TODO 盤面表現により変わるので最大サイズでとりあえず確保
	uint16_t n_moves;
	dnn_move_index move_indices[MAX_MOVES];
};

class dnn_result_obj
{
public:
	dnn_table_index index;
	int16_t static_value;
	uint16_t n_moves;
	dnn_move_prob move_probs[MAX_MOVES];
};
