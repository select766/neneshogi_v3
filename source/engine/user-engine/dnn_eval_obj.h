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
#include "mt_queue.h"
const int MAX_SEARCH_PATH_LENGTH = 64;

class UCTNode;

class dnn_table_index
{
public:
	int path_length;
	UCTNode* path_indices[MAX_SEARCH_PATH_LENGTH];//path_indices[path_length-1]は新規末端ノード
	uint16_t path_child_indices[MAX_SEARCH_PATH_LENGTH];//path_child_indices[path_length-1]は無効
};
#endif

class dnn_move_index
{
public:
	uint16_t move;
	uint16_t index;
	float prob;
};

class dnn_eval_obj
{
public:
	dnn_table_index index;
	float input_array[119 * 9 * 9];//TODO 盤面表現により変わるので最大サイズでとりあえず確保
	uint16_t n_moves;
	dnn_move_index move_indices[MAX_MOVES];
	float static_value;//局面の静的評価値(-1~1)
	MTQueue<dnn_eval_obj*> *response_queue;//評価完了時にこのオブジェクトのポインタをputするキュー
	bool found_mate;
};
