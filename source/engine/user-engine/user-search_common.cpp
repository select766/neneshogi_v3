#include "user-search_common.h"

#ifdef USER_ENGINE

//ipqueue<dnn_eval_obj> *eval_queue;
//ipqueue<dnn_result_obj> *result_queue;

//void init_dnn_queues(const string prefix)
//{
//	eval_queue = new ipqueue<dnn_eval_obj>(0, 0, prefix + "_eval", false);
//	result_queue = new ipqueue<dnn_result_obj>(0, 0, prefix + "_result", false);
//}

//void show_error_if_dnn_queue_fail()
//{
//	if (!eval_queue->ok)
//	{
//		sync_cout << "info string eval queue error!" << sync_endl;
//	}
//	if (!result_queue->ok)
//	{
//		sync_cout << "info string result queue error!" << sync_endl;
//	}
//}

//int dnn_get_move_index(const Position & pos, Move m)
//{
//	/*
//	AlphaZeroの論文を参考に作成
//	9x9は移動元。
//	139次元のうち、64次元は「クイーン」の動き(8方向*最大8マス)。
//	2次元は桂馬の動き。66次元は前述の64+2次元と動きは同じで、成る場合。
//	7次元は駒を打つ場合で、この場合の座標は打つ先。
//	クイーンの動きは(筋,段)=(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)
//	* 後手番の際は、盤面・駒の所属を反転して先手番の状態にする。
//	*/
//	Color side_to_move = pos.side_to_move();
//	Square _move_to = move_to(m);
//	if (side_to_move == WHITE) {
//		_move_to = Inv(_move_to);
//	}
//
//	if (is_drop(m))
//	{
//		return (move_dropped_piece(m) - PAWN + 132) * (int)SQ_NB + _move_to;
//	}
//	else
//	{
//		Square _move_from = move_from(m);
//		if (side_to_move == WHITE) {
//			_move_from = Inv(_move_from);
//		}
//
//		int file_diff = file_of(_move_to) - file_of(_move_from);
//		int rank_diff = rank_of(_move_to) - rank_of(_move_from);
//		int ch;
//		if (file_diff == -1 && rank_diff == -2)
//		{
//			ch = 64;
//		}
//		else if (file_diff == 1 && rank_diff == -2)
//		{
//			ch = 65;
//		}
//		else if (file_diff < 0)
//		{
//			if (rank_diff < 0)
//			{
//				ch = -1 + -file_diff;
//			}
//			else if (rank_diff == 0)
//			{
//				ch = 7 + -file_diff;
//			}
//			else
//			{
//				ch = 15 + -file_diff;
//			}
//		}
//		else if (file_diff == 0)
//		{
//			if (rank_diff < 0)
//			{
//				ch = 23 + -rank_diff;
//			}
//			else
//			{
//				ch = 31 + rank_diff;
//			}
//		}
//		else
//		{
//			// fild_diff > 0
//			if (rank_diff < 0)
//			{
//				ch = 39 + file_diff;
//			}
//			else if (rank_diff == 0)
//			{
//				ch = 47 + file_diff;
//			}
//			else
//			{
//				ch = 55 + file_diff;
//			}
//		}
//
//		if (is_promote(m))
//		{
//			ch += 66;
//		}
//
//		return ch * (int)SQ_NB + _move_from;
//	}
//}

#endif
