// 詰将棋探索、mate-search.cppをMCTS Engineと組み合わせて使われるように改造。

#include "../../shogi.h"

#ifdef USER_ENGINE_MCTS
#ifdef USE_MCTS_MATE_ENGINE

#include "../../extra/all.h"

#include "mate-search_for_mcts.h"

using namespace std;
using namespace Search;

// --- 詰み将棋探索

// df-pn with Threshold Controlling Algorithm (TCA)の実装。
// 岸本章宏氏の "Dealing with infinite loops, underestimation, and overestimation of depth-first
// proof-number search." に含まれる擬似コードを元に実装しています。
//
// TODO(someone): 優越関係の実装
// TODO(someone): 証明駒の実装
// TODO(someone): Source Node Detection Algorithm (SNDA)の実装
// 
// リンク＆参考文献
//
// Ayumu Nagai , Hiroshi Imai , "df-pnアルゴリズムの詰将棋を解くプログラムへの応用",
// 情報処理学会論文誌,43(6),1769-1777 (2002-06-15) , 1882-7764
// http://id.nii.ac.jp/1001/00011597/
//
// Nagai, A.: Df-pn algorithm for searching AND/OR trees and its applications, PhD thesis,
// Department of Information Science, The University of Tokyo (2002)
//
// Ueda T., Hashimoto T., Hashimoto J., Iida H. (2008) Weak Proof-Number Search. In: van den Herik
// H.J., Xu X., Ma Z., Winands M.H.M. (eds) Computers and Games. CG 2008. Lecture Notes in Computer
// Science, vol 5131. Springer, Berlin, Heidelberg
//
// Toru Ueda, Tsuyoshi Hashimoto, Junichi Hashimoto, Hiroyuki Iida, Weak Proof - Number Search,
// Proceedings of the 6th international conference on Computers and Games, p.157 - 168, September 29
// - October 01, 2008, Beijing, China
//
// Kishimoto, A.: Dealing with infinite loops, underestimation, and overestimation of depth-first
// proof-number search. In: Proceedings of the AAAI-10, pp. 108-113 (2010)
//
// A. Kishimoto, M. Winands, M. Müller and J. Saito. Game-Tree Search Using Proof Numbers: The First
// Twenty Years. ICGA Journal 35(3), 131-156, 2012. 
//
// A. Kishimoto and M. Mueller, Tutorial 4: Proof-Number Search Algorithms
// 
// df-pnアルゴリズム学習記(1) - A Succulent Windfall
// http://caprice-j.hatenablog.com/entry/2014/02/14/010932
//
// IS将棋の詰将棋解答プログラムについて
// http://www.is.titech.ac.jp/~kishi/pdf_file/csa.pdf
//
// df-pn探索おさらい - 思うだけで学ばない日記
// http://d.hatena.ne.jp/GMA0BN/20090520/1242825044
//
// df-pn探索のコード - 思うだけで学ばない日記
// http://d.hatena.ne.jp/GMA0BN/20090521/1242911867
//

namespace MateEngine
{
	// 不詰を意味する無限大を意味するPn,Dnの値。
	static const constexpr uint32_t kInfinitePnDn = 100000000;

	// 最大深さ(これだけしかスタックとか確保していない)
	// static const constexpr uint16_t kMaxDepth = MAX_PLY;

	// 正確なPVを返すときのUsiOptionで使うnameの文字列。
	static const constexpr char* kMorePreciseMatePv = "MorePreciseMatePv";


	// TODO(tanuki-): ネガマックス法的な書き方に変更する
	void MateSearchForMCTS::DFPNwithTCA(Position& n, uint32_t thpn, uint32_t thdn, bool inc_flag, bool or_node, uint16_t depth, Color root_color) {
		if (Threads.stop.load(std::memory_order_relaxed)) {
			return;
		}

		//auto nodes_searched = n.this_thread()->nodes.load(memory_order_relaxed);
		//if (nodes_searched && nodes_searched % 10000000 == 0) {
		//	sync_cout << "info string nodes_searched=" << nodes_searched << sync_endl;
		//}

		auto& entry = transposition_table.LookUp(n, root_color);

		if (depth > max_depth) {
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.minimum_distance = std::min(entry.minimum_distance, depth);
			return;
		}

		// if (n is a terminal node) { handle n and return; }

		// 1手読みルーチンによるチェック
		if (or_node && !n.in_check() && n.mate1ply()) {
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.minimum_distance = std::min(entry.minimum_distance, depth);
			return;
		}

		// 千日手のチェック
		// 対局規定（抄録）｜よくある質問｜日本将棋連盟 https://www.shogi.or.jp/faq/taikyoku-kitei.html
		// 第8条 反則
		// 7. 連続王手の千日手とは、同一局面が４回出現した一連の手順中、片方の手が
		// すべて王手だった場合を指し、王手を続けた側がその時点で負けとなる。
		// 従って開始局面により、連続王手の千日手成立局面が王手をかけた状態と
		// 王手を解除した状態の二つのケースがある。 （※）
		// （※）は平成25年10月1日より暫定施行。
		auto draw_type = n.is_repetition(n.game_ply());
		switch (draw_type) {
		case REPETITION_WIN:
			// 連続王手の千日手による勝ち
			if (or_node) {
				// ここは通らないはず
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
				entry.minimum_distance = std::min(entry.minimum_distance, depth);
			}
			else {
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.minimum_distance = std::min(entry.minimum_distance, depth);
			}
			return;

		case REPETITION_LOSE:
			// 連続王手の千日手による負け
			if (or_node) {
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.minimum_distance = std::min(entry.minimum_distance, depth);
			}
			else {
				// ここは通らないはず
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
				entry.minimum_distance = std::min(entry.minimum_distance, depth);
			}
			return;

		case REPETITION_DRAW:
			// 普通の千日手
			// ここは通らないはず
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.minimum_distance = std::min(entry.minimum_distance, depth);
			return;

		default:
			break;
		}

		MovePicker move_picker(n, or_node);
		if (move_picker.empty()) {
			// nが先端ノード

			if (or_node) {
				// 自分の手番でここに到達した場合は王手の手が無かった、
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
			}
			else {
				// 相手の手番でここに到達した場合は王手回避の手が無かった、
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
			}

			entry.minimum_distance = std::min(entry.minimum_distance, depth);
			return;
		}

		// minimum distanceを保存する
		// TODO(nodchip): このタイミングでminimum distanceを保存するのが正しいか確かめる
		entry.minimum_distance = std::min(entry.minimum_distance, depth);

		bool first_time = true;
		while (!Threads.stop.load(std::memory_order_relaxed)) {
			++entry.num_searched;

			// determine whether thpn and thdn are increased.
			// if (n is a leaf) inc flag = false;
			if (entry.pn == 1 && entry.dn == 1) {
				inc_flag = false;
			}

			// if (n has an unproven old child) inc flag = true;
			for (const auto& move : move_picker) {
				// unproven old childの定義はminimum distanceがこのノードよりも小さいノードだと理解しているのだけど、
				// 合っているか自信ない
				const auto& child_entry = transposition_table.LookUpChildEntry(n, move, root_color);
				if (entry.minimum_distance > child_entry.minimum_distance &&
					child_entry.pn != kInfinitePnDn &&
					child_entry.dn != kInfinitePnDn) {
					inc_flag = true;
					break;
				}
			}

			// expand and compute pn(n) and dn(n);
			if (or_node) {
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				for (const auto& move : move_picker) {
					const auto& child_entry = transposition_table.LookUpChildEntry(n, move, root_color);
					entry.pn = std::min(entry.pn, child_entry.pn);
					entry.dn += child_entry.dn;
				}
				entry.dn = std::min(entry.dn, kInfinitePnDn);
			}
			else {
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
				for (const auto& move : move_picker) {
					const auto& child_entry = transposition_table.LookUpChildEntry(n, move, root_color);
					entry.pn += child_entry.pn;
					entry.dn = std::min(entry.dn, child_entry.dn);
				}
				entry.pn = std::min(entry.pn, kInfinitePnDn);
			}

			// if (first time && inc flag) {
			//   // increase thresholds
			//   thpn = max(thpn, pn(n) + 1);
			//   thdn = max(thdn, dn(n) + 1);
			// }
			if (first_time && inc_flag) {
				thpn = std::max(thpn, entry.pn + 1);
				thpn = std::min(thpn, kInfinitePnDn);
				thdn = std::max(thdn, entry.dn + 1);
				thdn = std::min(thdn, kInfinitePnDn);
			}

			// if (pn(n) ≥ thpn || dn(n) ≥ thdn)
			//   break; // termination condition is satisfied
			if (entry.pn >= thpn || entry.dn >= thdn) {
				break;
			}

			// first time = false;
			first_time = false;

			// find the best child n1 and second best child n2;
			// if (n is an OR node) { /* set new thresholds */
			//   thpn child = min(thpn, pn(n2) + 1);
			//   thdn child = thdn - dn(n) + dn(n1);
			// else {
			//   thpn child = thpn - pn(n) + pn(n1);
			//   thdn child = min(thdn, dn(n2) + 1);
			// }
			Move best_move;
			int thpn_child;
			int thdn_child;
			if (or_node) {
				// ORノードでは最も証明数が小さい = 玉の逃げ方の個数が少ない = 詰ましやすいノードを選ぶ
				uint32_t best_pn = kInfinitePnDn;
				uint32_t second_best_pn = kInfinitePnDn;
				uint32_t best_dn = 0;
				uint32_t best_num_search = UINT32_MAX;
				for (const auto& move : move_picker) {
					const auto& child_entry = transposition_table.LookUpChildEntry(n, move, root_color);
					if (child_entry.pn < best_pn ||
						(child_entry.pn == best_pn && best_num_search > child_entry.num_searched)) {
						second_best_pn = best_pn;
						best_pn = child_entry.pn;
						best_dn = child_entry.dn;
						best_move = move;
						best_num_search = child_entry.num_searched;
					}
					else if (child_entry.pn < second_best_pn) {
						second_best_pn = child_entry.pn;
					}
				}

				thpn_child = std::min(thpn, second_best_pn + 1);
				thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);
			}
			else {
				// ANDノードでは最も反証数の小さい = 王手の掛け方の少ない = 不詰みを示しやすいノードを選ぶ
				uint32_t best_dn = kInfinitePnDn;
				uint32_t second_best_dn = kInfinitePnDn;
				uint32_t best_pn = 0;
				uint32_t best_num_search = UINT32_MAX;
				for (const auto& move : move_picker) {
					const auto& child_entry = transposition_table.LookUpChildEntry(n, move, root_color);
					if (child_entry.dn < best_dn ||
						(child_entry.dn == best_dn && best_num_search > child_entry.num_searched)) {
						second_best_dn = best_dn;
						best_dn = child_entry.dn;
						best_pn = child_entry.pn;
						best_move = move;
					}
					else if (child_entry.dn < second_best_dn) {
						second_best_dn = child_entry.dn;
					}
				}

				thpn_child = std::min(thpn - entry.pn + best_pn, kInfinitePnDn);
				thdn_child = std::min(thdn, second_best_dn + 1);
			}

			StateInfo state_info;
			n.do_move(best_move, state_info);
			DFPNwithTCA(n, thpn_child, thdn_child, inc_flag, !or_node, depth + 1, root_color);
			n.undo_move(best_move);
		}
	}

	// 詰み手順を1つ返す
	// 最短の詰み手順である保証はない
	bool MateSearchForMCTS::SearchMatePvFast(bool or_node, Color root_color, Position& pos, std::vector<Move>& moves, std::unordered_set<Key>& visited) {
		// 一度探索したノードを探索しない
		if (visited.find(pos.key()) != visited.end()) {
			return false;
		}
		visited.insert(pos.key());

		MovePicker move_picker(pos, or_node);
		Move mate1ply = pos.mate1ply();
		if (mate1ply || move_picker.empty()) {
			if (mate1ply) {
				moves.push_back(mate1ply);
			}
			//std::ostringstream oss;
			//oss << "info string";
			//for (const auto& move : moves) {
			//  oss << " " << move;
			//}
			//sync_cout << oss.str() << sync_endl;
			//if (mate1ply) {
			//  moves.pop_back();
			//}
			return true;
		}

		const auto& entry = transposition_table.LookUp(pos, root_color);

		for (const auto& move : move_picker) {
			const auto& child_entry = transposition_table.LookUpChildEntry(pos, move, root_color);
			if (child_entry.pn != 0) {
				continue;
			}

			StateInfo state_info;
			pos.do_move(move, state_info);
			moves.push_back(move);
			if (SearchMatePvFast(!or_node, root_color, pos, moves, visited)) {
				pos.undo_move(move);
				return true;
			}
			moves.pop_back();
			pos.undo_move(move);
		}

		return false;
	}


	// 詰み手順を1つ返す
	// df-pn探索ルーチンが探索したノードの中で、攻め側からみて最短、受け側から見て最長の手順を返す
	// SearchMatePvFast()に比べて遅い
	// df-pn探索ルーチンが詰将棋の詰み手順として正規の手順を探索していない場合、
	// このルーチンも正規の詰み手順を返さない
	// (詰み手順は返すが詰将棋の詰み手順として正規のものである保証はない)
	// or_node ORノード=攻め側の手番の場合はtrue、そうでない場合はfalse
	// pos 盤面
	// memo 過去に探索した盤面のキーと探索状況のmap
	// return 詰みまでの手数、詰みの局面は0、ループがある場合はkLoop、不詰みの場合はkNotMated
	int MateSearchForMCTS::SearchMatePvMorePrecise(bool or_node, Color root_color, Position& pos, std::unordered_map<Key, MateState>& memo) {
		// 過去にこのノードを探索していないか調べる
		auto key = pos.key();
		if (memo.find(key) != memo.end()) {
			auto& mate_state = memo[key];
			if (mate_state.num_moves_to_mate == kSearching) {
				// 読み筋がループしている
				return kLoop;
			}
			else if (mate_state.num_moves_to_mate == kNotMate) {
				return kNotMate;
			}
			else {
				return mate_state.num_moves_to_mate;
			}
		}
		auto& mate_state = memo[key];

		auto mate1ply = pos.mate1ply();
		if (or_node && !pos.in_check() && mate1ply) {
			mate_state.num_moves_to_mate = 1;
			mate_state.move_to_mate = mate1ply;

			// 詰みの局面をメモしておく
			StateInfo state_info = {};
			pos.do_move(mate1ply, state_info);
			auto& mate_state_mated = memo[pos.key()];
			mate_state_mated.num_moves_to_mate = 0;
			pos.undo_move(mate1ply);
			return 1;
		}

		MovePicker move_picker(pos, or_node);
		if (move_picker.empty()) {
			if (or_node) {
				// 攻め側にもかかわらず王手が続かなかった
				// dfpnで弾いているため、ここを通ることはないはず
				mate_state.num_moves_to_mate = kNotMate;
				return kNotMate;
			}
			else {
				// 受け側にもかかわらず玉が逃げることができなかった
				// 詰み
				mate_state.num_moves_to_mate = 0;
				return 0;
			}
		}

		auto best_num_moves_to_mate = or_node ? INT_MAX : INT_MIN;
		auto best_move_to_mate = Move::MOVE_NONE;
		const auto& entry = transposition_table.LookUp(pos, root_color);

		for (const auto& move : move_picker) {
			const auto& child_entry = transposition_table.LookUpChildEntry(pos, move, root_color);
			if (child_entry.pn != 0) {
				continue;
			}

			StateInfo state_info;
			pos.do_move(move, state_info);
			int num_moves_to_mate_candidate = SearchMatePvMorePrecise(!or_node, root_color, pos, memo);
			pos.undo_move(move);

			if (num_moves_to_mate_candidate < 0) {
				continue;
			}
			else if (or_node) {
				// ORノード=攻め側の場合は最短手順を選択する
				if (best_num_moves_to_mate > num_moves_to_mate_candidate) {
					best_num_moves_to_mate = num_moves_to_mate_candidate;
					best_move_to_mate = move;
				}
			}
			else {
				// ANDノード=受け側の場合は最長手順を選択する
				if (best_num_moves_to_mate < num_moves_to_mate_candidate) {
					best_num_moves_to_mate = num_moves_to_mate_candidate;
					best_move_to_mate = move;
				}
			}
		}

		if (best_num_moves_to_mate == INT_MAX || best_num_moves_to_mate == INT_MIN) {
			mate_state.num_moves_to_mate = kNotMate;
			return kNotMate;
		}
		else {
			ASSERT_LV3(best_num_moves_to_mate >= 0);
			mate_state.num_moves_to_mate = best_num_moves_to_mate + 1;
			mate_state.move_to_mate = best_move_to_mate;
			return best_num_moves_to_mate + 1;
		}
	}

	void MateSearchForMCTS::get_pv_from_search(Position &pos, std::unordered_map<Key, MateState>& memo, vector<Move> &moves)
	{
		// 局面におけるbestmoveで進め、再帰的に詰みまでの筋を収集する
		auto& mate_state = memo[pos.key()];
		if (mate_state.num_moves_to_mate <= 0)
		{
			return;
		}
		auto move = mate_state.move_to_mate;
		moves.push_back(move);
		StateInfo state_info;
		pos.do_move(move, state_info);
		get_pv_from_search(pos, memo, moves);
		pos.undo_move(move);
	}

	// 詰将棋探索のエントリポイント
	bool MateSearchForMCTS::dfpn(Position& r, std::vector<Move> *moves) {
		if (r.in_check()) {
			return false;
		}

		// キャッシュの世代を進める
		transposition_table.NewSearch();


		Color root_color = r.side_to_move();
		DFPNwithTCA(r, kInfinitePnDn, kInfinitePnDn, false, true, 0, root_color);
		const auto& entry = transposition_table.LookUp(r, root_color);

#if 1
		// SearchMatePvMorePreciseを使う版
		std::unordered_map<Key, MateState> memo;
		if (SearchMatePvMorePrecise(true, root_color, r, memo) > 0)
		{
			get_pv_from_search(r, memo, *moves);
			return true;
		}
		else
		{
			// 詰まない(or ルート局面が詰み)
			return false;
		}
#else
		// SearchMatePvFastを使う版
		// しばしば楽観的過ぎたりmate+2のような表示になったりして何かおかしい
		std::unordered_set<Key> visited;
		SearchMatePvFast(true, root_color, r, *moves, visited);
		return !moves->empty();
#endif
	}

	void MateSearchForMCTS::init(int64_t hash_size_mb, int max_depth) {
		transposition_table.Resize(hash_size_mb);
		this->max_depth = max_depth;
	}
}



#endif
#endif
