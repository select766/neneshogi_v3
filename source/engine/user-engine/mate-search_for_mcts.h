#include "../../shogi.h"
#ifdef USER_ENGINE_MCTS
#ifdef USE_MCTS_MATE_ENGINE

#include <atomic>
#include <unordered_set>
#include "../../position.h"

// --- 詰め探索

namespace MateEngine
{
	// 詰将棋エンジン用のMovePicker
	struct MovePicker
	{
		MovePicker(Position& pos, bool or_node) {
			// たぬき詰めであれば段階的に指し手を生成する必要はない。
			// 自分の手番なら王手の指し手(CHECKS)、
			// 相手の手番ならば回避手(EVASIONS)を生成。
			endMoves = or_node ?
				generateMoves<CHECKS_ALL>(pos, moves) :
				generateMoves<EVASIONS_ALL>(pos, moves);
			endMoves = std::remove_if(moves, endMoves, [&pos](const auto& move) {
				return !pos.legal(move);
			});
		}

		bool empty() {
			return moves == endMoves;
		}

		ExtMove* begin() { return moves; }
		ExtMove* end() { return endMoves; }
		const ExtMove* begin() const { return moves; }
		const ExtMove* end() const { return endMoves; }

	private:
		ExtMove moves[MAX_MOVES], *endMoves = moves;
	};

	// 置換表
	// 通常の探索エンジンとは置換表に保存したい値が異なるため
	// 詰め将棋専用の置換表を用いている
	// ただしSmallTreeGCは実装せず、Stockfishの置換表の実装を真似ている
	struct TranspositionTable {
		static const constexpr uint32_t kInfiniteDepth = 1000000;
		static const constexpr int CacheLineSize = 64;
		struct TTEntry {
			// ハッシュの上位32ビット
			uint32_t hash_high; // 0
								// TTEntryのインスタンスを作成したタイミングで先端ノードを表すよう1で初期化する
			int pn; // 1
			int dn; // 1
			uint32_t generation : 8; // 0
									 // ルートノードからの最短距離
									 // 初期値を∞として全てのノードより最短距離が長いとみなす
			int minimum_distance : 24; // UINT_MAX
									   // TODO(nodchip): 指し手が1手しかない場合の手を追加する
			int num_searched; // 0
		};
		static_assert(sizeof(TTEntry) == 20, "");

		struct Cluster {
			TTEntry entries[3];
			int padding;
		};
		static_assert(sizeof(Cluster) == 64, "");
		static_assert(CacheLineSize % sizeof(Cluster) == 0, "");

		virtual ~TranspositionTable() {
			if (tt_raw) {
				std::free(tt_raw);
				tt_raw = nullptr;
				tt = nullptr;
			}
		}

		TTEntry& LookUp(Key key) {
			auto& entries = tt[key & clusters_mask];
			uint32_t hash_high = key >> 32;
			// 検索条件に合致するエントリを返す
			for (auto& entry : entries.entries) {
				if (entry.hash_high == 0) {
					// 空のエントリが見つかった場合
					entry.hash_high = hash_high;
					entry.pn = 1;
					entry.dn = 1;
					entry.generation = generation;
					entry.minimum_distance = kInfiniteDepth;
					entry.num_searched = 0;
					return entry;
				}

				if (hash_high == entry.hash_high) {
					// keyが合致するエントリを見つけた場合
					entry.generation = generation;
					return entry;
				}
			}

			// 合致するエントリが見つからなかったので
			// 世代が一番古いエントリをつぶす
			TTEntry* best_entry = nullptr;
			uint32_t best_generation = UINT_MAX;
			for (auto& entry : entries.entries) {
				uint32_t temp_generation;
				if (generation < entry.generation) {
					temp_generation = 256 - entry.generation + generation;
				}
				else {
					temp_generation = generation - entry.generation;
				}

				if (best_generation > temp_generation) {
					best_entry = &entry;
					best_generation = temp_generation;
				}
			}
			best_entry->hash_high = hash_high;
			best_entry->pn = 1;
			best_entry->dn = 1;
			best_entry->generation = generation;
			best_entry->minimum_distance = kInfiniteDepth;
			best_entry->num_searched = 0;
			return *best_entry;
		}

		TTEntry& LookUp(Position& n) {
			return LookUp(n.key());
		}

		// moveを指した後の子ノードの置換表エントリを返す
		TTEntry& LookUpChildEntry(Position& n, Move move) {
			return LookUp(n.key_after(move));
		}

		void Resize(int64_t hash_size_mb) {
			if (hash_size_mb == 16) {
				hash_size_mb = 4096;
			}
			int64_t new_num_clusters = 1LL << MSB64((hash_size_mb * 1024 * 1024) / sizeof(Cluster));
			if (new_num_clusters == num_clusters) {
				return;
			}

			num_clusters = new_num_clusters;

			if (tt_raw) {
				std::free(tt_raw);
				tt_raw = nullptr;
				tt = nullptr;
			}

			tt_raw = std::calloc(new_num_clusters * sizeof(Cluster) + CacheLineSize, 1);
			tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
			clusters_mask = num_clusters - 1;
		}

		void NewSearch() {
			generation = (generation + 1) & 0xff;
		}

		int tt_mask = 0;
		void* tt_raw = nullptr;
		Cluster* tt = nullptr;
		int64_t num_clusters = 0;
		int64_t clusters_mask = 0;
		uint32_t generation = 0; // 256で一周する
	};

	class MateSearchForMCTS
	{
		int max_depth;
		TranspositionTable transposition_table;
		void DFPNwithTCA(Position& n, int thpn, int thdn, bool inc_flag, bool or_node, int depth);
		bool dfs(bool or_node, Position& pos, std::vector<Move>& moves, std::unordered_set<Key>& visited);
	public:
		bool dfpn(Position& r, std::vector<Move> *moves);
		void init(int64_t hash_size_mb, int max_depth);
	};
} // end of namespace

#endif
#endif
