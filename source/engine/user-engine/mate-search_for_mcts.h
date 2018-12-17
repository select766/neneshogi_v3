#include "../../shogi.h"
#ifdef USER_ENGINE_MCTS
#ifdef USE_MCTS_MATE_ENGINE

#include <atomic>
#include <unordered_set>
#include "../../position.h"

// --- 詰め探索

namespace MateEngine
{
	// 詰将棋エンジン用のMovePicker(指し手生成器)
	struct MovePicker
	{
		// or_node == trueなら攻め方の手番である。王手となる指し手をすべて生成する。
		// or_node == falseなら受け方の手番である。王手回避の指し手だけをすべて生成する。
		MovePicker(Position& pos, bool or_node) {
			// たぬき詰めであれば段階的に指し手を生成する必要はない。
			// 自分の手番なら王手の指し手(CHECKS)、
			// 相手の手番ならば回避手(EVASIONS)を生成。
			endMoves = or_node ?
				generateMoves<CHECKS_ALL>(pos, moves) :
				generateMoves<EVASIONS_ALL>(pos, moves);

			// 非合法手が交じるとempty()でそのnodeが詰みかどうかが判定できなくなりややこしいので、
			// ここで除外しておく。
			endMoves = std::remove_if(moves, endMoves, [&pos](const auto& move) {
				return !pos.legal(move);
			});
		}

		// 生成された指し手が空であるかの判定
		bool empty() const {
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

		// 無限大を意味する探索深さの定数
		static const constexpr uint16_t kInfiniteDepth = UINT16_MAX;

		// CPUのcache line(1回のメモリアクセスでこのサイズまでCPU cacheに載る)
		static const constexpr int CacheLineSize = 64;

		// 置換表のEntry
		struct TTEntry
		{
			// ハッシュの上位32ビット
			uint32_t hash_high; // 初期値 : 0

			// TTEntryのインスタンスを作成したタイミングで先端ノードを表すよう1で初期化する
			uint32_t pn; // 初期値 : 1
			uint32_t dn; // 初期値 : 1

			// このTTEntryに関して探索したnode数(桁数足りてる？)
			uint32_t num_searched; // 初期値 : 0

			// ルートノードからの最短距離
			// 初期値を∞として全てのノードより最短距離が長いとみなす
			uint16_t minimum_distance; // 初期値 : kInfiniteDepth

			// 置換表世代
			uint16_t generation;

			// TODO(nodchip): 指し手が1手しかない場合の手を追加する

			// このTTEntryを初期化する。
			void init(uint32_t hash_high_, uint16_t generation_)
			{
				hash_high = hash_high_;
				pn = 1;
				dn = 1;
				minimum_distance = kInfiniteDepth;
				num_searched = 0;
				generation = generation_;
			}
		};
		static_assert(sizeof(TTEntry) == 20, "");

		// TTEntryを束ねたもの。
		struct Cluster {
			// TTEntry 20バイト×3 + 4(padding) == 64
			static constexpr int kNumEntries = 3;
			int padding;

			TTEntry entries[kNumEntries];
		};
		// Clusterのサイズは、CacheLineSizeの整数倍であること。
		static_assert((sizeof(Cluster) % CacheLineSize) == 0, "");

		virtual ~TranspositionTable() {
			Release();
		}

		// 指定したKeyのTTEntryを返す。見つからなければ初期化された新規のTTEntryを返す。
		TTEntry& LookUp(Key key, Color root_color) {
			auto& entries = tt[key & clusters_mask];
			uint32_t hash_high = ((key >> 32) & ~1) | root_color;

			// 検索条件に合致するエントリを返す

			for (auto& entry : entries.entries)
				if (hash_high == entry.hash_high && entry.generation == generation)
					return entry;

			// 合致するTTEntryが見つからなかったので空きエントリーを探して返す

			for (auto& entry : entries.entries)
				// 世代が違うので空きとみなせる
				// ※ hash_high == 0を条件にしてしまうと 1/2^32ぐらいの確率でいつまでも書き込めないentryができてしまう。
				if (entry.generation != generation)
				{
					entry.init(hash_high, generation);
					return entry;
				}

			// 空きエントリが見つからなかったので一番不要っぽいentryを潰す。

			// 探索したノード数が一番少ないnodeから優先して潰す。
			TTEntry* best_entry = nullptr;
			uint32_t best_node_searched = UINT32_MAX;

			for (auto& entry : entries.entries)
			{
				if (best_node_searched > entry.num_searched) {
					best_entry = &entry;
					best_node_searched = entry.num_searched;
				}
			}

			best_entry->init(hash_high, generation);
			return *best_entry;
		}

		TTEntry& LookUp(Position& n, Color root_color) {
			return LookUp(n.key(), root_color);
		}

		// moveを指した後の子ノードの置換表エントリを返す
		TTEntry& LookUpChildEntry(Position& n, Move move, Color root_color) {
			return LookUp(n.key_after(move), root_color);
		}

		// 置換表を確保する。
		// 現在のOptions["Hash"]の値だけ確保する。
		void Resize(int64_t hash_size_mb)
		{
			// int64_t hash_size_mb = (int)Options["Hash"];

			// 作成するクラスターの数。2のべき乗にする。
			int64_t new_num_clusters = 1LL << MSB64((hash_size_mb * 1024 * 1024) / sizeof(Cluster));

			// いま確保しているメモリのクラスター数と同じなら変更がないということなので再確保はしない。
			if (new_num_clusters == num_clusters) {
				return;
			}

			num_clusters = new_num_clusters;

			Release();

			// tt_rawをCacheLineSizeにalignしたものがtt。
			tt_raw = std::calloc(new_num_clusters * sizeof(Cluster) + CacheLineSize, 1);
			tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));

			clusters_mask = num_clusters - 1;
		}

		// 置換表のメモリを確保済みであるなら、それを解放する。
		void Release()
		{
			if (tt_raw) {
				std::free(tt_raw);
				tt_raw = nullptr;
				tt = nullptr;
			}
		}

		// "go mate"ごとに呼び出される
		void NewSearch() {
			++generation;
		}

		// HASH使用率を1000分率で返す。
		// TTEntryを先頭から1000個を調べ、使用中の個数を返す。
		int hashfull() const
		{
			// 使用中のTTEntryの数
			int num_used = 0;

			// 使用中であるかをチェックしたTTEntryの数
			int num_checked = 0;

			for (int cluster_index = 0; ; ++cluster_index)
				for (int entry_index = 0; entry_index < Cluster::kNumEntries; ++entry_index)
				{
					auto& entry = tt[cluster_index].entries[entry_index];
					// 世代が同じ時は使用中であるとみなせる。
					if (entry.generation == generation)
						++num_used;

					if (++num_checked == 1000)
						return num_used;
				}
		}

		// 確保した生のメモリ
		void* tt_raw = nullptr;

		// tt_rawをCacheLineSizeでalignした先頭アドレス
		Cluster* tt = nullptr;

		// 確保されたClusterの数
		int64_t num_clusters = 0;

		// tt[key & clusters_mask] のようにして使う。
		// clusters_mask == (num_clusters - 1)
		int64_t clusters_mask = 0;

		// 置換表世代。NewSearch()のごとにインクリメントされる。
		uint16_t generation;
	};

	// 探索中のノードを表す
	static constexpr const int kSearching = -1;
	// 詰み手順にループが含まれることを表す
	static constexpr const int kLoop = -2;
	// 不詰みを表す
	static constexpr const int kNotMate = -3;

	struct MateState {
		int num_moves_to_mate = kSearching;
		Move move_to_mate = Move::MOVE_NONE;
	};

	class MateSearchForMCTS
	{
		int max_depth;
		TranspositionTable transposition_table;
		void DFPNwithTCA(Position& n, uint32_t thpn, uint32_t thdn, bool inc_flag, bool or_node, uint16_t depth, Color root_color);
		bool SearchMatePvFast(bool or_node, Color root_color, Position& pos, std::vector<Move>& moves, std::unordered_set<Key>& visited);
		int SearchMatePvMorePrecise(bool or_node, Color root_color, Position& pos, std::unordered_map<Key, MateState>& memo);
	public:
		bool dfpn(Position& r, std::vector<Move> *moves);
		void init(int64_t hash_size_mb, int max_depth);
	};
} // end of namespace

#endif
#endif
