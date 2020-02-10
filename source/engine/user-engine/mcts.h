
#include <mutex>
#include "../../extra/all.h"
#include "dnn_eval_obj.h"
#include "dnn_converter.h"
#include "mt_queue.h"
#include "mate-search_for_mcts.h"

// 1ノードに対する探索回数(value_n_sum)の上限値。
// 勝敗をfloatに蓄積するので、16777216を超えるとインクリメントしても増えなくなる問題が生じ、
// むしろ探索結果が悪化してしまう。
#define NODES_LIMIT_MAX 10000000

class NodeHashEntry
{
public:
	Key key;
	int game_ply;
	bool flag;
};

class DupEvalChain
{
public:
	dnn_table_index path;
	DupEvalChain *next;
};

class UCTNode
{
public:
	float value_n_sum;
	bool terminal;
	bool evaled;
	DupEvalChain *dup_eval_chain;//複数回評価が呼ばれたとき、ここにリストをつなげて各経路でbackupする。
	float score;
	int n_children;
	Move move_list[MAX_UCT_CHILDREN];
	float value_n[MAX_UCT_CHILDREN];
	float value_w[MAX_UCT_CHILDREN];
	float value_p[MAX_UCT_CHILDREN];

	void pprint();
};

// MCTS用置換表
class MCTSTT
{
public:
	MCTSTT(size_t uct_hash_size);
	~MCTSTT();
	void clear();
	UCTNode* find_or_create_entry(const Position &pos, bool &created);
	UCTNode* find_or_create_entry(Key key, int game_ply, bool &created);
	UCTNode* find_entry(const Position &pos);
	UCTNode* find_entry(Key key, int game_ply);
	// ハッシュの使用率を千分率で返す
	int get_hashfull() const;
	// max_size_mbで与えた上限を超えない範囲で、2のべき乗のハッシュサイズを決定する。
	static size_t calc_uct_hash_size(int max_size_mb);

private:
	size_t _uct_hash_size;
	size_t _uct_hash_mask;
	// 使用中のエントリ数
	size_t _used;
	//この値未満の手数のエントリーはもう使われないとみなし、新規ノード作成時に上書きできる
	int _obsolete_game_ply;
	NodeHashEntry *entries;
	UCTNode *nodes;
};

class MCTSSearchInfo
{
public:
	// 置換表のロック中か
	bool has_tt_lock;
	// DNN評価をリクエストしたかどうか
	bool put_dnn_eval;
	// 探索結果が現在DNN評価途中の局面だったかどうか
	bool leaf_dup;
	// 末端での詰み探索の結果、詰みだった（局面そのものが詰んでいるのとは異なる）
	bool leaf_mate_search_found;
	DNNConverter *cvt;
	MTQueue<dnn_eval_obj*> *request_queue;
	MTQueue<dnn_eval_obj*> *response_queue;
	MateEngine::MateSearchForMCTS *mate_searcher;

	MCTSSearchInfo(DNNConverter *cvt, MTQueue<dnn_eval_obj*> *request_queue, MTQueue<dnn_eval_obj*> *response_queue, MateEngine::MateSearchForMCTS *mate_searcher)
		: cvt(cvt), request_queue(request_queue), response_queue(response_queue), has_tt_lock(false), put_dnn_eval(false), leaf_dup(false), mate_searcher(mate_searcher)
	{
	}
};

// MCTSの実装
class MCTS
{
public:
	MCTS(size_t uct_hash_size);
	~MCTS();
	void search(UCTNode *root, Position &pos, MCTSSearchInfo &sei, dnn_eval_obj *eval_info);
	// DNNの結果が得られた際のbackup処理
	void backup_dnn(dnn_eval_obj *eval_info, bool do_backup=true);
	UCTNode* make_root(Position &pos, MCTSSearchInfo &sei, dnn_eval_obj *eval_info, bool &created);
	UCTNode * make_root_with_children(Position & pos, MCTSSearchInfo & sei, int &n_put, int max_put);
	UCTNode* get_root(const Position &pos);
	Move get_bestmove(UCTNode *root, Position &pos, bool policy_only=false);
	void get_pv(UCTNode *root, Position &pos, std::vector<Move> &pv, float &winrate);
	// ハッシュの使用率を千分率で返す
	int get_hashfull();
	// 初期化(コンストラクタ直後に呼ぶ必要はない)
	void clear();

	float c_puct;
	float virtual_loss;
private:
	void search_recursive(UCTNode *root, Position &pos, MCTSSearchInfo &sei, dnn_eval_obj *eval_info);
	// treeのbackup操作。
	void backup_tree(dnn_table_index &path, float leaf_score);
	// 末端ノードが評価不要ノードだった場合
	void update_on_terminal(dnn_table_index &path, float leaf_score);
	// 新規展開ノードがmateだったときの処理
	void update_on_mate(dnn_table_index &path, float mate_score);
	// UCBに従い次に探索する子ノードのインデックスを選択する
	size_t select_edge(UCTNode *node);
	bool enqueue_pos(const Position &pos, MCTSSearchInfo &sei, dnn_eval_obj *eval_info, float &score);
	void get_pv_recursive(UCTNode *node, Position &pos, std::vector<Move> &pv, float &winrate, bool root);
	void make_root_with_children_recursive(int depth, Position & pos, MCTSSearchInfo & sei, int &n_put, int max_put);

	std::mutex mutex_;//置換表のロック
	MCTSTT* tt;//置換表(MCTSオブジェクトと1対1対応)
};
