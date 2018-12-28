
#include <mutex>
#include "../../extra/all.h"
#include "dnn_eval_obj.h"
#include "dnn_converter.h"
#include "mt_queue.h"

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
	int value_n_sum;
	bool terminal;
	bool evaled;
	DupEvalChain *dup_eval_chain;//複数回評価が呼ばれたとき、ここにリストをつなげて各経路でbackupする。
	float score;
	int n_children;
	Move move_list[MAX_UCT_CHILDREN];
	int value_n[MAX_UCT_CHILDREN];
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
	DNNConverter *cvt;
	MTQueue<dnn_eval_obj*> *request_queue;
	MTQueue<dnn_eval_obj*> *response_queue;

	MCTSSearchInfo(DNNConverter *cvt, MTQueue<dnn_eval_obj*> *request_queue, MTQueue<dnn_eval_obj*> *response_queue)
		: cvt(cvt), request_queue(request_queue), response_queue(response_queue), has_tt_lock(false), put_dnn_eval(false), leaf_dup(false)
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
	void backup_dnn(dnn_eval_obj *eval_info);
	UCTNode* make_root(Position &pos, MCTSSearchInfo &sei, dnn_eval_obj *eval_info, bool &created);
	UCTNode* get_root(const Position &pos);
	Move get_bestmove(UCTNode *root, Position &pos);

	float c_puct;
	int virtual_loss;
private:
	void search_recursive(UCTNode *root, Position &pos, MCTSSearchInfo &sei, dnn_eval_obj *eval_info);
	// treeのbackup操作。
	void backup_tree(dnn_table_index &path, float leaf_score);
	// 末端ノードが評価不要ノードだった場合
	void update_on_terminal(dnn_table_index &path, float leaf_score);
	// 新規展開ノードがmateだったときの処理
	void update_on_mate(dnn_table_index &path, float mate_score);
	// UCBに従い次に探索する子ノードのインデックスを選択する
	int select_edge(UCTNode *node);
	bool enqueue_pos(const Position &pos, MCTSSearchInfo &sei, dnn_eval_obj *eval_info, float &score, bool use_mate_search);

	std::mutex mutex_;//置換表のロック
	MCTSTT* tt;//置換表(MCTSオブジェクトと1対1対応)
};
