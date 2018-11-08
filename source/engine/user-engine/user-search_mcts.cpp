#include "../../extra/all.h"
#include <thread>
#include <chrono>
#include <numeric>
#include <functional>
#include "CNTKLibrary.h"
#include "mate-search_for_mcts.h"
#include "dnn_converter.h"
#include "ipqueue.h"
#include "dnn_eval_obj.h"
#include "dnn_thread.h"

#ifdef USER_ENGINE_MCTS

#define MAX_UCT_CHILDREN 16//UCTノードの子ノード数最大
static int obsolete_removed_count = 0;
ipqueue<dnn_eval_obj> *eval_queue;
ipqueue<dnn_result_obj> *result_queue;

class TreeConfig
{
public:
	float c_puct;
	float play_temperature;
	int virtual_loss;
	float value_scale;
	bool clear_table_before_search;
};

namespace MCTSAsync
{
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

	class UctNode
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
		//float value_q[MAX_UCT_CHILDREN];
		//int vloss_ctr[MAX_UCT_CHILDREN];//virtual lossがちゃんと復帰されたか確認用
	};

	class NodeHash
	{
	public:
		int uct_hash_size;
		int uct_hash_limit;
		unsigned int uct_hash_mask;
		int used;
		int obsolete_game_ply;
		bool enough_size;
		NodeHashEntry *entries;
		UctNode *nodes;

		NodeHash(int uct_hash_size) :uct_hash_size(uct_hash_size), used(0), enough_size(true), obsolete_game_ply(0)
		{
			uct_hash_limit = (int)(uct_hash_size * 0.9);
			uct_hash_mask = (unsigned int)uct_hash_size - 1;
			entries = new NodeHashEntry[uct_hash_size]();
			nodes = new UctNode[uct_hash_size]();
		}

		void clear()
		{
			used = 0;
			enough_size = true;
			memset(entries, 0, sizeof(NodeHashEntry)*uct_hash_size);
			memset(nodes, 0, sizeof(UctNode)*uct_hash_size);
		}

		int find_or_create_index(const Position &pos, bool *created)
		{
			return find_or_create_index(pos.key(), pos.game_ply(), created);
		}

		int find_or_create_index(Key key, int game_ply, bool *created)
		{
			int orig_index = (unsigned int)key & uct_hash_mask;
			int index = orig_index;
			while (true)
			{
				NodeHashEntry *nhe = &entries[index];
				if (nhe->flag)
				{
					if (nhe->game_ply < obsolete_game_ply)
					{
						// ここを上書きする
						nhe->key = key;
						nhe->game_ply = game_ply;

						*created = true;
						memset(&nodes[index], 0, sizeof(UctNode));
						obsolete_removed_count++;
						return index;
					}
					if (nhe->key == key && nhe->game_ply == game_ply)
					{
						*created = false;
						return index;
					}

				}
				else
				{
					nhe->key = key;
					nhe->game_ply = game_ply;
					nhe->flag = true;
					used++;
					if (used > uct_hash_limit)
					{
						enough_size = false;
					}
					*created = true;
					return index;
				}
				index = (index + 1) & uct_hash_mask;
				if (index == orig_index)
				{
					// full
					return -1;
				}
			}
		}

		int find_index(const Position &pos)
		{
			return find_index(pos.key(), pos.game_ply());
		}

		int find_index(Key key, int game_ply)
		{
			int orig_index = (unsigned int)key & uct_hash_mask;
			int index = orig_index;
			while (true)
			{
				NodeHashEntry *nhe = &entries[index];
				if (nhe->flag)
				{
					if (nhe->key == key && nhe->game_ply == game_ply)
					{
						return index;
					}

				}
				else
				{
					return -1;
				}
				index = (index + 1) & uct_hash_mask;
				if (index == orig_index)
				{
					// full
					return -1;
				}
			}
		}

		~NodeHash()
		{
			delete[] entries;
			delete[] nodes;
		}

		bool save(const string& path, int min_visit)
		{
			ofstream f;
			f.open(path.c_str(), ios::out | ios::binary | ios::trunc);
			if (!f)
			{
				return false;
			}

			for (int i = 0; i < uct_hash_size; i++)
			{
				NodeHashEntry *nhe = &entries[i];
				if (!nhe->flag)
				{
					continue;
				}
				UctNode *node = &nodes[i];
				if (node->value_n_sum < min_visit)
				{
					continue;
				}

				f.write(reinterpret_cast<char*>(nhe), sizeof(NodeHashEntry));
				f.write(reinterpret_cast<char*>(node), sizeof(UctNode));
				if (!f)
				{
					return false;
				}
			}

			f.close();

			if (!f)
			{
				return false;
			}
			return true;
		}

		bool load(const string& path)
		{
			ifstream f;
			f.open(path.c_str(), ios::in | ios::binary);
			if (!f)
			{
				return false;
			}

			bool ok = true;

			while (!f.eof())
			{
				NodeHashEntry f_nhe;
				UctNode f_node;
				f.read(reinterpret_cast<char*>(&f_nhe), sizeof(NodeHashEntry));
				f.read(reinterpret_cast<char*>(&f_node), sizeof(UctNode));
				bool created;
				int index = find_or_create_index(f_nhe.key, f_nhe.game_ply, &created);
				if (index < 0)
				{
					ok = false;
					break;
				}

				NodeHashEntry *t_nhe = &entries[index];
				UctNode *t_node = &nodes[index];
				if (!created)
				{
					// 既存データより訪問回数が多いときだけ上書き
					if (t_node->value_n_sum >= f_node.value_n_sum)
					{
						continue;
					}
				}

				memcpy(t_node, &f_node, sizeof(UctNode));
			}

			f.close();

			return true;
		}
	};

}

using namespace MCTSAsync;

static NodeHash *node_hash = nullptr;
static int eval_queue_batch_index = 0;
static ipqueue_item<dnn_eval_obj> *eval_objs = nullptr;
static int n_batch_put = 0;
static int n_batch_get = 0;
static TreeConfig tree_config;
static int max_select = 1000;
static int pv_interval = 1000;
static int eval_count_this_search = 0;// 探索開始から評価したノード数。nps表示用。詰みに達した場合等は加算しない。
static int special_terminal_count_this_search = 0;// 探索開始から、評価関数呼び出し以外の終端ノードに到達した回数。
static bool dnn_initialized = false;
static int block_queue_length = 2;
//TODO mate_searchを利用可能にする
//static MateEngine::MateSearchForMCTS *mate_search_root = nullptr;
//static MateEngine::MateSearchForMCTS *mate_search_leaf = nullptr;
static int mate_search_leaf_count = 0;//末端ノードの詰み探索で詰みと判定された回数
static vector<Move> root_mate_pv;
static atomic<bool> root_mate_found;
static vector<std::thread*> dnn_threads;

// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos_, istringstream& is)
{
	string token;
	is >> token;
	if (token == "save_tt")
	{
		int min_visit;
		string path;
		is >> min_visit;
		is >> path;
		if (node_hash->save(path, min_visit))
		{
			sync_cout << "ok" << sync_endl;
		}
		else
		{
			sync_cout << "failed" << sync_endl;
		}
	}
	else if (token == "load_tt")
	{
		string path;
		is >> path;
		if (node_hash->load(path))
		{
			int hashfull = (int)((long long)node_hash->used * 1000 / node_hash->uct_hash_size);
			sync_cout << "ok hashfull " << hashfull << sync_endl;
		}
		else
		{
			sync_cout << "failed" << sync_endl;
		}

	}
}

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	o["GPU"] << Option("-1");//使用するGPU番号(-1==CPU)、カンマ区切りで複数指定可能
	o["format_board"] << Option(0, 0, 16);//DNNのboard表現形式
	o["format_move"] << Option(0, 0, 16);//DNNのmove表現形式
	o["PvInterval"] << Option(300, 0, 100000);//PV出力する間隔[ms]
	o["MCTSHash"] << Option(1, 1, 1024);//MCTSのハッシュテーブルのサイズ[GB]上限。
	o["c_puct"] << Option("1.0");
	o["play_temperature"] << Option("1.0");
	o["softmax"] << Option("1.0");
	o["value_scale"] << Option("1.0");
	o["value_slope"] << Option("1.0");
	o["virtual_loss"] << Option(1, 0, 100);
	o["clear_table"] << Option(false);
	o["batch_size"] << Option(16, 1, 65536);
	o["block_queue_length"] << Option(2, 1, 64);
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
	if (node_hash)
	{
		delete node_hash;
		node_hash = nullptr;
	}
	max_select = (int)Options["NodesLimit"];
	pv_interval = (int)Options["PvInterval"];
	block_queue_length = (int)Options["block_queue_length"];
	int batch_size = (int)Options["batch_size"];
	unsigned long long hash_max_gb = (unsigned int)Options["MCTSHash"];//このサイズ以下で、2^n要素数を選択
	unsigned long long hash_max_bytes = hash_max_gb * (1024 * 1024 * 1024);
	unsigned long long node_hash_size = 1;
	while (node_hash_size * (sizeof(NodeHashEntry) + sizeof(UctNode)) <= hash_max_bytes)
	{
		node_hash_size <<= 1;
	}
	node_hash_size >>= 1;

	sync_cout << "info string node hash " << (node_hash_size / (1024 * 1024)) << "M elements (Max " << MAX_UCT_CHILDREN << "moves / node)" << sync_endl;

	auto hash_init_thread = std::thread([node_hash_size] {
		node_hash = new NodeHash((int)node_hash_size);
		sync_cout << "info string node hash init completed" << sync_endl;
	});
	tree_config.c_puct = (float)atof(((string)Options["c_puct"]).c_str());
	tree_config.play_temperature = (float)atof(((string)Options["play_temperature"]).c_str());
	tree_config.virtual_loss = (int)Options["virtual_loss"];
	tree_config.value_scale = (float)atof(((string)Options["value_scale"]).c_str());
	tree_config.clear_table_before_search = (bool)Options["clear_table"];

#ifdef USE_MCTS_MATE_ENGINE
	if (mate_search_root == nullptr)
	{
		mate_search_root = new MateEngine::MateSearchForMCTS();
		mate_search_root->init(1024, MAX_PLY);
	}
	if (mate_search_leaf == nullptr)
	{
		mate_search_leaf = new MateEngine::MateSearchForMCTS();
		mate_search_leaf->init(1024, 3);
	}
#endif

	if (!dnn_initialized)
	{
		// モデルのロード
		// 本来はファイル名からフォーマットを推論したい
		// 将棋所からは日本語WindowsだとオプションがCP932で来る。mbstowcsにそれを認識させ、日本語ファイル名を正しく変換
		setlocale(LC_ALL, "");
		int format_board = (int)Options["format_board"], format_move = (int)Options["format_move"];
		wchar_t model_path[1024];
		string evaldir = Options["EvalDir"];
		wchar_t evaldir_w[1024];
		mbstowcs(evaldir_w, evaldir.c_str(), sizeof(model_path) / sizeof(model_path[0]) - 1); // C4996
		swprintf(model_path, sizeof(model_path) / sizeof(model_path[0]), L"%s/nene_%d_%d.cmf", evaldir_w, format_board, format_move);
		// デバイス数だけモデルをロードし各デバイスに割り当てる
		stringstream ss(Options["GPU"]);//カンマ区切りでGPU番号を並べる
		string item;
		while (getline(ss, item, ',')) {
			if (!item.empty()) {
				int gpu_id = stoi(item);
				CNTK::DeviceDescriptor device = gpu_id >= 0 ? CNTK::DeviceDescriptor::GPUDevice((unsigned int)gpu_id) : CNTK::DeviceDescriptor::CPUDevice();
				CNTK::FunctionPtr modelFunc = CNTK::Function::Load(model_path, device, CNTK::ModelFormat::CNTKv2);
				device_models.push_back(DeviceModel(device, modelFunc));
			}
		}
		cvt = shared_ptr<DNNConverter>(new DNNConverter(format_board, format_move));

		// スレッド間キュー初期化
		eval_queue = new ipqueue<dnn_eval_obj>(block_queue_length, batch_size);
		result_queue = new ipqueue<dnn_result_obj>(block_queue_length, batch_size);

		// 評価スレッドを立てる
		for (int i = 0; i < device_models.size(); i++)
		{
			dnn_threads.push_back(new std::thread(dnn_thread_main, i));
		}
		dnn_initialized = true;
	}

	hash_init_thread.join();

#ifdef _DEBUG
	std::this_thread::sleep_for(std::chrono::seconds(5));
#endif
}

void flush_queue()
{
	if (eval_queue_batch_index > 0)
	{
		eval_objs->count = eval_queue_batch_index;
		eval_queue->end_write();
		eval_objs = nullptr;
		eval_queue_batch_index = 0;
		n_batch_put++;
	}
}


bool dnn_write_eval_obj(dnn_eval_obj *eval_obj, const Position &pos)
{
	cvt->get_board_array(pos, eval_obj->input_array);

	int m_i = 0;
	bool not_mate = false;
	for (auto m : MoveList<LEGAL>(pos))
	{
		dnn_move_index dmi;
		dmi.move = (uint16_t)m.move;
		dmi.index = (uint16_t)cvt->get_move_index(pos, m.move);
		eval_obj->move_indices[m_i] = dmi;
		m_i++;
		not_mate = true;
	}
	eval_obj->n_moves = m_i;
	return not_mate;
}


bool enqueue_pos(Position &pos, dnn_table_index &path, float &score, bool use_mate_search)
{
	if (pos.DeclarationWin() != MOVE_NONE)
	{
		// 勝ち局面
		score = 1.0;
		return false;
	}
	//if (use_mate_search && mate_search_leaf->dfpn(pos, nullptr))
	//{
	//	// 詰めて勝てる局面
	//	score = 1.0;
	//	mate_search_leaf_count++;
	//	return false;
	//}
	if (!eval_objs)
	{
		while (!(eval_objs = eval_queue->begin_write()))
		{
			std::this_thread::sleep_for(std::chrono::microseconds(1));
		}
		eval_queue_batch_index = 0;
	}
	dnn_eval_obj *eval_obj = &eval_objs->elements[eval_queue_batch_index];
	bool not_mate = dnn_write_eval_obj(eval_obj, pos);
	if (not_mate)
	{
		eval_obj->index = path;
		eval_queue_batch_index++;
		if (eval_queue_batch_index == eval_queue->batch_size())
		{
			flush_queue();
		}
		score = 0.0;//この値は使われないはず
	}
	else
	{
		score = -1.0;
	}
	return not_mate;
}

// treeのbackup操作。
void backup_tree(float leaf_score, dnn_table_index &path)
{
	float score = leaf_score;

	// treeをたどり値を更新
	for (int i = path.path_length - 2; i >= 0; i--)
	{
		//score = -score;
		score = score * -0.99F;//逃げる時はより長い詰み筋、追うときは短い詰み筋を選ぶよう調整
		UctNode &inner_node = node_hash->nodes[path.path_indices[i]];
		uint16_t edge = path.path_child_indices[i];
		int new_value_n = inner_node.value_n[edge] + 1 - tree_config.virtual_loss;
		inner_node.value_n[edge] = new_value_n;
		float new_value_w = inner_node.value_w[edge] + score + tree_config.virtual_loss;
		inner_node.value_w[edge] = new_value_w;
		// inner_node.vloss_ctr[edge]--;
		// inner_node.value_q[edge] = new_value_w / new_value_n;
		inner_node.value_n_sum += 1 - tree_config.virtual_loss;
	}
}

bool operator<(const dnn_move_prob& left, const dnn_move_prob& right) {
	// 確率で降順ソート用
	return left.prob_scaled > right.prob_scaled;
}

void update_on_dnn_result(dnn_result_obj *result_obj)
{
	dnn_table_index &path = result_obj->index;
	// 末端ノードの評価を記録
	UctNode &leaf_node = node_hash->nodes[path.path_indices[path.path_length - 1]];
	leaf_node.evaled = true;
	// 事前確率でソートし、上位 MAX_UCT_CHILDREN だけ記録
	int n_moves_use = result_obj->n_moves;
	if (n_moves_use > MAX_UCT_CHILDREN)
	{
		std::sort(&result_obj->move_probs[0], &result_obj->move_probs[result_obj->n_moves]);
		n_moves_use = MAX_UCT_CHILDREN;
	}
	for (int i = 0; i < n_moves_use; i++)
	{
		dnn_move_prob &dmp = result_obj->move_probs[i];
		leaf_node.move_list[i] = (Move)dmp.move;
		leaf_node.value_p[i] = dmp.prob_scaled / 65535.0F;
		// n, w, qは0初期化されている
	}
	leaf_node.n_children = n_moves_use;
	float score = result_obj->static_value / 32000.0F * tree_config.value_scale; // [-1.0, 1.0]
	leaf_node.score = score;

	backup_tree(score, path);
	DupEvalChain *dec = leaf_node.dup_eval_chain;
	while (dec != nullptr)
	{
		backup_tree(score, dec->path);
		DupEvalChain *dec_next = dec->next;
		delete dec;
		dec = dec_next;
	}
}

void update_on_mate(dnn_table_index &path, float mate_score)
{
	// 新規展開ノードがmateだったときの処理
	UctNode &leaf_node = node_hash->nodes[path.path_indices[path.path_length - 1]];
	leaf_node.evaled = true;
	leaf_node.terminal = true;
	leaf_node.score = mate_score;
	backup_tree(mate_score, path);
}

// 末端ノードが評価不要ノードだった場合
void update_on_terminal(float leaf_score, dnn_table_index &path)
{
	special_terminal_count_this_search++;
	backup_tree(leaf_score, path);
}

bool receive_result(bool block)
{
	int receive_count = 0;
	ipqueue_item<dnn_result_obj> *result_objs = nullptr;
	if (block)
	{
		while (!(result_objs = result_queue->begin_read()))
		{
			std::this_thread::sleep_for(std::chrono::microseconds(1));
		}
	}
	else
	{
		result_objs = result_queue->begin_read();
		if (!result_objs)
		{
			return false;
		}
	}

	for (size_t i = 0; i < result_objs->count; i++)
	{
		update_on_dnn_result(&result_objs->elements[i]);
		receive_count++;
	}

	result_queue->end_read();
	n_batch_get++;

	eval_count_this_search += receive_count;

	return true;
}

int get_or_create_root(Position &pos)
{
	bool created;

	int index = node_hash->find_or_create_index(pos, &created);
	if (!created)
	{
		sync_cout << "info string root cached" << sync_endl;
		return index;
	}

	sync_cout << "info string creating root node" << sync_endl;
	UctNode *node = &node_hash->nodes[index];

	// 局面評価
	dnn_table_index path;
	path.path_length = 1;
	path.path_indices[0] = index;
	float mate_score;
	bool not_mate = enqueue_pos(pos, path, mate_score, false);
	if (not_mate)
	{
		// 評価待ち
		flush_queue();
		receive_result(true);
	}
	else
	{
		// 詰んでいて評価対象にならない
		update_on_mate(path, mate_score);
	}

	return index;
}

int select_edge(UctNode &node)
{
	float n_sum_sqrt = sqrt((float)node.value_n_sum) + 0.001F;//完全に0だと最初の1手が事前確率に沿わなくなる
	int best_index = 0;
	float best_value = -100.0F;
	for (size_t i = 0; i < node.n_children; i++)
	{
		float value_n = (float)node.value_n[i];
		float value_u = node.value_p[i] / (value_n + 1) * tree_config.c_puct * n_sum_sqrt;
		float value_q = node.value_w[i] / (value_n + 1e-8F);//0除算回避
		float value_sum = value_q + value_u;
		if (value_sum > best_value)
		{
			best_value = value_sum;
			best_index = i;
		}
	}

	return best_index;
}

int total_dup_eval = 0;
bool dup_eval_flag = false;
int current_max_depth = 0;

void mcts_select(int node_index, dnn_table_index &path, Position &pos)
{
	if (path.path_length >= MAX_SEARCH_PATH_LENGTH)
	{
		// 千日手模様の筋などで起こるかもしれないので一応対策
		// 引き分けとみなして終了する
		update_on_terminal(0.0, path);
		return;
	}

	UctNode &node = node_hash->nodes[node_index];
	if (node.terminal)
	{
		// 詰みノード
		// 評価は不要で、親へ評価値を再度伝播する
		update_on_terminal(node.score, path);
		return;
	}

	if (path.path_length > 1) // ルートノード自体を千日手とは判定しない
	{
		RepetitionState rep_state = pos.is_repetition(pos.game_ply() - path.path_length);
		if (rep_state != RepetitionState::REPETITION_NONE)
		{
			float score;
			switch (rep_state)
			{
			case REPETITION_WIN:
			case REPETITION_SUPERIOR:
				score = 1.0;
				break;
			case REPETITION_LOSE:
			case REPETITION_INFERIOR:
				score = -1.0;
				break;
			default:
				score = 0.0;
				break;
			}

			update_on_terminal(score, path);
			return;
		}
	}

	if (!node.evaled)
	{
		// ノードが評価中だった場合
		// virtual lossがあるので、評価が終わったときに追加でbackupを呼ぶようにする
		// link listにつなぐ
		DupEvalChain *dec = new DupEvalChain();
		memcpy(&dec->path, &path, sizeof(dnn_table_index));
		dec->next = node.dup_eval_chain;
		node.dup_eval_chain = dec;
		total_dup_eval++;
		dup_eval_flag = true;
		return;
	}

	// エッジ選択
	int edge = select_edge(node);

	// virtual loss加算
	node.value_n[edge] += tree_config.virtual_loss;
	node.value_n_sum += tree_config.virtual_loss;
	node.value_w[edge] -= tree_config.virtual_loss;
	// node.vloss_ctr[edge]++;
	// node.value_q[edge] = node.value_w[edge] / node.value_n[edge];

	Move m = node.move_list[edge];
	StateInfo si;
	pos.do_move(m, si);

	// 子ノードを選択するか生成
	bool created;
	int child_index = node_hash->find_or_create_index(pos, &created);
	path.path_child_indices[path.path_length - 1] = edge;
	path.path_indices[path.path_length] = child_index;
	path.path_length++;
	if (path.path_length > current_max_depth)
	{
		current_max_depth = path.path_length;
	}

	if (created)
	{
		// 新規子ノードなので、評価
		float mate_score;
		bool not_mate = enqueue_pos(pos, path, mate_score, false);
		if (not_mate)
		{
			// 評価待ち
			// 非同期に処理される
		}
		else
		{
			// 詰んでいて評価対象にならない
			update_on_mate(path, mate_score);
		}
	}
	else
	{
		// 再帰的に探索
		mcts_select(child_index, path, pos);
	}

	pos.undo_move(m);
}

// pv取得。winrateはルートノードでのbestMoveの勝率。mate_inは、読み筋の末端が詰みのときの手数。ルートが詰んでいたら0。詰まないとき負の値。
void get_pv(int cur_index, vector<Move> &pv, Position &pos, bool root, float &winrate, int &mate_in)
{
	if (root)
	{
		// mate_inが初期化されないコードパス対策
		mate_in = -1000;
	}
	UctNode *node = &node_hash->nodes[cur_index];
	if (node->terminal)
	{
		if (root)
		{
			winrate = node->score;
		}
		mate_in = 0;//入玉宣言だと符号が変になるが表示上の問題だけ
		return;
	}
	int best_n = -1;
	Move bestMove = MOVE_RESIGN;
	int best_child_i = 0;
	for (size_t i = 0; i < node->n_children; i++)
	{
		if (node->value_n[i] > best_n)
		{
			best_n = node->value_n[i];
			bestMove = node->move_list[i];
			best_child_i = i;
		}
	}
	if (pos.pseudo_legal(bestMove) && pos.legal(bestMove))
	{
		pv.push_back(bestMove);
		StateInfo si;
		pos.do_move(bestMove, si);
		int child_index = node_hash->find_index(pos);
		if (child_index >= 0)
		{
			get_pv(child_index, pv, pos, false, winrate, mate_in);
			mate_in++;
		}
		else
		{
			// 読み筋が途切れた
			// 詰まないものとして扱う
			mate_in = -1000;
		}
		pos.undo_move(bestMove);
	}
	if (root)
	{
		winrate = node->value_w[best_child_i] / node->value_n[best_child_i];
	}
}

int winrate_to_cp(float winrate)
{
	// 勝率-1.0~1.0を評価値に変換する
	// tanhの逆関数 (1/2)*log((1+x)/(1-x))
	// 1, -1ならinfになるので丸める
	// 1歩=100だが、そういう評価関数を作っていないためスケールはそれっぽく見えるものにするほかない
	float v = (log1pf(winrate) - log1pf(-winrate)) * 600;
	if (v < -30000)
	{
		v = -30000;
	}
	else if (v > 30000)
	{
		v = 30000;
	}
	return (int)v;
}

void print_pv(int root_index, Position &rootPos)
{
	UctNode *root_node = &node_hash->nodes[root_index];
	vector<Move> pv;
	float winrate;
	int mate_in;
	get_pv(root_index, pv, rootPos, true, winrate, mate_in);
	int elapsed_ms = Time.elapsed();
	int nps = (int)((long long)eval_count_this_search * 1000 / max(elapsed_ms, 1));//0除算回避, 2Mノード以上読むとintではオーバーフロー
	int hashfull = (int)((long long)node_hash->used * 1000 / node_hash->uct_hash_size);
	sync_cout << "info nodes " << root_node->value_n_sum << " depth " << pv.size();
	//if (mate_in >= 0)
	//{
	//	char* sign = "";
	//	if (mate_in % 2 == 0)
	//	{
	//		// 詰まされる方向のときはマイナスをつける(詰んでいて0手のときも-を付ける)
	//		sign = "-";
	//	}
	//	cout << " score mate " << sign << mate_in;
	//}
	//else
	//{
	// // 現状mate_inが信用ならないので表示しない(greedyな読み筋の末端がmateなだけで外すことが多い)
	cout << " score cp " << winrate_to_cp(winrate);
	//}

	cout << " time " << elapsed_ms << " nps " << nps << " hashfull " << hashfull << " pv";
	for (auto m : pv)
	{
		std::cout << " " << m;
	}
	std::cout << sync_endl;
}

void select_best_move(Position &rootPos, UctNode &root_node, Move &bestMove, Move &ponderMove)
{
	int best_n = -1;
	sync_cout << "info string n ";
	int best_child_index = -1;
	// greedy
	// TODO: play temperature版
	for (size_t i = 0; i < root_node.n_children; i++)
	{
		std::cout << root_node.value_n[i] << "(" << root_node.move_list[i] << ") ";
		if (root_node.value_n[i] > best_n)
		{
			best_child_index = i;
			best_n = root_node.value_n[i];
			bestMove = root_node.move_list[i];
		}
	}
	std::cout << sync_endl;

	if (best_child_index >= 0)
	{
		// 自分が指した後の局面でgreedyに指し手を選びponderにする
		StateInfo si;
		rootPos.do_move(bestMove, si);
		int child_index = node_hash->find_index(rootPos);
		if (child_index >= 0)
		{
			auto &best_child_node = node_hash->nodes[child_index];
			int best_child_n = -1;
			for (size_t i = 0; i < best_child_node.n_children; i++)
			{
				int node_n = best_child_node.value_n[i];
				if (node_n > best_child_n)
				{
					best_child_n = node_n;
					ponderMove = best_child_node.move_list[i];
				}
			}
		}
		rootPos.undo_move(bestMove);
	}
}

std::atomic_bool in_search_time;

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。
void MainThread::think()
{
	Time.init(Search::Limits, rootPos.side_to_move(), rootPos.game_ply());
	long long next_pv_time = 0;
	in_search_time = true;
	if (tree_config.clear_table_before_search)
	{
		node_hash->clear();
	}

	Move bestMove = MOVE_RESIGN;
	Move ponderMove = MOVE_RESIGN;
	Move declarationWinMove = rootPos.DeclarationWin();
	if (declarationWinMove != MOVE_NONE)
	{
		// 入玉宣言勝ち
		bestMove = declarationWinMove;
		while (Threads.ponder && !Threads.stop)
		{
			// ここのThreads.stopは実際にstopコマンドが来たことを表さないといけない。探索終了時間などで書き換えると違反になる。
			// ponder中は返してはいけない
			sleep(1);
		}
	}
	else
	{
		// これより古いハッシュテーブル要素はもう使わないので消していい。
		// ponderの場合、rootの親（現在相手が思考中の局面と同じ手数）までは残さないといけない。
		node_hash->obsolete_game_ply = rootPos.game_ply() - 1;

		int root_index = get_or_create_root(rootPos);
		UctNode &root_node = node_hash->nodes[root_index];
		int n_select = root_node.value_n_sum;

#ifdef USE_MCTS_MATE_ENGINE
		root_mate_found = false;
		root_mate_pv.clear();
		Threads[1]->start_searching();
#endif

		if (!root_node.terminal && bestMove == MOVE_RESIGN)
		{
			eval_count_this_search = 0;
			special_terminal_count_this_search = 0;
			mate_search_leaf_count = 0;

			// 事前確率表示
			sync_cout << "info string prob ";
			float best_p = -10.0;
			Move best_p_move = MOVE_RESIGN;
			for (size_t i = 0; i < root_node.n_children; i++)
			{
				std::cout << root_node.move_list[i] << "(" << (int)(root_node.value_p[i] * 100) << "%) ";
				if (root_node.value_p[i] > best_p)
				{
					best_p = root_node.value_p[i];
					best_p_move = root_node.move_list[i];
				}
			}
			std::cout << sync_endl;
			sync_cout << "info score cp " << winrate_to_cp(best_p) << " pv " << best_p_move << sync_endl;

			total_dup_eval = 0;
			auto timer_thread = std::thread([] {
				while (!Threads.stop && (Threads.ponder || Time.elapsed() < Time.optimum()) && in_search_time)
				{
					sleep(10);
				}
				// Time.elapsed() < Time.optimum()の場合、ponderで開始してからの時間になる。フィッシャークロックルールならこれで問題ない。
				// 秒読み状態だと無駄になってしまう。
				// これで停止フラグを立てた後、DNN評価が返ってくるまで待つ必要があるのでTime.maximum()は危険。
				in_search_time = false;
			});
			// 探索時間内は木構造探索をする。同時に評価結果を回収。時間切れになったらflushして投入済み評価バッチを回収。
			bool no_more_search = false;
			while (true)
			{
				if (in_search_time && (n_select < max_select || Threads.ponder))  // Ponder中は指定ノード数を超えても探索する
				{
					dnn_table_index path;
					path.path_length = 1;
					path.path_indices[0] = root_index;
					dup_eval_flag = false;
					mcts_select(root_index, path, rootPos);
					n_select++;
					if (dup_eval_flag)
					{
						flush_queue();
					}
				}
				else
				{
					if (!no_more_search)
					{
						// これ以上投入しないのでflushする
						flush_queue();
					}
					no_more_search = true;
				}

				int pending_batches = n_batch_put - n_batch_get;
				if (no_more_search && pending_batches == 0)
				{
					break;
				}
				if (pending_batches > 0)
				{
					bool block = false;
					if (root_node.value_n_sum < 10000)
					{
						// 探索回数が少ないうちに複数のバッチを評価待ちにすると、重複が多くなりバイアスが大きくなる
						block = true;
					}
					else if (pending_batches >= block_queue_length)
					{
						block = true;
					}

					if (receive_result(block || no_more_search))
					{
						// 頻繁に時刻取得をするのも無駄そうなのでここで
						auto elapsed = Time.elapsed();
						if (elapsed >= next_pv_time)
						{
							print_pv(root_index, rootPos);
							next_pv_time += pv_interval;
						}
					}
				}
			}

			in_search_time = false;//n_select < max_selectの条件でwhileを抜けてもtimer_threadを終了させるため
			timer_thread.join();

			select_best_move(rootPos, root_node, bestMove, ponderMove);

			sync_cout << "info string dup eval=" << total_dup_eval << " special=" << special_terminal_count_this_search << " mate_leaf=" << mate_search_leaf_count << sync_endl;
			sync_cout << "info string max depth=" << current_max_depth << sync_endl;
			sync_cout << "info string obsolete_removed=" << obsolete_removed_count << sync_endl;
			print_pv(root_index, rootPos);
		}


#ifdef _DEBUG
		std::this_thread::sleep_for(std::chrono::seconds(5));
#endif
		while (Threads.ponder && !Threads.stop)
		{
			// ここのThreads.stopは実際にstopコマンドが来たことを表さないといけない。探索終了時間などで書き換えると違反になる。
			// ponder中は返してはいけない
			sleep(1);
		}
		Threads.stop = true;
#ifdef USE_MCTS_MATE_ENGINE
		Threads[1]->wait_for_search_finished();

		if (root_mate_found)
		{
			bestMove = root_mate_pv[0];
			if (root_mate_pv.size() >= 2)
			{
				ponderMove = root_mate_pv[1];
			}
			else
			{
				// 1手詰めのときはponderを出さない
				ponderMove = MOVE_RESIGN;
			}
		}
#endif
	}
	sync_cout << "bestmove " << bestMove;
	if (ponderMove != MOVE_RESIGN)
	{
		cout << " ponder " << ponderMove;
	}
	cout << sync_endl;

}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
// この関数を呼び出したいときは、Thread::search()とすること。
void Thread::search()
{
	// 詰み探索slaveスレッドを1個だけ立てる
	//sync_cout << "info string mate search thread started" << sync_endl;
	//if (mate_search_root->dfpn(rootPos, &root_mate_pv))
	//{
	//	root_mate_found = true;
	//	sync_cout << "info string root MATE FOUND!" << sync_endl;
	//	sync_cout << "info depth " << root_mate_pv.size() << " score mate " << root_mate_pv.size() << " pv";
	//	for (auto m : root_mate_pv)
	//	{
	//		cout << " " << m;
	//	}
	//	cout << sync_endl;
	//}
	//else
	//{
	//	sync_cout << "info string NO root mate" << sync_endl;
	//}
}

#endif // USER_ENGINE
