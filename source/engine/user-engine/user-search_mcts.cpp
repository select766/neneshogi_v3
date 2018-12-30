#ifdef USER_ENGINE_MCTS
#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "mcts.h"
#include "dnn_thread.h"
#include "gpu_lock.h"

static MCTS *mcts = nullptr;
static vector<MTQueue<dnn_eval_obj*>*> response_queues;
static vector<MateEngine::MateSearchForMCTS*> leaf_mate_searchers;
static MateEngine::MateSearchForMCTS *root_mate_searcher = nullptr;
static int pv_interval;//PV表示間隔[ms]
static int root_mate_thread_id = -1;//ルート局面からの詰み探索をするスレッドのid(-1の場合はしない)
static vector<Move> root_mate_pv;
static atomic_bool root_mate_found = false;//ルート局面からの詰み探索で詰みがあった場合
static int nodes_limit = NODES_LIMIT_MAX;//探索ノード数の上限
static bool already_initialized = false;//一度Search::clearで初期化済みかどうか。

// 定跡の指し手を選択するモジュール
static Book::BookMoveSelector book;

// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos_, istringstream& is)
{
	string token;
	is >> token;
	if (token == "dnnbench")
	{
		// DNNを単純に動作させた場合のnpsをベンチマークする。
		// isreadyでモデルを読み込み終わっている必要がある。
		int count;//サンプル数
		is >> count;

		sync_cout << "info string start bench" << sync_endl;
		MTQueue<dnn_eval_obj*> *response_queue = response_queues[0];
		int n_put = 0, n_get = 0;
		std::chrono::system_clock::time_point bench_start = std::chrono::system_clock::now();
		while (n_get < count)
		{
			if (n_put < count && (n_put - n_get) < batch_size * n_gpu_threads * 2)
			{
				for (size_t i = 0; i < batch_size; i++)
				{
					dnn_eval_obj *eobj = new dnn_eval_obj();
					// ダミーデータを入れておく
					eobj->n_moves = 1;
					eobj->move_indices[0].move = MOVE_NONE;
					eobj->move_indices[0].index = 0;
					memset(eobj->input_array, 0, sizeof(eobj->input_array));

					eobj->response_queue = response_queue;
					request_queues[n_put % request_queues.size()]->push(eobj);
					n_put++;
				}

			}

			dnn_eval_obj *eobj_ret = nullptr;
			while (response_queue->pop_nb(eobj_ret))
			{
				n_get++;
				delete eobj_ret;
			}
		}
		std::chrono::system_clock::time_point bench_end = std::chrono::system_clock::now();
		// 正しいキャスト方法がよく分かってない
		double elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start)).count() / 1000.0;
		int nps = (int)(n_put / elapsed);

		sync_cout << "info string bench done " << elapsed << " sec, nps=" << nps << sync_endl;
	}
}


// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	//   定跡設定
	book.init(o);

	o["PvInterval"] << Option(300, 0, 100000);//PV表示間隔[ms]
	o["BatchSize"] << Option(16, 1, 65536);
	o["GPU"] << Option("-1");//使用するGPU番号(-1==CPU)、カンマ区切りで複数指定可能
	o["DNNFormatBoard"] << Option(0, 0, 16);//DNNのboard表現形式
	o["DNNFormatMove"] << Option(0, 0, 16);//DNNのmove表現形式
	o["LeafMateSearchDepth"] << Option(0, 0, 16);//末端局面での詰み探索深さ(0なら探索しない)
	o["MCTSHash"] << Option(1024, 1, 1048576);//MCTSのハッシュテーブルサイズ(MB)
	o["RootMateSearch"] << Option(false);//ルート局面からの詰み探索専用スレッドを用いるか(Threadsのうちの1つが使われる)
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
	if (!already_initialized)
	{
		// 初期化する
		gpu_lock_thread_start();
		int hash_size_mb = (int)Options["MCTSHash"];
		mcts = new MCTS(MCTSTT::calc_uct_hash_size(hash_size_mb));
		batch_size = (int)Options["BatchSize"];
		pv_interval = (int)Options["PvInterval"];
		if (pv_interval == 0)
		{
			//PVの定期的な表示をしない
			pv_interval = 100000000;
		}
		nodes_limit = (int)Options["NodesLimit"];
		if (nodes_limit <= 0)
		{
			nodes_limit = NODES_LIMIT_MAX;
		}

		sync_cout << "info string initializing dnn threads" << sync_endl;
		vector<int> gpuIds;
		string evalDir = Options["EvalDir"];
		stringstream ss(Options["GPU"]);//カンマ区切りでGPU番号を並べる
		string item;
		while (getline(ss, item, ',')) {
			if (!item.empty()) {
				int gpu_id = stoi(item);
				gpuIds.push_back(gpu_id);
			}
		}
		start_dnn_threads(evalDir, (int)Options["DNNFormatBoard"], (int)Options["DNNFormatMove"], gpuIds);

		// スレッド間キュー初期化
		int threads = (int)Options["Threads"];
		for (int i = 0; i < threads; i++)
		{
			response_queues.push_back(new MTQueue<dnn_eval_obj*>());
		}

		// 末端詰み探索の初期化
		int LeafMateSearchDepth = (int)Options["LeafMateSearchDepth"];
		for (int i = 0; i < threads; i++)
		{
			if (LeafMateSearchDepth > 0)
			{
				auto ms = new MateEngine::MateSearchForMCTS();
				ms->init(16, LeafMateSearchDepth);
				leaf_mate_searchers.push_back(ms);
			}
			else
			{
				leaf_mate_searchers.push_back(nullptr);
			}
		}

		// ルート局面からの詰み探索
		if ((bool)Options["RootMateSearch"])
		{
			root_mate_thread_id = threads - 1;//最終スレッドを使う
			auto ms = new MateEngine::MateSearchForMCTS();
			ms->init(128, MAX_PLY);
			root_mate_searcher = ms;
		}
		else
		{
			root_mate_thread_id = -1;
		}

		// -----------------------
		//   定跡の読み込み
		// -----------------------

		book.read_book();

		sync_cout << "info string initialized all dnn threads" << sync_endl;

		already_initialized = true;
	}
	else
	{
		// 同じ設定で2度目以降の対局を行う。
		// もし違う設定が来ていても認識しない。
		sync_cout << "info string already initialized" << sync_endl;
		mcts->clear();
	}
}

// 探索に関する統計情報のリセット。思考開始時に呼ぶ。
static void reset_stats()
{
	n_dnn_evaled_batches = 0;
	n_dnn_evaled_samples = 0;
}

// 探索に関する統計情報の表示。
static void display_stats()
{
	int d_batches = n_dnn_evaled_batches;
	int d_samples = n_dnn_evaled_samples;
	int avg_batchsize = d_samples / std::max(d_batches, 1);
	sync_cout << "info string DNN stat " << d_batches << " batch, " << d_samples << " samples, "
		" average bs=" << avg_batchsize << " (" << (avg_batchsize * 100 / batch_size) << "%)"
		<< sync_endl;
}

static int winrate_to_cp(float winrate)
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

// PVおよび付随情報(nps等)の表示
static vector<Move> display_pv(UCTNode *root, Position &rootPos)
{
	int elapsed_ms = Time.elapsed();
	int nps = (int)((long long)n_dnn_evaled_samples * 1000 / max(elapsed_ms, 1));
	int hashfull = mcts->get_hashfull();
	vector<Move> pv;
	float winrate;
	mcts->get_pv(root, rootPos, pv, winrate);
	sync_cout << "info";
	cout << " nodes " << root->value_n_sum;
	cout << " depth " << pv.size();
	cout << " score cp " << winrate_to_cp(winrate);
	cout << " time " << elapsed_ms << " nps " << nps << " hashfull " << hashfull << " pv";
	for (auto m : pv)
	{
		cout << " " << m;
	}
	cout << sync_endl;
	return pv;
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。
void MainThread::think()
{
	// 例)
	//  for (auto th : Threads.slaves) th->start_searching();
	//  Thread::search();
	//  for (auto th : Threads.slaves) th->wait_for_search_finished();
	Time.init(Search::Limits, rootPos.side_to_move(), rootPos.game_ply());
	gpu_lock_extend();
	reset_stats();
	Move bestMove = MOVE_RESIGN;
	Move ponderMove = MOVE_RESIGN;
	Move declarationWinMove = rootPos.DeclarationWin();
	Move bookMove;
	if ((bookMove = book.probe(rootPos)) != MOVE_NONE)
	{
		// 定跡
		sync_cout << "info string book " << bookMove << sync_endl;
		bestMove = bookMove;
		while (Threads.ponder && !Threads.stop)
		{
			// ponder中は返してはいけない。
			sleep(1);
		}
	}
	else if (declarationWinMove != MOVE_NONE)
	{
		// 入玉宣言勝ち
		bestMove = declarationWinMove;
		while (Threads.ponder && !Threads.stop)
		{
			// ponder中は返してはいけない。
			sleep(1);
		}
	}
	else if (!rootPos.is_mated())
	{
		// ルートノードの作成
		MCTSSearchInfo sei(cvt, request_queues[thread_id() % request_queues.size()], response_queues[thread_id()], nullptr);
		dnn_eval_obj *eobj = new dnn_eval_obj();
		bool created;
		UCTNode *root = mcts->make_root(rootPos, sei, eobj, created);
		sync_cout << "info string created root " << created << " dnn " << sei.put_dnn_eval << sync_endl;
		// DNN評価が生じた場合はその結果を待つ
		if (sei.put_dnn_eval)
		{
			dnn_eval_obj *sentback;
			sei.response_queue->pop(sentback);
			mcts->backup_dnn(sentback);
			delete sentback;
			root->pprint();
		}
		else
		{
			delete eobj;
		}
		if (root->terminal)
		{
			// ルート局面にて以前詰みが見つかっているが、それだと指し手が決まらないのでそのフラグを解除して探索させる
			sync_cout << "info string root is terminal (found mate)" << sync_endl;
			root->terminal = false;

		}

		// slaveスレッドで探索を開始
		root_mate_found = false;
		for (Thread* th : Threads)
			if (th != this)
				th->start_searching();

		int lastPvTime = Time.elapsed();
		// masterは探索終了タイミングの決定のみ行う
		while (!Threads.stop)
		{
			sleep(10);
			if (lastPvTime + pv_interval < Time.elapsed())
			{
				display_pv(root, rootPos);
				lastPvTime += pv_interval;
			}

			// 探索終了条件判定
			if (!Threads.ponder)
			{
				// Ponder中は探索を止めない。
				// Ponderが外れた時、Threads.ponder==trueのままThreads.stop==trueとなる
				if (Time.elapsed() >= Time.optimum() || root->value_n_sum > nodes_limit || root_mate_found)
				{
					// 思考時間が来たら、新たな探索は停止する。
					// ただし、評価途中のものの結果を受け取ってからbestmoveを決める。
					// TODO: root->value_n_sum をロックすべき
					Threads.stop = true;
				}
			}
		}

		// slaveスレッドが探索を終わるのを待つ
		for (Thread* th : Threads)
			if (th != this)
				th->wait_for_search_finished();
		root->pprint();
		display_stats();
		vector<Move> pv = display_pv(root, rootPos);

		// 詰み探索が成功していれば、そちらを優先
		if (root_mate_found)
		{
			sync_cout << "info string override move by mate search" << sync_endl;
			pv = root_mate_pv;
		}

		if (pv.size() >= 1)
		{
			bestMove = pv[0];
			if (pv.size() >= 2)
			{
				ponderMove = pv[1];
			}
		}
	}

	if (ponderMove != MOVE_RESIGN)
	{
		sync_cout << "bestmove " << bestMove << " ponder " << ponderMove << sync_endl;
	}
	else
	{
		sync_cout << "bestmove " << bestMove << sync_endl;
	}
}

void root_mate_search(Position &rootPos)
{
	root_mate_pv.clear();
	if (root_mate_searcher->dfpn(rootPos, &root_mate_pv))
	{
		root_mate_found = true;
		sync_cout << "info string root MATE FOUND!" << sync_endl;
		sync_cout << "info depth " << root_mate_pv.size() << " score mate " << root_mate_pv.size() << " pv";
		for (auto m : root_mate_pv)
		{
			cout << " " << m;
		}
		cout << sync_endl;
	}
	else
	{
		sync_cout << "info string NO root mate" << sync_endl;
	}
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
// この関数を呼び出したいときは、Thread::search()とすること。
void Thread::search()
{
	if (thread_id() == root_mate_thread_id)
	{
		return root_mate_search(rootPos);
	}
	UCTNode *root = mcts->get_root(rootPos);
	int n_put = 0, n_get = 0, leaf_dup = 0, leaf_mate_search_found = 0;
	MTQueue<dnn_eval_obj*> *response_queue = response_queues[thread_id()];
	MTQueue<dnn_eval_obj*> *request_queue = request_queues[thread_id() % request_queues.size()];
	while (!Threads.stop || (n_put != n_get))
	{
		bool enable_search = !Threads.stop && (n_put - n_get < batch_size * 2);
		if (enable_search)
		{
			// 探索
			MCTSSearchInfo sei(cvt, request_queue, response_queue, leaf_mate_searchers[thread_id()]);
			dnn_eval_obj *eobj = new dnn_eval_obj();
			mcts->search(root, rootPos, sei, eobj);
			if (sei.put_dnn_eval)
			{
				n_put++;
				if (sei.leaf_mate_search_found)
				{
					leaf_mate_search_found++;
				}
			}
			else
			{
				delete eobj;
				if (sei.leaf_dup)
				{
					// すでに評価中の局面に到達
					// 木構造が狭い間に無理にたくさん評価しようとすると訪問回数が異常になるので
					// ヒューリスティックに、短い時間待機
					// もっと洗練された方法が欲しい
					leaf_dup++;
					sleep(1);
				}
			}

		}

		if (true)
		{
			dnn_eval_obj *eobj = nullptr;
			if (enable_search)
			{
				response_queue->pop_nb(eobj);
			}
			else
			{
				// 探索を停止している条件のため、結果が来るまでブロッキング
				response_queue->pop(eobj);
			}
			if (eobj)
			{
				mcts->backup_dnn(eobj);
				delete eobj;
				n_get++;
			}
		}
	}
	sync_cout << "info string thread " << thread_id() << " n_put " << n_put
		<< " leaf_dup " << leaf_dup << " leaf_mate_search_found " << leaf_mate_search_found << sync_endl;
}

#endif // USER_ENGINE_MCTS
