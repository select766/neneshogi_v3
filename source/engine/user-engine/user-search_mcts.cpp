﻿#include "../../extra/all.h"
#ifdef USER_ENGINE_MCTS
#include <cstdlib>
#include "mcts.h"
#include "dnn_thread.h"
#include "gpu_lock.h"
#include "tensorrt_engine_builder.h"

static MCTS *mcts = nullptr;
static vector<MTQueue<dnn_eval_obj *> *> response_queues;
static vector<MateEngine::MateSearchForMCTS *> leaf_mate_searchers;
static MateEngine::MateSearchForMCTS *root_mate_searcher = nullptr;
static int pv_interval;				 //PV表示間隔[ms]
static int root_mate_thread_id = -1; //ルート局面からの詰み探索をするスレッドのid(-1の場合はしない)
static vector<Move> root_mate_pv;
static atomic_bool root_mate_found(false); //ルート局面からの詰み探索で詰みがあった場合
static int nodes_limit = NODES_LIMIT_MAX;  //探索ノード数の上限
// floatで探索数をカウントしており、2^24になるとインクリメントができなくなる。
// 詰みに近い局面でPonderしているとこれに達して評価値がおかしくなるので、
// これ以上の探索数になったらウェイトを挿入して異常値を防止する。
static const int nodes_safety_max = 16777216 / 2;
static bool already_initialized = false; //一度Search::clearで初期化済みかどうか。
static atomic_size_t pending_limit(1);   //DNN評価待ちの要素数の最大数(スレッドごと)
static int pending_limit_factor = 16;
static size_t normal_slave_threads = 1; //通常探索をするslaveスレッド数
static bool policy_only = false;
static int limited_batch_size = 1;
static int limited_until = 0;
static int print_status_interval = 0;
static int early_stop_prob = 0;
// 環境変数で指定したサイズの置換表を事前確保
static std::thread *advance_hash_init_thread = nullptr;
static int advance_node_hash_size = 0; //MB単位

// 定跡の指し手を選択するモジュール
static Book::BookMoveSelector book;

// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position &pos_, istringstream &is)
{
	string token;
	is >> token;
	if (token == "dnnbench")
	{
		// DNNを単純に動作させた場合のnpsをベンチマークする。
		// isreadyでモデルを読み込み終わっている必要がある。
		int count; //サンプル数
		is >> count;

		sync_cout << "info string start bench" << sync_endl;
		MTQueue<dnn_eval_obj *> *response_queue = response_queues[0];
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
#ifndef DNN_EXTERNAL
	if (token == "tensorrt_engine_builder")
	{
		sync_cout << "info string tensorrt_engine_builder start" << sync_endl;
		string onnxModelPath, dstDir, profileBatchSizeRange;
		int batchSizeMin, batchSizeMax, fpbit;
		is >> onnxModelPath >> dstDir >> batchSizeMin >> batchSizeMax >> profileBatchSizeRange >> fpbit;
		bool ok = tensorrt_engine_builder(onnxModelPath.c_str(), dstDir.c_str(), batchSizeMin, batchSizeMax, profileBatchSizeRange.c_str(), fpbit);

		sync_cout << "info string tensorrt_engine_builder " << (ok ? "succeeded" : "failed") << sync_endl;
	}
#endif
}

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap &o)
{
	//   定跡設定
	book.init(o);

	o["PvInterval"] << Option(300, 0, 100000); //PV表示間隔[ms]
	o["BatchSize"] << Option(16, 1, 65536);
	o["GPU"] << Option("-1");					  //使用するGPU番号(-1==CPU)、カンマ区切りで複数指定可能
	o["DNNFormatBoard"] << Option(0, 0, 16);	  //DNNのboard表現形式
	o["DNNFormatMove"] << Option(0, 0, 16);		  //DNNのmove表現形式
	o["LeafMateSearchDepth"] << Option(0, 0, 16); //末端局面での詰み探索深さ(0なら探索しない)
	o["MCTSHash"] << Option(1024, 1, 1048576);	//MCTSのハッシュテーブルサイズ(MB)
	o["RootMateSearch"] << Option(false);		  //ルート局面からの詰み探索専用スレッドを用いるか(Threadsのうちの1つが使われる)
	o["PolicyOnly"] << Option(false);			  //policy評価だけで指し手を決定し、探索を行わない
	o["LimitedBatchSize"] << Option(16, 1, 65536);
	o["LimitedUntil"] << Option(0, 0, 1000000); //ルートのvalue_n_sumがこの値未満の時、バッチサイズがLimitedBatchSizeだとみなして評価待ち要素数を制限する
	// o["VirtualLoss"] << Option(1, 1, 1024);
	o["VirtualLoss"] << Option("1");
	o["CPuct"] << Option(100, 1, 10000);			   //c_puctの100倍
	o["PrintStatusInterval"] << Option(0, 0, 1000000); //ルートノードの状態表示間隔[nodes]
	o["EarlyStopProb"] << Option(0, 0, 100);		   //指し手変化確率[%]がこれを下回ったら、予定時間にかかわらず指す
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
	// 環境変数で指定したサイズの置換表を事前確保
	// 大会で、数十GBのメモリをisreadyの際に確保&ゼロクリアしようとすると数十秒かかる。
	// 対局開始になってからisreadyが来るため、相手を待たせてしまう。
	// 起動時に環境変数でサイズを指定された場合は、ここで確保しておくことによりサーバログイン直後に時間を使える。
	// 2局以上連続することは想定していない。最初の1局に対してのみ有効。
	char *advance_node_hash_size_str = getenv("NENESHOGI_NODE_HASH_SIZE"); //MB単位の文字列(MCTSHashと同じ)
	if (advance_node_hash_size_str && strlen(advance_node_hash_size_str) > 0)
	{
		advance_node_hash_size = strtol(advance_node_hash_size_str, nullptr, 10);
		if (advance_node_hash_size > 0)
		{
			advance_hash_init_thread = new std::thread([] {
				sync_cout << "info string advance node hash initializing" << sync_endl;
				mcts = new MCTS(MCTSTT::calc_uct_hash_size(advance_node_hash_size));
				sync_cout << "info string advance node hash init completed" << sync_endl;
			});
		}
	}
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void Search::clear()
{
	if (!already_initialized)
	{
		// 初期化する
		gpu_lock_thread_start();
		int hash_size_mb = (int)Options["MCTSHash"];
		bool advance_initialized = false;
		if (advance_hash_init_thread)
		{
			// Search::initで専用初期化スレッドが開始しているので、それを待つ
			if (hash_size_mb != advance_node_hash_size)
			{
				//サイズが間違ってるのでエラーとして終了
				sync_cout << "info string node hash size mismatch! " << hash_size_mb << "!=" << advance_node_hash_size << sync_endl;
				return;
			}
			advance_hash_init_thread->join();
			delete advance_hash_init_thread;
			advance_hash_init_thread = nullptr;
			advance_initialized = true;
		}
		else
		{
			mcts = new MCTS(MCTSTT::calc_uct_hash_size(hash_size_mb));
		}
		// mcts->virtual_loss = (int)Options["VirtualLoss"];
		mcts->virtual_loss = stof((string)Options["VirtualLoss"]);
		mcts->c_puct = ((int)Options["CPuct"]) * 0.01F;
		batch_size = (int)Options["BatchSize"];
		limited_batch_size = (int)Options["LimitedBatchSize"];
		limited_until = (int)Options["LimitedUntil"];
		pv_interval = (int)Options["PvInterval"];
		print_status_interval = (int)Options["PrintStatusInterval"];
		early_stop_prob = (int)Options["EarlyStopProb"];
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
		policy_only = (bool)Options["PolicyOnly"];

		sync_cout << "info string initializing dnn threads" << sync_endl;
		vector<int> gpuIds;
		string evalDir = Options["EvalDir"];
		stringstream ss(Options["GPU"]); //カンマ区切りでGPU番号を並べる
		string item;
		while (getline(ss, item, ','))
		{
			if (!item.empty())
			{
				int gpu_id = stoi(item);
				gpuIds.push_back(gpu_id);
			}
		}
		start_dnn_threads(evalDir, (int)Options["DNNFormatBoard"], (int)Options["DNNFormatMove"], gpuIds);

		// スレッド間キュー初期化
		int threads = (int)Options["Threads"];
		for (int i = 0; i < threads; i++)
		{
			response_queues.push_back(new MTQueue<dnn_eval_obj *>());
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
			root_mate_thread_id = threads - 1; //最終スレッドを使う
			auto ms = new MateEngine::MateSearchForMCTS();
			ms->init(128, MAX_PLY);
			root_mate_searcher = ms;
			normal_slave_threads = threads - 2;
		}
		else
		{
			root_mate_thread_id = -1;
			normal_slave_threads = threads - 1;
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
																					" average bs="
			  << avg_batchsize << " (" << (avg_batchsize * 100 / batch_size) << "%)"
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

static UCTNode *make_initial_nodes(Position &rootPos)
{
	// ルートノードの作成
#if 0
	// ルートだけを作成するのはバッチサイズが埋まらない&直後に同じ浅いノードの評価が殺到して評価値がゆがむので、
	// 幅優先探索でノードをまとめて評価を行い、置換表を埋めておく（backupはしない）
	MCTSSearchInfo sei(cvt, request_queues[0 % request_queues.size()], response_queues[0], nullptr);
	int n_put = 0;
	UCTNode *root = mcts->make_root_with_children(rootPos, sei, n_put, batch_size);
	int n_get = 0;
	sync_cout << "info string put root children " << n_put << sync_endl;
	while (n_get < n_put)
	{
		dnn_eval_obj *sentback;
		sei.response_queue->pop(sentback);
		mcts->backup_dnn(sentback, false);
		delete sentback;
		n_get++;
	}
	sync_cout << "info string evaluated root children " << n_put << sync_endl;
#else
	// ルートノードだけ作る
	MCTSSearchInfo sei(cvt, request_queues[0 % request_queues.size()], response_queues[0], nullptr);
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
#endif
	if (root->terminal)
	{
		// ルート局面にて以前詰みが見つかっているが、それだと指し手が決まらないのでそのフラグを解除して探索させる
		sync_cout << "info string root is terminal (found mate)" << sync_endl;
		root->terminal = false;
	}

	return root;
}

void update_pending_limit(UCTNode *root)
{
	size_t plimit_cand;
	if (root->value_n_sum < limited_until)
	{
		plimit_cand = limited_batch_size * n_gpu_threads * 2;
	}
	else
	{
		plimit_cand = batch_size * n_gpu_threads * 2;
	}
#if 0
	size_t plimit_cand = pending_limit_factor * log2(std::max(root->value_n_sum, 1));
	plimit_cand = std::min(std::max(plimit_cand, (size_t)16), batch_size * n_gpu_threads * 2);
#endif
	pending_limit = plimit_cand / normal_slave_threads;
}

// 探索途中でのルートノードからの各指し手情報のデバッグプリント
void print_search_status(UCTNode *root)
{
	// 自動処理したいのでjsonでパースできるようにする
	sync_cout << "info string PSS {";
	std::cout << "\"moves\":[";
	for (int i = 0; i < root->n_children; i++)
	{
		if (i > 0)
		{
			std::cout << ",";
		}
		std::cout << "{\"move\":\"" << root->move_list[i] << "\",";
		std::cout << "\"n\":" << root->value_n[i] << ",\"p\":" << root->value_p[i]
				  << ",\"w\":" << root->value_w[i] << "}";
	}
	std::cout << "]}";
	std::cout << sync_endl;
}

// 探索状況を見て、早期終了するかどうか判定
// 指し手変化がなさそうな場合に終了する。
bool decide_early_stop(UCTNode *root)
{
	int elapsed_ms = Time.elapsed();
	if (elapsed_ms <= Time.minimum())
	{
		// 一定時間以上考える
		return false;
	}
	int nps = (int)((long long)n_dnn_evaled_samples * 1000 / max(elapsed_ms, 1));
	int remaining_ms = Time.optimum() - Time.elapsed();
	float estimated_future_nodes = nps * remaining_ms / 1000.0F;
	float cur_nodes = root->value_n_sum;
	if (cur_nodes < limited_until)
	{
		// 一定ノード数以上探索する
		return false;
	}

	float max_nodes = -1;
	for (size_t i = 0; i < root->n_children; i++)
	{
		if (root->value_n[i] > max_nodes)
		{
			max_nodes = root->value_n[i];
		}
	}

	float pv_prob = max_nodes / cur_nodes;
	// PVの確率、現在ノード数、今後探索できそうなノード数から、指し手が変化する確率を推定
	float change_score = pv_prob * -5.326F + log2f(cur_nodes / 1024) * -0.306F + log2f(estimated_future_nodes / 1024) * 0.377F + 0.798F;
	float change_prob = 1.0F / (expf(-change_score) + 1.0F);
	if (change_prob < (early_stop_prob / 100.0F))
	{
		return true;
	}
	return false;
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
	else if (policy_only)
	{
		UCTNode *root = make_initial_nodes(rootPos);
		bestMove = mcts->get_bestmove(root, rootPos, true);
		while (Threads.ponder && !Threads.stop)
		{
			// ponder中は返してはいけない。
			sleep(1);
		}
	}
	else if (!rootPos.is_mated())
	{
		UCTNode *root = make_initial_nodes(rootPos);
		update_pending_limit(root);
		// slaveスレッドで探索を開始
		root_mate_found = false;
		for (Thread *th : Threads)
			if (th != this)
				th->start_searching();

		int lastPvTime = Time.elapsed();
		int next_status_print_nodes = 0;
		// masterは探索終了タイミングの決定のみ行う
		while (!Threads.stop)
		{
			update_pending_limit(root);

			// 探索終了条件判定
			if (!Threads.ponder)
			{
				// Ponder中は探索を止めない。
				// Ponderが外れた時、Threads.ponder==trueのままThreads.stop==trueとなる
				if (Time.elapsed() >= Time.optimum() || root->value_n_sum >= nodes_limit || root_mate_found || decide_early_stop(root))
				{
					// 思考時間が来たら、新たな探索は停止する。
					// ただし、評価途中のものの結果を受け取ってからbestmoveを決める。
					// TODO: root->value_n_sum をロックすべき
					Threads.stop = true;
				}
			}

			if (lastPvTime + pv_interval < Time.elapsed())
			{
				display_pv(root, rootPos);
				lastPvTime += pv_interval;
				// sync_cout << "info string pending_limit " << pending_limit << sync_endl;
			}

			if (print_status_interval > 0 && next_status_print_nodes <= root->value_n_sum)
			{
				// 置換表に最初からルートノードがあり、初回からroot_node.value_n_sumが大きい場合あり
				// root_node.value_n_sumより大きい最小のprint_statusの倍数
				print_search_status(root);
				next_status_print_nodes = ((int)root->value_n_sum + print_status_interval) / print_status_interval * print_status_interval;
			}

			sleep(10);
		}

		// slaveスレッドが探索を終わるのを待つ
		for (Thread *th : Threads)
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
	else
	{
		// ponderする局面が詰んでいる場合、ここに到達
		while (Threads.ponder && !Threads.stop)
		{
			// ponder中は返してはいけない。
			sleep(1);
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
	MTQueue<dnn_eval_obj *> *response_queue = response_queues[thread_id()];
	MTQueue<dnn_eval_obj *> *request_queue = request_queues[thread_id() % request_queues.size()];
	bool block_until_all_get = false;
	while (!Threads.stop || (n_put != n_get))
	{
		if (root->value_n_sum >= nodes_safety_max)
		{
			sleep(1);
		}

		bool enable_search = !Threads.stop && (n_put - n_get < pending_limit) && !block_until_all_get;
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
				if (sei.leaf_dup && (root->value_n_sum < limited_until))
				{
					// すでに評価中の局面に到達
					// 木構造が狭い間に無理にたくさん評価しようとすると訪問回数が異常になるので
					// ヒューリスティックに、短い時間待機
					// もっと洗練された方法が欲しい
					leaf_dup++;
					sleep(1);
					//if (n_put > n_get)
					//{
					//	block_until_all_get = true;
					//	enable_search = false;
					//}
				}
			}
		}

		if (n_put > n_get)
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
				if (n_put == n_get)
				{
					block_until_all_get = false;
				}

				update_pending_limit(root); //実験のためバッチサイズ変更が遅延しないようここでも処理
			}
		}
	}
	/*
	sync_cout << "info string thread " << thread_id() << " n_put " << n_put
		<< " leaf_dup " << leaf_dup << " leaf_mate_search_found " << leaf_mate_search_found << sync_endl;
	*/
}

#endif // USER_ENGINE_MCTS
