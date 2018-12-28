#ifdef USER_ENGINE_MCTS
#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "mcts.h"
#include "dnn_thread.h"

MTQueue<dnn_eval_obj*> *request_queue = nullptr;
static MCTS *mcts = nullptr;
DNNConverter *cvt = nullptr;
static MTQueue<dnn_eval_obj*> **response_queues;
int batch_size;
static vector<std::thread*> dnn_threads;

// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos_, istringstream& is)
{
}


// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	o["batch_size"] << Option(16, 1, 65536);
	o["GPU"] << Option("-1");//使用するGPU番号(-1==CPU)、カンマ区切りで複数指定可能
	o["format_board"] << Option(0, 0, 16);//DNNのboard表現形式
	o["format_move"] << Option(0, 0, 16);//DNNのmove表現形式
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
	request_queue = new MTQueue<dnn_eval_obj*>();
	mcts = new MCTS(1024 * 1024);
	batch_size = (int)Options["batch_size"];

	sync_cout << "info string initializing dnn threads" << sync_endl;
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
	cvt =  new DNNConverter(format_board, format_move);

	// スレッド間キュー初期化
	request_queue = new MTQueue<dnn_eval_obj*>();
	int threads = (int)Options["Threads"];
	response_queues = new MTQueue<dnn_eval_obj*>*[threads];
	for (int i = 0; i < threads; i++)
	{
		response_queues[i] = new MTQueue<dnn_eval_obj*>();
	}

	// 評価スレッドを立てる
	for (int i = 0; i < device_models.size(); i++)
	{
		dnn_threads.push_back(new std::thread(dnn_thread_main, i));
	}

	// スレッドの動作開始(DNNの初期化)まで待つ
	while (n_dnn_thread_initalized < dnn_threads.size())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	sync_cout << "info string initialized all dnn threads" << sync_endl;
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
	MCTSSearchInfo sei;
	sei.request_queue = request_queue;
	sei.response_queue = response_queues[0];
	sei.cvt = cvt;
	dnn_eval_obj *eobj = new dnn_eval_obj();
	bool created;
	UCTNode *root = mcts->make_root(rootPos, sei, eobj, created);
	sync_cout << "info string created root " << created << " dnn " << sei.put_dnn_eval << sync_endl;
	if (sei.put_dnn_eval)
	{
		dnn_eval_obj *sentback;
		sei.response_queue->pop(sentback);
		sync_cout << "info string put " << eobj << " sentback " << sentback << sync_endl;
		mcts->backup_dnn(sentback);
		delete sentback;
		root->pprint();
	}
	else
	{
		delete eobj;
	}

	Move bestMove = mcts->get_bestmove(root, rootPos);
	sync_cout << "bestmove " << bestMove << sync_endl;

}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
// この関数を呼び出したいときは、Thread::search()とすること。
void Thread::search()
{
}

#endif // USER_ENGINE_MCTS
