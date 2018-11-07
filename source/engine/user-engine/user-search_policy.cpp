#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "dnn_converter.h"
#include <numeric>
#include <functional>

#ifdef USER_ENGINE_POLICY

void user_test(Position& pos_, istringstream& is)
{
}

CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::CPUDevice();
CNTK::FunctionPtr modelFunc;
shared_ptr<DNNConverter> cvt;

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	o["GPU"] << Option(-1, -1, 16);//使用するGPU番号(-1==CPU)
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
	// 評価デバイス選択
	int gpu_id = (int)Options["GPU"];
	if (gpu_id >= 0)
	{
		device = CNTK::DeviceDescriptor::GPUDevice((unsigned int)gpu_id);
	}

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
	modelFunc = CNTK::Function::Load(model_path, device, CNTK::ModelFormat::CNTKv2);
	cvt = shared_ptr<DNNConverter>(new DNNConverter(format_board, format_move));
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。
void MainThread::think()
{
	sync_cout << "info string start evaluation" << sync_endl;

	auto input_shape = cvt->board_shape();
	vector<float> inputData(accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>()));
	cvt->get_board_array(rootPos, &inputData[0]);

	// Get input variable. The model has only one single input.
	CNTK::Variable inputVar = modelFunc->Arguments()[0];

	// The model has only one output.
	// If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
	auto outputVars = modelFunc->Outputs();
	CNTK::Variable policyVar = outputVars[0];
	CNTK::Variable valueVar = outputVars[1];


	// Create input value and input data map
	CNTK::ValuePtr inputVal = CNTK::Value::CreateBatch(inputVar.Shape(), inputData, device);
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputDataMap = { { inputVar, inputVal } };

	// Create output data map. Using null as Value to indicate using system allocated memory.
	// Alternatively, create a Value object and add it to the data map.
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputDataMap = { { policyVar, nullptr }, { valueVar, nullptr } };

	// Start evaluation on the device
	modelFunc->Evaluate(inputDataMap, outputDataMap, device);

	// Get evaluate result as dense output
	CNTK::ValuePtr policyVal = outputDataMap[policyVar];
	std::vector<std::vector<float>> policyData;
	policyVal->CopyVariableValueTo(policyVar, policyData);
	CNTK::ValuePtr valueVal = outputDataMap[valueVar];
	std::vector<std::vector<float>> valueData;
	valueVal->CopyVariableValueTo(valueVar, valueData);
	float static_value = valueData[0][0];

	std::vector<float> &policy_scores = policyData[0];

	Move bestMove = MOVE_RESIGN;
	float bestScore = -INFINITY;
	for (auto m : MoveList<LEGAL>(rootPos))
	{
		int dnn_index = cvt->get_move_index(rootPos, m.move);
		float score = policy_scores[dnn_index];
		if (bestScore < score)
		{
			bestScore = score;
			bestMove = m.move;
		}
	}

	sync_cout << "info string bestscore " << bestScore << " static_value " << static_value << sync_endl;
	sync_cout << "bestmove " << bestMove << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
// この関数を呼び出したいときは、Thread::search()とすること。
void Thread::search()
{
}

#endif // USER_ENGINE
