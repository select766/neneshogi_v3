#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "dnn_converter.h"

// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos_, istringstream& is)
{
}

#ifdef USER_ENGINE

CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::CPUDevice();
CNTK::FunctionPtr modelFunc;
shared_ptr<DNNConverter> cvt;

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
	modelFunc = CNTK::Function::Load(L"../../policy.cmf", device, CNTK::ModelFormat::CNTKv2);
	cvt = shared_ptr<DNNConverter>(new DNNConverter(0, 0));
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。
void MainThread::think()
{
	sync_cout << "info string start evaluation" << sync_endl;

	vector<float> inputData(85 * 9 * 9);
	cvt->get_board_array(rootPos, &inputData[0]);

	// Get input variable. The model has only one single input.
	CNTK::Variable inputVar = modelFunc->Arguments()[0];

	// The model has only one output.
	// If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
	CNTK::Variable outputVar = modelFunc->Output();


	// Create input value and input data map
	CNTK::ValuePtr inputVal = CNTK::Value::CreateBatch(inputVar.Shape(), inputData, device);
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputDataMap = { { inputVar, inputVal } };

	// Create output data map. Using null as Value to indicate using system allocated memory.
	// Alternatively, create a Value object and add it to the data map.
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputDataMap = { { outputVar, nullptr } };

	// Start evaluation on the device
	modelFunc->Evaluate(inputDataMap, outputDataMap, device);

	// Get evaluate result as dense output
	CNTK::ValuePtr outputVal = outputDataMap[outputVar];
	std::vector<std::vector<float>> outputData;
	outputVal->CopyVariableValueTo(outputVar, outputData);

	std::vector<float> &scores = outputData[0];

	Move bestMove = MOVE_RESIGN;
	float bestScore = -INFINITY;
	for (auto m : MoveList<LEGAL>(rootPos))
	{
		int dnn_index = cvt->get_move_index(rootPos, m.move);
		float score = scores[dnn_index];
		if (bestScore < score)
		{
			bestScore = score;
			bestMove = m.move;
		}
	}

	sync_cout << "info string bestscore " << bestScore << sync_endl;
	sync_cout << "bestmove " << bestMove << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
// この関数を呼び出したいときは、Thread::search()とすること。
void Thread::search()
{
}

#endif // USER_ENGINE
