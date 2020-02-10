#include "../../extra/all.h"
#ifdef USER_ENGINE_POLICY

#include "CNTKLibrary.h"
#include "dnn_converter.h"
#include <numeric>
#include <functional>
#include <random>

void user_test(Position& pos_, istringstream& is)
{
}

static CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::CPUDevice();
static CNTK::FunctionPtr modelFunc;
static shared_ptr<DNNConverter> cvt;
static float softmaxTemperature = 0.0F;
static std::mt19937 mt;

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	o["GPU"] << Option(-1, -1, 16);//使用するGPU番号(-1==CPU)
	o["format_board"] << Option(0, 0, 16);//DNNのboard表現形式
	o["format_move"] << Option(0, 0, 16);//DNNのmove表現形式
	o["temperature"] << Option("1.0");//指し手決定のsoftmax temperature(0ならgreedy)
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
	std::random_device rd;
	mt.seed(rd());
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
	softmaxTemperature = (float)atof(((string)Options["temperature"]).c_str());
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
	float static_value = valueData[0][0] - valueData[0][1];

	std::vector<float> &policy_scores_raw = policyData[0];

	Move bestMove = MOVE_RESIGN;
	float bestScore = -INFINITY;
	// スコアを確率に換算し、サンプリング
	std::vector<float> scores;
	std::vector<Move> moves;
	for (auto m : MoveList<LEGAL>(rootPos))
	{
		moves.push_back(m.move);
		int dnn_index = cvt->get_move_index(rootPos, m.move);
		scores.push_back(policy_scores_raw[dnn_index]);
	}

	if (scores.size() > 0)
	{
		if (softmaxTemperature <= 0.0F)
		{
			// greedy
			for (size_t i = 0; i < scores.size(); i++)
			{
				float s = scores[i];
				if (s > bestScore)
				{
					bestScore = s;
					bestMove = moves[i];
				}
			}
		}
		else
		{
			// max scoreを引き、temperatureで割ってexp
			float maxvalue = *std::max_element(scores.begin(), scores.end());
			float exp_sum = 0.0F;
			std::vector<float> exps;
			for (size_t i = 0; i < scores.size(); i++)
			{
				float sexp = std::exp((scores[i] - maxvalue) / softmaxTemperature);
				exps.push_back(sexp);
				exp_sum += sexp;
			}

			std::vector<float> probs;
			for (size_t i = 0; i < scores.size(); i++)
			{
				probs.push_back(exps[i] / exp_sum);
			}

			// exps[i]/exp_sumの確率で指し手をサンプリング
			std::uniform_real_distribution<float> smpl(0.0, 1.0);
			float rndval = smpl(mt);//0.0~1.0の乱数
			bestMove = moves[0];//数値誤差でbreakしなかった場合の対策
			for (size_t i = 0; i < exps.size(); i++)
			{
				rndval -= probs[i];
				if (rndval <= 0.0F)
				{
					bestMove = moves[i];
					sync_cout << "info string sampled " << bestMove << " " << (int)(probs[i] * 100.0F) << "%" << sync_endl;
					break;
				}
			}
		}

		sync_cout << "info score cp " << (int)(static_value * 600.0F) << " pv " << bestMove << sync_endl;
	}
	else
	{
		bestMove = MOVE_RESIGN;
	}

	sync_cout << "bestmove " << bestMove << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
// この関数を呼び出したいときは、Thread::search()とすること。
void Thread::search()
{
}

#endif // USER_ENGINE
