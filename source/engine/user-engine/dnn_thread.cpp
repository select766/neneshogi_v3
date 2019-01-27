// CNTKでDNN評価を行うスレッド


#ifdef USER_ENGINE_MCTS
#include "../../extra/all.h"
#include "dnn_eval_obj.h"
#include "dnn_thread.h"

vector<MTQueue<dnn_eval_obj*>*> request_queues;
static vector<std::thread*> dnn_threads;
DNNConverter *cvt = nullptr;
size_t batch_size = 0;
size_t n_gpu_threads = 0;//GPUスレッドの数(GPU数と必ずしも一致しない)
float policy_temperature = 1.0;
float value_temperature = 1.0;
float value_scale = 1.0;
static std::atomic_int n_dnn_thread_initalized = 0;
std::atomic_int n_dnn_evaled_samples = 0;
std::atomic_int n_dnn_evaled_batches = 0;

static void dnn_thread_main(size_t worker_idx, CNTK::DeviceDescriptor device, CNTK::FunctionPtr modelFunc);

void start_dnn_threads(string& evalDir, int format_board, int format_move, vector<int>& gpuIds)
{
	cvt = new DNNConverter(format_board, format_move);
	// モデルのロード
	// 本来はファイル名からフォーマットを推論したい
	// 将棋所からは日本語WindowsだとオプションがCP932で来る。mbstowcsにそれを認識させ、日本語ファイル名を正しく変換
	// デバイス数だけモデルをロードし各デバイスに割り当てる
	setlocale(LC_ALL, "");
	wchar_t model_path[1024];
	wchar_t evaldir_w[1024];
	size_t n_chars;
	if (mbstowcs_s(&n_chars, evaldir_w, sizeof(evaldir_w) / sizeof(evaldir_w[0]), evalDir.c_str(), _TRUNCATE) != 0)
	{
		throw runtime_error("Failed converting model path");
	}
	swprintf(model_path, sizeof(model_path) / sizeof(model_path[0]), L"%s/nene_%d_%d.cmf", evaldir_w, format_board, format_move);
	n_gpu_threads = gpuIds.size();
#ifdef MULTI_REQUEST_QUEUE
	for (size_t i = 0; i < n_gpu_threads; i++)
	{
		// リクエストキューをGPUスレッド分立てる
		request_queues.push_back(new MTQueue<dnn_eval_obj*>());
	}
#else
	// リクエストキューは1個だけ
	request_queues.push_back(new MTQueue<dnn_eval_obj*>());
#endif // MULTI_REQUEST_QUEUE

	// 評価スレッドを立てる
	for (size_t i = 0; i < gpuIds.size(); i++)
	{
		int gpu_id = gpuIds[i];
		CNTK::DeviceDescriptor device = gpu_id >= 0 ? CNTK::DeviceDescriptor::GPUDevice((unsigned int)gpu_id) : CNTK::DeviceDescriptor::CPUDevice();
		CNTK::FunctionPtr modelFunc = CNTK::Function::Load(model_path, device, CNTK::ModelFormat::CNTKv2);
		dnn_threads.push_back(new std::thread(dnn_thread_main, i, device, modelFunc));
	}

	// スレッドの動作開始(DNNの初期化)まで待つ
	while (n_dnn_thread_initalized < dnn_threads.size())
	{
		sleep(1);
	}
}

// CNTKによる評価。
// なぜか関数を分けないとスタック破壊らしき挙動が起きる。softmax計算がおかしな結果になる等。
static void do_eval(CNTK::FunctionPtr &modelFunc, CNTK::DeviceDescriptor &device, vector<float> &inputData, std::vector<std::vector<float>> &policyData, std::vector<std::vector<float>> &valueData)
{
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
	policyVal->CopyVariableValueTo(policyVar, policyData);
	CNTK::ValuePtr valueVal = outputDataMap[valueVar];
	valueVal->CopyVariableValueTo(valueVar, valueData);
}

static void dnn_thread_main(size_t worker_idx, CNTK::DeviceDescriptor device, CNTK::FunctionPtr modelFunc)
{
	sync_cout << "info string from dnn thread " << worker_idx << sync_endl;
	MTQueue<dnn_eval_obj*> *request_queue = request_queues[worker_idx % request_queues.size()];

	auto input_shape = cvt->board_shape();
	int sample_size = accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());

	vector<float> inputData(sample_size * batch_size);
	if (true)
	{
		// ダミー評価。対局中に初回の評価を行うと各種初期化が走って持ち時間をロスするため。
		std::vector<std::vector<float>> policyData;
		std::vector<std::vector<float>> valueData;
		do_eval(modelFunc, device, inputData, policyData, valueData);
	}

	n_dnn_thread_initalized.fetch_add(1);
	sync_cout << "info string dnn initialize ok" << sync_endl;

	dnn_eval_obj** eval_targets = new dnn_eval_obj*[batch_size];
	while (true)
	{
		size_t item_count = request_queue->pop_batch(eval_targets, batch_size);
#if 1
		// 実際のアイテム数で毎回バッチサイズを変える場合
		vector<float> inputData(sample_size * item_count);
#endif
		// eval_targetsをDNN評価
		for (size_t i = 0; i < item_count; i++)
		{
			memcpy(&inputData[sample_size*i], eval_targets[i]->input_array, sample_size * sizeof(float));
		}

		std::vector<std::vector<float>> policyData;
		std::vector<std::vector<float>> valueData;
		do_eval(modelFunc, device, inputData, policyData, valueData);

		for (size_t i = 0; i < item_count; i++)
		{
			dnn_eval_obj &eval_obj = *eval_targets[i];

			// 勝率=tanh(valueData[i][0] - valueData[i][1])
#ifdef EVAL_KPPT
			result_obj.static_value = eval_obj.static_value;
#else
			eval_obj.static_value = tanh((valueData[i][0] - valueData[i][1]) / value_temperature) * value_scale;
#endif

			// 合法手内でsoftmax確率を取る
			float raw_values[MAX_MOVES];
			float raw_max = -10000.0F;
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				raw_values[j] = policyData[i][eval_obj.move_indices[j].index];
				if (raw_max < raw_values[j])
				{
					raw_max = raw_values[j];
				}
			}
			float exps[MAX_MOVES];
			float exp_sum = 0.0F;
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				float e = std::exp((raw_values[j] - raw_max) / policy_temperature);//temperatureで割る
				exps[j] = e;
				exp_sum += e;
			}
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				eval_obj.move_indices[j].prob = exps[j] / exp_sum;
			}

			// response_queueに送り返す
			eval_obj.response_queue->push(&eval_obj);
		}

		n_dnn_evaled_batches.fetch_add(1);
		n_dnn_evaled_samples.fetch_add((int)item_count);
	}

}

#endif
