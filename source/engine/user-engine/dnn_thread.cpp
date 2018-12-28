// CNTKでDNN評価を行うスレッド


#ifdef USER_ENGINE_MCTS
#include "../../extra/all.h"
#include "dnn_eval_obj.h"
#include "dnn_thread.h"

vector<DeviceModel> device_models;
float policy_temperature = 1.0;
float value_temperature = 1.0;
float value_scale = 1.0;
std::atomic_int n_dnn_thread_initalized = 0;

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

void dnn_thread_main(int worker_idx)
{
	sync_cout << "info string from dnn thread " << worker_idx << sync_endl;
	auto &device_model = device_models[worker_idx];
	auto &device = device_model.device;
	auto &modelFunc = device_model.modelFunc;
	
	auto input_shape = cvt->board_shape();
	int sample_size = accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
	
	sync_cout << "info string dnn batch size " << batch_size << sync_endl;

	int ctr = 0;
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
		sync_cout << "info string dnn batch=" << item_count << sync_endl;
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
		sync_cout << "info string dnn sent back" << sync_endl;
	}

}

#endif
