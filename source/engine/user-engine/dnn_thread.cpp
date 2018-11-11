// CNTKでDNN評価を行うスレッド

#include "../../extra/all.h"
#include "dnn_thread.h"
#include "dnn_eval_obj.h"
#include "ipqueue.h"

#ifdef USER_ENGINE_MCTS
vector<DeviceModel> device_models;
shared_ptr<DNNConverter> cvt;

void dnn_thread_main(int worker_idx)
{
	sync_cout << "info string from dnn thread " << worker_idx << sync_endl;
	auto &device_model = device_models[worker_idx];
	auto &device = CNTK::DeviceDescriptor::CPUDevice();// device_model.device;
	// auto &modelFunc = device_model.modelFunc;
	auto modelFunc = CNTK::Function::Load(L"D:\\dev\\shogi\\neneshogi\\data\\model\\train_147000000\\nene_0_1.cmf", device, CNTK::ModelFormat::CNTKv2);

	auto input_shape = cvt->board_shape();
	int sample_size = accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
	
	sync_cout << "info string dnn batch size " << eval_queue->batch_size() << sync_endl;

	// Get input variable. The model has only one single input.
	CNTK::Variable inputVar = modelFunc->Arguments()[0];

	// The model has only one output.
	// If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
	auto outputVars = modelFunc->Outputs();
	CNTK::Variable policyVar = outputVars[0];
	CNTK::Variable valueVar = outputVars[1];

	int ctr = 0;
	while (true)
	{
		ipqueue_item<dnn_eval_obj> *eval_objs;
		sync_cout << "info string dnn 1" << sync_endl;
		while (!(eval_objs = eval_queue->begin_read()))
		{
			std::this_thread::sleep_for(std::chrono::microseconds(1));
		}
		sync_cout << "info string dnn 2" << sync_endl;
		vector<float> inputData(sample_size * eval_queue->batch_size());
		// eval_objsをDNN評価
		for (int i = 0; i < eval_objs->count; i++)
		{
			memcpy(&inputData[sample_size*i], eval_objs->elements[i].input_array, sample_size * sizeof(float));
		}

		sync_cout << "info string dnn 3" << sync_endl;

		// Create input value and input data map
		CNTK::ValuePtr inputVal = CNTK::Value::CreateBatch(inputVar.Shape(), inputData, device);
		sync_cout << "info string dnn 3.5" << sync_endl;
		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputDataMap = { { inputVar, inputVal } };

		sync_cout << "info string dnn 3.7" << sync_endl;
		// Create output data map. Using null as Value to indicate using system allocated memory.
		// Alternatively, create a Value object and add it to the data map.
		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputDataMap = { { policyVar, nullptr }, { valueVar, nullptr } };

		sync_cout << "info string dnn 4" << sync_endl;
		// Start evaluation on the device
		modelFunc->Evaluate(inputDataMap, outputDataMap, device);

		sync_cout << "info string dnn 5" << sync_endl;
		// Get evaluate result as dense output
		CNTK::ValuePtr policyVal = outputDataMap[policyVar];
		std::vector<std::vector<float>> policyData;
		policyVal->CopyVariableValueTo(policyVar, policyData);
		CNTK::ValuePtr valueVal = outputDataMap[valueVar];
		std::vector<std::vector<float>> valueData;
		valueVal->CopyVariableValueTo(valueVar, valueData);

		sync_cout << "info string psize " << policyData.size() << "," << policyData[0].size() << sync_endl;
		sync_cout << "info string vsize " << valueData.size() << "," << valueData[0].size() << sync_endl;

		sync_cout << "info string dnn 6" << sync_endl;
		ipqueue_item<dnn_result_obj> *result_objs;
		while (!(result_objs = result_queue->begin_write()))
		{
			std::this_thread::sleep_for(std::chrono::microseconds(1));
		}

		sync_cout << "info string dnn 7" << sync_endl;
		for (int i = 0; i < eval_objs->count; i++)
		{
			dnn_eval_obj &eval_obj = eval_objs->elements[i];
			dnn_result_obj &result_obj = result_objs->elements[i];
			result_obj.static_value = (int16_t)((valueData[i][0] - 0.5) * 32000);//sigmoidで0~1で出てくる？

			// 合法手内でsoftmax確率を取る
			sync_cout << "info string vals[" << eval_obj.n_moves << "] ";
			float raw_values[MAX_MOVES];
			float raw_max = -10000.0F;
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				raw_values[j] = policyData[i][eval_obj.move_indices[j].index];
				std::cout << eval_obj.move_indices[j].index << "=" << raw_values[j] << ",";
				if (raw_max < raw_values[j])
				{
					raw_max = raw_values[j];
				}
			}
			std::cout << sync_endl;
			float exps[MAX_MOVES];
			float exp_sum = 0.0F;
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				float e = std::exp((raw_values[j] - raw_max) / 1.0F);//temperatureで割る
				exps[j] = e;
				exp_sum += e;
			}
			//sync_cout << "info string probs[" << eval_obj.n_moves << "] ";
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				result_obj.move_probs[j].move = eval_obj.move_indices[j].move;
				result_obj.move_probs[j].prob_scaled = (uint16_t)((exps[j] / exp_sum) * 65535);
				//std::cout << (Move)result_obj.move_probs[j].move << "=" << result_obj.move_probs[j].prob_scaled << ",";
			}
			//std::cout << sync_endl;
			result_obj.index = eval_obj.index;
			result_obj.n_moves = eval_obj.n_moves;
		}
		result_objs->count = eval_objs->count;

		sync_cout << "info string dnn 8" << sync_endl;
		// reqult_queueに結果を書く
		eval_queue->end_read();
		result_queue->end_write();
		sync_cout << "info string dnn 9" << sync_endl;
	}

}
#endif
