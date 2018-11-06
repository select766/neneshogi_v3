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

// USI�ɒǉ��I�v�V������ݒ肵�����Ƃ��́A���̊֐����`���邱�ƁB
// USI::init()�̂Ȃ�����R�[���o�b�N�����B
void USI::extra_option(USI::OptionsMap & o)
{
	o["GPU"] << Option(-1, -1, 16);//�g�p����GPU�ԍ�(-1==CPU)
	o["format_board"] << Option(0, 0, 16);//DNN��board�\���`��
	o["format_move"] << Option(0, 0, 16);//DNN��move�\���`��
}

// �N�����ɌĂяo�����B���Ԃ̂�����Ȃ��T���֌W�̏����������͂����ɏ������ƁB
void Search::init()
{
}

// isready�R�}���h�̉������ɌĂяo�����B���Ԃ̂����鏈���͂����ɏ������ƁB
void  Search::clear()
{
	// �]���f�o�C�X�I��
	int gpu_id = (int)Options["GPU"];
	if (gpu_id >= 0)
	{
		device = CNTK::DeviceDescriptor::GPUDevice((unsigned int)gpu_id);
	}

	// ���f���̃��[�h
	// �{���̓t�@�C��������t�H�[�}�b�g�𐄘_������
	// ����������͓��{��Windows���ƃI�v�V������CP932�ŗ���Bmbstowcs�ɂ����F�������A���{��t�@�C�����𐳂����ϊ�
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

// �T���J�n���ɌĂяo�����B
// ���̊֐����ŏ��������I��点�Aslave�X���b�h���N������Thread::search()���Ăяo���B
// ���̂���slave�X���b�h���I�������A�x�X�g�Ȏw�����Ԃ����ƁB
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

// �T���{�́B���񉻂��Ă���ꍇ�A������slave�̃G���g���[�|�C���g�B
// MainThread::search()��virtual�ɂȂ��Ă���think()���Ăяo�����̂ŁAMainThread::think()����
// ���̊֐����Ăяo�������Ƃ��́AThread::search()�Ƃ��邱�ƁB
void Thread::search()
{
}

#endif // USER_ENGINE
