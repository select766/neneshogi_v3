// nenefwd.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"

using namespace std;

const int INPUT_COUNT = 119 * 9 * 9;
const int INPUT_BYTE_LENGTH = INPUT_COUNT * 4;
const int OUTPUT_POLICY_COUNT = 27 * 9 * 9;
const int OUTPUT_VALUE_COUNT = 2;
const int OUTPUT_BYTE_LENGTH = (OUTPUT_POLICY_COUNT + OUTPUT_VALUE_COUNT) * 4;
const int FORMAT_BOARD = 1;
const int FORMAT_MOVE = 1;

CNTK::FunctionPtr load_model(const char* model_dir, int gpu_id, CNTK::DeviceDescriptor &device);
SOCKET connect_server(const char* hostname, int port);
bool read_batch(SOCKET &sock, vector<float> &data);
void write_result(SOCKET &sock, vector<vector<float>> &policyData, vector<vector<float>> &valueData);

int main2()
{
	int gpu_id = 0;
	CNTK::DeviceDescriptor device = gpu_id >= 0 ? CNTK::DeviceDescriptor::GPUDevice((unsigned int)gpu_id) : CNTK::DeviceDescriptor::CPUDevice();
	// nenefwd.exe model_dir gpu_id hostname port
	CNTK::FunctionPtr modelFunc = CNTK::Function::Load(L"D:\\dev\\shogi\\neneshogi\\data\\model\\20181226\\resnetaz_128_39_fb1\\checkpoint\\train_170000128\\nene_1_1.cmf", device, CNTK::ModelFormat::CNTKv2);

	vector<float> inputData(119*9*9);
	while (true)
	{
		std::cerr << "read size=" << inputData.size() << std::endl;
		// do inference
		std::vector<std::vector<float>> policyData;
		std::vector<std::vector<float>> valueData;
		// Get input variable. The model has only one single input.
		CNTK::Variable inputVar = modelFunc->Arguments()[0];

		// The model has only one output.
		// If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
		auto outputVars = modelFunc->Outputs();
		CNTK::Variable policyVar = outputVars[0];
		CNTK::Variable valueVar = outputVars[1];

		// Create input value and input data map
		CNTK::ValuePtr inputVal = CNTK::Value::CreateBatch(inputVar.Shape(), inputData, device);
		std::cerr << "created batch" << std::endl;
		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputDataMap = { { inputVar, inputVal } };

		// Create output data map. Using null as Value to indicate using system allocated memory.
		// Alternatively, create a Value object and add it to the data map.
		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputDataMap = { { policyVar, nullptr }, { valueVar, nullptr } };

		// Start evaluation on the device
		modelFunc->Evaluate(inputDataMap, outputDataMap, device);
		std::cerr << "evaluated" << std::endl;

		// Get evaluate result as dense output
		CNTK::ValuePtr policyVal = outputDataMap[policyVar];
		policyVal->CopyVariableValueTo(policyVar, policyData);
		CNTK::ValuePtr valueVal = outputDataMap[valueVar];
		valueVal->CopyVariableValueTo(valueVar, valueData);

		std::cerr << "writtten" << std::endl;
		break;
	}

	return 0;
}

int main(int argc, const char** argv)
{
	//main2();
	//return 0;
	if (argc != 5)
	{
		throw runtime_error("nenefwd.exe model_dir gpu_id hostname port");
	}
	CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::CPUDevice();
	// nenefwd.exe model_dir gpu_id hostname port
	CNTK::FunctionPtr modelFunc = load_model(argv[1], strtol(argv[2], nullptr, 10), device);

	SOCKET sock = connect_server(argv[3], strtol(argv[4], nullptr, 10));
	//std::cerr << "connected" << std::endl;

	vector<float> inputData;
	while (read_batch(sock, inputData))
	{
		//std::cerr << "read size=" << inputData.size() << std::endl;
		// do inference
		std::vector<std::vector<float>> policyData;
		std::vector<std::vector<float>> valueData;
		// Get input variable. The model has only one single input.
		CNTK::Variable inputVar = modelFunc->Arguments()[0];

		// The model has only one output.
		// If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
		auto outputVars = modelFunc->Outputs();
		CNTK::Variable policyVar = outputVars[0];
		CNTK::Variable valueVar = outputVars[1];

		// Create input value and input data map
		CNTK::ValuePtr inputVal = CNTK::Value::CreateBatch(inputVar.Shape(), inputData, device);
		//std::cerr << "created batch" << std::endl;
		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputDataMap = { { inputVar, inputVal } };

		// Create output data map. Using null as Value to indicate using system allocated memory.
		// Alternatively, create a Value object and add it to the data map.
		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputDataMap = { { policyVar, nullptr }, { valueVar, nullptr } };

		// Start evaluation on the device
		modelFunc->Evaluate(inputDataMap, outputDataMap, device);
		//std::cerr << "evaluated" << std::endl;

		// Get evaluate result as dense output
		CNTK::ValuePtr policyVal = outputDataMap[policyVar];
		policyVal->CopyVariableValueTo(policyVar, policyData);
		CNTK::ValuePtr valueVal = outputDataMap[valueVar];
		valueVal->CopyVariableValueTo(valueVar, valueData);

		write_result(sock, policyData, valueData);
		//std::cerr << "writtten" << std::endl;
	}

	//std::cerr << "closing" << std::endl;
	closesocket(sock);

	WSACleanup();
}

CNTK::FunctionPtr load_model(const char* model_dir, int gpu_id, CNTK::DeviceDescriptor &device)
{
	setlocale(LC_ALL, "");
	wchar_t model_path[1024];
	wchar_t evaldir_w[1024];
	size_t n_chars;
	if (mbstowcs_s(&n_chars, evaldir_w, sizeof(evaldir_w) / sizeof(evaldir_w[0]), model_dir, _TRUNCATE) != 0)
	{
		throw runtime_error("Failed converting model path");
	}
	swprintf(model_path, sizeof(model_path) / sizeof(model_path[0]), L"%s/nene_%d_%d.cmf", evaldir_w, FORMAT_BOARD, FORMAT_MOVE);
	device = gpu_id >= 0 ? CNTK::DeviceDescriptor::GPUDevice((unsigned int)gpu_id) : CNTK::DeviceDescriptor::CPUDevice();
	CNTK::FunctionPtr modelFunc = CNTK::Function::Load(model_path, device, CNTK::ModelFormat::CNTKv2);

	return modelFunc;
}

SOCKET connect_server(const char* hostname, int port)
{
	WSADATA wsaData;
	int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (err != 0)
	{
		throw runtime_error("Failed WSAStartup");
	}

	SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == INVALID_SOCKET)
	{
		throw runtime_error("Failed socket");
	}

	// TCP_NODELAYを有効化して、最後のパケットがさっさと出るようにする（効果があるかは不明）
	int flag = 1;
	if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(flag)) != 0)
	{
		throw runtime_error("Failed TCP_NODELAY");
	}
	struct sockaddr_in server;
	server.sin_family = AF_INET;
	server.sin_port = htons((unsigned short)port);
	inet_pton(server.sin_family, hostname, &server.sin_addr.S_un.S_addr);


	if (connect(sock, (struct sockaddr *)&server, sizeof(server)) != 0)
	{
		throw runtime_error("Failed connect");
	}

	return sock;
}

bool read_batch(SOCKET &sock, vector<float> &data)
{
	// バッチサイズ取得
	int batch_size;
	int batch_size_received_size = 0;
	// 最悪batch_sizeの4バイトが分割されてしまうこともあるので一応whileで読む
	while (batch_size_received_size < sizeof(batch_size))
	{
		int n = recv(sock, ((char*)&batch_size) + batch_size_received_size, sizeof(batch_size) - batch_size_received_size, 0);
		if (n == 0)
		{
			// 正常切断
			return false;
		}
		if (n < 0)
		{
			// エラー切断
			throw runtime_error("Failed recv");
		}
		batch_size_received_size += n;
	}

	// データを全部バッファに読み込む
	int expect_count = INPUT_COUNT * batch_size;
	int expect_byte_length = INPUT_BYTE_LENGTH * batch_size;
	data.resize(expect_count);
	int received_size = 0;
	while (received_size < expect_byte_length)
	{
		int n = recv(sock, (char*)&data[0] + received_size, expect_byte_length - received_size, 0);
		if (n == 0)
		{
			// 正常切断
			throw runtime_error("Disconnected while reading content");
		}
		if (n < 0)
		{
			// エラー切断
			throw runtime_error("Failed recv");
		}

		received_size += n;
	}

	return true;
}

void write_result(SOCKET &sock, vector<vector<float>> &policyData, vector<vector<float>> &valueData)
{
	// 送信データにまとめる
	int batch_size = (int)policyData.size();
	int send_data_byte_length = sizeof(batch_size) + OUTPUT_BYTE_LENGTH * batch_size;
	char* send_data = new char[send_data_byte_length];

	*((int*)&send_data[0]) = batch_size;
	float* send_data_cursor = (float*)&send_data[sizeof(batch_size)];
	for (size_t i = 0; i < (size_t)batch_size; i++)
	{
		vector<float> &policySample = policyData[i];
		memcpy(send_data_cursor, &policySample[0], OUTPUT_POLICY_COUNT * sizeof(float));
		send_data_cursor += OUTPUT_POLICY_COUNT;
		vector<float> &valueSample = valueData[i];
		memcpy(send_data_cursor, &valueSample[0], OUTPUT_VALUE_COUNT * sizeof(float));
		send_data_cursor += OUTPUT_VALUE_COUNT;
	}

	int sent_byte_length = 0;
	while (sent_byte_length < send_data_byte_length)
	{
		int sent_size = send(sock, send_data + sent_byte_length, send_data_byte_length - sent_byte_length, 0);
		if (sent_size < 0)
		{
			throw runtime_error("Failed send");
		}
		sent_byte_length += sent_size;
	}

	delete[] send_data;
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
