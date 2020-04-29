// CNTKでDNN評価を行うスレッド

#include "../../extra/all.h"
#ifdef USER_ENGINE_MCTS
#include "dnn_eval_obj.h"
#include "dnn_thread.h"

vector<MTQueue<dnn_eval_obj *> *> request_queues;
static vector<std::thread *> dnn_threads;
DNNConverter *cvt = nullptr;
size_t batch_size = 0;
size_t n_gpu_threads = 0; //GPUスレッドの数(GPU数と必ずしも一致しない)
float policy_temperature = 1.0;
float value_temperature = 1.0;
float value_scale = 1.0;
static std::atomic_uint n_dnn_thread_initalized(0);
static std::atomic_bool all_dnn_thread_initialized(false);
std::atomic_int n_dnn_evaled_samples(0);
std::atomic_int n_dnn_evaled_batches(0);

#ifdef DNN_EXTERNAL
#ifdef _WIN64
#include <WinSock2.h>
#include <WS2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define BOOL int
#endif

const int port_offset = 25250;
static void dnn_thread_main(size_t worker_idx, string evalDir, int gpu_id, int port);

#ifdef _WIN64
static bool wsa_startup()
{
	WSADATA wsaData;
	int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (err != 0)
	{
		sync_cout << "info string failed WSAStartup" << sync_endl;
		return false;
	}
	return true;
}
#else
static bool wsa_startup()
{
	return true;
}
#endif

void start_dnn_threads(string &evalDir, int format_board, int format_move, vector<int> &gpuIds)
{
	if (!wsa_startup())
	{
		return;
	}
	cvt = new DNNConverter(format_board, format_move);
	// デバイス数だけ評価exeを立てる
	n_gpu_threads = gpuIds.size();
#ifdef MULTI_REQUEST_QUEUE
	for (size_t i = 0; i < n_gpu_threads; i++)
	{
		// リクエストキューをGPUスレッド分立てる
		request_queues.push_back(new MTQueue<dnn_eval_obj *>());
	}
#else
	// リクエストキューは1個だけ
	request_queues.push_back(new MTQueue<dnn_eval_obj *>());
#endif // MULTI_REQUEST_QUEUE

	// 評価スレッドを立てる
	for (size_t i = 0; i < gpuIds.size(); i++)
	{
		int gpu_id = gpuIds[i];
		dnn_threads.push_back(new std::thread(dnn_thread_main, i, evalDir, gpu_id, port_offset + (int)i));
	}

	// スレッドの動作開始(DNNの初期化)まで待つ
	while (n_dnn_thread_initalized < dnn_threads.size())
	{
		sleep(1);
	}
}

static SOCKET start_listen(size_t worker_idx, int *port)
{
	// portにはポート番号の初期値を設定する。もし使用されていたらインクリメントされ、実際に確保されたポート番号が得られる。
	SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, 0);
	if (listen_sock == INVALID_SOCKET)
	{
		sync_cout << "info string failed socket worker=" << worker_idx << sync_endl;
		return INVALID_SOCKET;
	}
	// TCP_NODELAYを有効化して、最後のパケットがさっさと出るようにする（効果があるかは不明）
	int flag = 1;
	if (setsockopt(listen_sock, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(flag)) != 0)
	{
		sync_cout << "info string failed TCP_NODELAY worker=" << worker_idx << sync_endl;
		return INVALID_SOCKET;
	}

	// プロセスを再起動したらすぐポートを再利用できるようにする
	// -> 生きているプロセスが同じポートをbindしてもエラーにならず、自己対戦に支障
	//BOOL yes = 1;
	//setsockopt(listen_sock,
	//		   SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));

	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(struct sockaddr_in));
	addr.sin_family = AF_INET;
#ifdef _WIN64
	addr.sin_addr.S_un.S_addr = INADDR_ANY;
#else
	addr.sin_addr.s_addr = INADDR_ANY;
#endif
	for (int retry = 0; retry < 100; retry++)
	{
		addr.sin_port = htons(*port);
		if (::bind(listen_sock, (struct sockaddr*) & addr, sizeof(addr)) != 0)
		{
#ifdef _WIN64
			int socket_error_code = WSAGetLastError();
			int addrinuse = WSAEADDRINUSE;
#else
			int socket_error_code = errno;
			int addrinuse = EADDRINUSE;
#endif

			if (socket_error_code == addrinuse)
			{
				// 使用されているポート番号
				(*port)++;
				continue;
			}
			else
			{
				sync_cout << "info string failed bind worker=" << worker_idx << "," << socket_error_code << sync_endl;
				return INVALID_SOCKET;
			}
		}

		if (listen(listen_sock, 1) != 0)
		{
#ifdef _WIN64
			int socket_error_code = WSAGetLastError();
			int addrinuse = WSAEADDRINUSE;
#else
			int socket_error_code = errno;
			int addrinuse = EADDRINUSE;
#endif

			if (socket_error_code == addrinuse)
			{
				// 使用されているポート番号
				(*port)++;
				continue;
			}
			else
			{
				sync_cout << "info string failed listen worker=" << worker_idx << "," << socket_error_code << sync_endl;
				return INVALID_SOCKET;
			}
		}

		return listen_sock;
	}

	return INVALID_SOCKET;
}

static SOCKET do_accept(size_t worker_idx, SOCKET listen_sock)
{
	struct sockaddr_in client;
#ifdef _WIN64
	int len = sizeof(client);
#else
	socklen_t len = sizeof(client);
#endif
	SOCKET sock = accept(listen_sock, (struct sockaddr *)&client, &len);
	if (sock == INVALID_SOCKET)
	{
		sync_cout << "info string failed accept worker=" << worker_idx << sync_endl;
		return INVALID_SOCKET;
	}

	return sock;
}

static const int INPUT_COUNT = 119 * 9 * 9;
static const int INPUT_BYTE_LENGTH = INPUT_COUNT * 4;
static const int OUTPUT_POLICY_COUNT = 27 * 9 * 9;
static const int OUTPUT_VALUE_COUNT = 2;
static const int OUTPUT_BYTE_LENGTH = (OUTPUT_POLICY_COUNT + OUTPUT_VALUE_COUNT) * 4;
static const int FORMAT_BOARD = 1;
static const int FORMAT_MOVE = 1;

static bool write_batch(SOCKET client_sock, vector<float> &inputData)
{
	int batch_size = (int)inputData.size() / INPUT_COUNT;
	int send_data_byte_length = sizeof(batch_size) + INPUT_BYTE_LENGTH * batch_size;
	char *send_data = new char[send_data_byte_length];

	*((int *)&send_data[0]) = batch_size;
	float *send_data_cursor = (float *)&send_data[sizeof(batch_size)];
	memcpy(send_data_cursor, &inputData[0], INPUT_BYTE_LENGTH * batch_size);

	int sent_byte_length = 0;
	while (sent_byte_length < send_data_byte_length)
	{
		int sent_size = send(client_sock, send_data + sent_byte_length, send_data_byte_length - sent_byte_length, 0);
		if (sent_size < 0)
		{
			return false;
		}
		sent_byte_length += sent_size;
	}

	delete[] send_data;
	return true;
}

static bool read_result(SOCKET client_sock, std::vector<std::vector<float>> &policyData, std::vector<std::vector<float>> &valueData)
{
	// バッチサイズ取得
	int batch_size;
	int batch_size_received_size = 0;
	// 最悪batch_sizeの4バイトが分割されてしまうこともあるので一応whileで読む
	while (batch_size_received_size < sizeof(batch_size))
	{
		int n = recv(client_sock, ((char *)&batch_size) + batch_size_received_size, sizeof(batch_size) - batch_size_received_size, 0);
		if (n == 0)
		{
			// 正常切断
			return false;
		}
		if (n < 0)
		{
			// エラー切断
			return false;
		}
		batch_size_received_size += n;
	}

	// データを全部バッファに読み込む
	int expect_byte_length = OUTPUT_BYTE_LENGTH * batch_size;
	char *raw_recv_data = new char[expect_byte_length];
	int received_size = 0;
	while (received_size < expect_byte_length)
	{
		int n = recv(client_sock, raw_recv_data + received_size, expect_byte_length - received_size, 0);
		if (n == 0)
		{
			// 正常切断
			return false;
		}
		if (n < 0)
		{
			// エラー切断
			return false;
		}

		received_size += n;
	}

	// パースする
	float *recv_data_cursor = (float *)raw_recv_data;
	policyData.resize(batch_size);
	valueData.resize(batch_size);
	for (size_t i = 0; i < (size_t)batch_size; i++)
	{
		vector<float> &pd = policyData[i];
		pd.resize(OUTPUT_POLICY_COUNT);
		memcpy(&pd[0], recv_data_cursor, OUTPUT_POLICY_COUNT * 4);
		recv_data_cursor += OUTPUT_POLICY_COUNT;
		vector<float> &vd = valueData[i];
		vd.resize(OUTPUT_VALUE_COUNT);
		memcpy(&vd[0], recv_data_cursor, OUTPUT_VALUE_COUNT * 4);
		recv_data_cursor += OUTPUT_VALUE_COUNT;
	}

	delete[] raw_recv_data;

	return true;
}

// ソケットでつながった外部プロセスでの評価
static bool do_eval(SOCKET client_sock, vector<float> &inputData, std::vector<std::vector<float>> &policyData, std::vector<std::vector<float>> &valueData)
{
	if (!write_batch(client_sock, inputData))
	{
		sync_cout << "info string failed socket write" << sync_endl;
		return false;
	}
	if (!read_result(client_sock, policyData, valueData))
	{
		sync_cout << "info string failed socket read" << sync_endl;
		return false;
	}
	return true;
}

static void dnn_thread_main(size_t worker_idx, string evalDir, int gpu_id, int port)
{
	sync_cout << "info string from dnn thread " << worker_idx << sync_endl;
	MTQueue<dnn_eval_obj *> *request_queue = request_queues[worker_idx % request_queues.size()];

	// TCP listen開始
	SOCKET listen_sock = start_listen(worker_idx, &port);
	if (listen_sock == INVALID_SOCKET)
	{
		return;
	}

	// 子プロセスを立てて接続を待つ
	// 非常に単純に、system関数を実行するだけのスレッドを立ててしまう
	auto system_thread = std::thread([evalDir, gpu_id, port, worker_idx] {
		string dnn_system_command("");
#ifdef _WIN64
		dnn_system_command += "nenefwd";
#else
		dnn_system_command += "./nenefwd";
#endif
		dnn_system_command += " ";
		dnn_system_command += evalDir;
		dnn_system_command += " ";
		dnn_system_command += std::to_string(gpu_id);
		dnn_system_command += " ";
		dnn_system_command += "127.0.0.1";
		dnn_system_command += " ";
		dnn_system_command += std::to_string(port);
		if (system(dnn_system_command.c_str()) == 0)
		{
			sync_cout << "info string exited dnn process " << worker_idx << sync_endl;
		}
		else
		{
			sync_cout << "info string failed dnn process " << worker_idx << sync_endl;
		}
	});
	system_thread.detach();
	SOCKET client_sock = do_accept(worker_idx, listen_sock);
	if (listen_sock == INVALID_SOCKET)
	{
		return;
	}
	sync_cout << "info string connected from dnn process " << worker_idx << sync_endl;

	// 評価

	auto input_shape = cvt->board_shape();
	int sample_size = accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());

	vector<float> inputData(sample_size * batch_size);
	if (true)
	{
		// ダミー評価。対局中に初回の評価を行うと各種初期化が走って持ち時間をロスするため。
		std::vector<std::vector<float>> policyData;
		std::vector<std::vector<float>> valueData;
		if (!do_eval(client_sock, inputData, policyData, valueData))
		{
			return;
		}
	}

	n_dnn_thread_initalized.fetch_add(1);
	sync_cout << "info string dnn initialize ok" << sync_endl;

	dnn_eval_obj **eval_targets = new dnn_eval_obj *[batch_size];
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
			memcpy(&inputData[sample_size * i], eval_targets[i]->input_array, sample_size * sizeof(float));
		}

		std::vector<std::vector<float>> policyData;
		std::vector<std::vector<float>> valueData;
		if (!do_eval(client_sock, inputData, policyData, valueData))
		{
			return;
		}

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
				float e = std::exp((raw_values[j] - raw_max) / policy_temperature); //temperatureで割る
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
#else

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <mutex>
#include <thread>
#include <random>
#include <atomic>
#include <chrono>

#include "tensorrt/common.h"
#include "tensorrt/buffers.h"
#include "dnn_engine_info.h"

static const int INPUT_COUNT = 119 * 9 * 9;
static const int INPUT_BYTE_LENGTH = INPUT_COUNT * 4;
static const int OUTPUT_POLICY_COUNT = 27 * 9 * 9;
static const int OUTPUT_VALUE_COUNT = 2;
static const int FORMAT_BOARD = 1;
static const int FORMAT_MOVE = 1;

static void dnn_thread_main(size_t worker_idx, int device, int threadInDevice, const char *evalDir);

static std::string addProfileSuffix(const std::string &name, int profile)
{
	std::ostringstream oss;
	oss << name;
	if (profile > 0)
	{
		oss << " [profile " << profile << "]";
	}

	return oss.str();
}


class ShogiOnnxExec
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	DNNEngineInfo engineInfo;
	ShogiOnnxExec()
		: mEngine(nullptr)
	{
	}

	//!
	//! \brief Function deserialize the network engine from file
	//!
	bool load(const char *evalDir);

	//!
	//! \brief Runs the TensorRT inference engine for this sample
	//!
	bool infer(int batchSize, float *inputData, float *outputPolicyData, float *outputValueData);

private:
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
	std::map<int, std::shared_ptr<nvinfer1::IExecutionContext>> mContextForProfile;
	nvinfer1::Dims mInputDims;		  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputPolicyDims; //!< The dimensions of the output to the network.
	nvinfer1::Dims mOutputValueDims;  //!< The dimensions of the output to the network.

	bool processInput(const samplesCommon::BufferManager &buffers, int batchSize, float *inputData);
	bool processOutput(const samplesCommon::BufferManager &buffers, int batchSize, float *outputPolicyData, float *outputValueData);
};

bool ShogiOnnxExec::load(const char *evalDir)
{
	string engineInfoPath(evalDir);
	engineInfoPath.append("/info.bin");
	ifstream engineInfoFile(engineInfoPath, ios::in | ios::binary);
	engineInfoFile.read((char *)&engineInfo, sizeof(engineInfo));
	if (!engineInfoFile)
	{
		return false;
	}

	string enginePath(evalDir);
	enginePath.append("/engine.bin");
	ifstream serializedModelFile(enginePath, ios::in | ios::binary);
	std::vector<char> fdata(engineInfo.serializedEngineSize);
	serializedModelFile.read((char *)fdata.data(), engineInfo.serializedEngineSize);
	if (!serializedModelFile)
	{
		return false;
	}

	auto runtime = createInferRuntime(gLogger);
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(fdata.data(), engineInfo.serializedEngineSize, nullptr), samplesCommon::InferDeleter());

	mInputDims = Dims4{engineInfo.inputDims[0], engineInfo.inputDims[1], engineInfo.inputDims[2], engineInfo.inputDims[3]};
	mOutputPolicyDims = Dims2{engineInfo.outputPolicyDims[0], engineInfo.outputPolicyDims[1]};
	mOutputValueDims = Dims2{engineInfo.outputValueDims[0], engineInfo.outputValueDims[1]};

	// different context for each profile is needed (switching causes error on setBindingDimensions)
	for (int i = 0; i < mEngine->getNbOptimizationProfiles(); i++)
	{
		auto ctx = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(), samplesCommon::InferDeleter());
		if (!ctx)
		{
			return false;
		}
		ctx->setOptimizationProfile(i);
		mContextForProfile[i] = ctx;
	}

	return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool ShogiOnnxExec::infer(int batchSize, float *inputData, float *outputPolicyData, float *outputValueData)
{
	auto mContext = mContextForProfile.at(engineInfo.profileForBatchSize[batchSize]);
	std::string inputBindingName = addProfileSuffix(engineInfo.inputTensorName, engineInfo.profileForBatchSize[batchSize]);
	int bidx = mEngine->getBindingIndex(inputBindingName.c_str());
	mContext->setBindingDimensions(bidx, Dims4{batchSize, engineInfo.inputDims[1], engineInfo.inputDims[2], engineInfo.inputDims[3]});
	// Create RAII buffer manager object
	samplesCommon::BufferManager buffers(mEngine, batchSize, mContext.get());

	// Read the input data into the managed buffers
	if (!processInput(buffers, batchSize, inputData))
	{
		return false;
	}

	// Memcpy from host input buffers to device input buffers
	buffers.copyInputToDevice();

	bool status = mContext->executeV2(buffers.getDeviceBindings().data());
	if (!status)
	{
		return false;
	}

	// Memcpy from device output buffers to host output buffers
	buffers.copyOutputToHost();

	// Read results
	if (!processOutput(buffers, batchSize, outputPolicyData, outputValueData))
	{
		return false;
	}

	return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool ShogiOnnxExec::processInput(const samplesCommon::BufferManager &buffers, int batchSize, float *inputData)
{
	std::string inputName = addProfileSuffix(engineInfo.inputTensorName, engineInfo.profileForBatchSize[batchSize]);
	float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(inputName));
	memcpy(hostDataBuffer, inputData, engineInfo.inputSizePerSample * sizeof(float) * batchSize);
	return true;
}

bool ShogiOnnxExec::processOutput(const samplesCommon::BufferManager &buffers, int batchSize, float *outputPolicyData, float *outputValueData)
{
	std::string outputPName = addProfileSuffix(engineInfo.outputPolicyTensorName, engineInfo.profileForBatchSize[batchSize]);
	float *outputPolicy = static_cast<float *>(buffers.getHostBuffer(outputPName));
	memcpy(outputPolicyData, outputPolicy, engineInfo.outputPolicySizePerSample * sizeof(float) * batchSize);
	std::string outputVName = addProfileSuffix(engineInfo.outputValueTensorName, engineInfo.profileForBatchSize[batchSize]);
	float *outputValue = static_cast<float *>(buffers.getHostBuffer(outputVName));
	memcpy(outputValueData, outputValue, engineInfo.outputValueSizePerSample * sizeof(float) * batchSize);
	return true;
}

static vector<ShogiOnnxExec *> runnerForGPU;
static vector<std::mutex *> gpuMutexes; //同じGPUに対する操作のロック

void start_dnn_threads(string &evalDir, int format_board, int format_move, vector<int> &gpuIds)
{
	cvt = new DNNConverter(format_board, format_move);
	// TensorRTから発生するメッセージを抑制(gLogError << "")
	setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
	n_gpu_threads = gpuIds.size();
#ifdef MULTI_REQUEST_QUEUE
	for (size_t i = 0; i < n_gpu_threads; i++)
	{
		// リクエストキューをGPUスレッド分立てる
		request_queues.push_back(new MTQueue<dnn_eval_obj *>());
	}
#else
	// リクエストキューは1個だけ
	request_queues.push_back(new MTQueue<dnn_eval_obj *>());
#endif // MULTI_REQUEST_QUEUE

	const char *evalDirPtr = evalDir.c_str(); //この関数の実行中は存続するのでOK
	int maxGpuId = 0;
	for (size_t i = 0; i < gpuIds.size(); i++)
	{
		int gpu_id = gpuIds[i];
		if (gpu_id > maxGpuId)
		{
			maxGpuId = gpu_id;
		}
	}
	for (int i = 0; i < maxGpuId + 1; i++)
	{
		gpuMutexes.push_back(new std::mutex());
	}
	runnerForGPU.resize(maxGpuId + 1);
	vector<int> threadInDeviceCount(maxGpuId + 1);
	// 評価スレッドを立てる
	for (size_t i = 0; i < gpuIds.size(); i++)
	{
		int gpu_id = gpuIds[i];
		int threadInDevice = threadInDeviceCount[gpu_id]++;
		dnn_threads.push_back(new std::thread(dnn_thread_main, i, gpu_id, threadInDevice, evalDirPtr));
	}

	// スレッドの動作開始(DNNの初期化)まで待つ
	while (n_dnn_thread_initalized < dnn_threads.size())
	{
		sleep(1);
	}

	sync_cout << "info string dnn all initialize ok" << sync_endl;
	all_dnn_thread_initialized = true;
}

static void dnn_thread_main(size_t worker_idx, int device, int threadInDevice, const char *evalDir)
{
	sync_cout << "info string from dnn thread " << worker_idx << sync_endl;
	MTQueue<dnn_eval_obj *> *request_queue = request_queues[worker_idx % request_queues.size()];

	if (cudaSetDevice(device) != cudaSuccess)
	{
		gLogError << "cudaSetDevice failed" << std::endl;
		return;
	}

	ShogiOnnxExec *pRunner;
	if (threadInDevice == 0)
	{
		pRunner = new ShogiOnnxExec();
		if (!pRunner->load(evalDir))
		{
			gLogError << "load failed" << std::endl;
			return;
		}

		// dummy run
		vector<float> dummyInputData(pRunner->engineInfo.inputSizePerSample * sizeof(float) * batch_size);
		vector<float> dummyOutputPolicyData(pRunner->engineInfo.outputPolicySizePerSample * sizeof(float) * batch_size);
		vector<float> dummyOutputValueData(pRunner->engineInfo.outputValueSizePerSample * sizeof(float) * batch_size);
		pRunner->infer(batch_size, dummyInputData.data(), dummyOutputPolicyData.data(), dummyOutputValueData.data());
		runnerForGPU[device] = pRunner;
		sync_cout << "info string dnn for gpu " << device << " initialize ok" << sync_endl;
	}

	n_dnn_thread_initalized.fetch_add(1);
	while (!all_dnn_thread_initialized)
	{
		sleep(1);
	}
	pRunner = runnerForGPU[device];

	dnn_eval_obj **eval_targets = new dnn_eval_obj *[batch_size];
	while (true)
	{
		size_t item_count = request_queue->pop_batch(eval_targets, batch_size);
		// 実際のアイテム数で毎回バッチサイズを変える場合
		vector<float> inputData(pRunner->engineInfo.inputSizePerSample * item_count);
		// eval_targetsをDNN評価
		for (size_t i = 0; i < item_count; i++)
		{
			memcpy(&inputData[pRunner->engineInfo.inputSizePerSample * i], eval_targets[i]->input_array, pRunner->engineInfo.inputSizePerSample * sizeof(float));
		}

		std::vector<float> policyData(pRunner->engineInfo.outputPolicySizePerSample * item_count);
		std::vector<float> valueData(pRunner->engineInfo.outputValueSizePerSample * item_count);
		gpuMutexes[device]->lock();
		pRunner->infer(item_count, inputData.data(), policyData.data(), valueData.data());
		gpuMutexes[device]->unlock();

		for (size_t i = 0; i < item_count; i++)
		{
			dnn_eval_obj &eval_obj = *eval_targets[i];

			// 勝率=tanh(valueData[i][0] - valueData[i][1])
#ifdef EVAL_KPPT
			result_obj.static_value = eval_obj.static_value;
#else
			eval_obj.static_value = tanh((valueData[i * pRunner->engineInfo.outputValueSizePerSample + 0] - valueData[i * pRunner->engineInfo.outputValueSizePerSample + 1]) / value_temperature) * value_scale;
#endif

			// 合法手内でsoftmax確率を取る
			float raw_values[MAX_MOVES];
			float raw_max = -10000.0F;
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				raw_values[j] = policyData[i * pRunner->engineInfo.outputPolicySizePerSample + eval_obj.move_indices[j].index];
				if (raw_max < raw_values[j])
				{
					raw_max = raw_values[j];
				}
			}
			float exps[MAX_MOVES];
			float exp_sum = 0.0F;
			for (int j = 0; j < eval_obj.n_moves; j++)
			{
				float e = std::exp((raw_values[j] - raw_max) / policy_temperature); //temperatureで割る
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
#endif
