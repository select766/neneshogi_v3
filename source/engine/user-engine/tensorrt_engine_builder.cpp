#include "../../extra/all.h"

#ifndef DNN_EXTERNAL
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
#include "tensorrt_engine_builder.h"

static string profileBatchSizeRange;
static int batchSizeMin = 1;
static int batchSizeMax = 16;
static bool fp16 = false;
static bool fp8 = false;
static const char *onnxModelPath = nullptr;
static const char *dstDir = nullptr;
static const char *inputTensorName = "input";
static const char *outputPolicyTensorName = "output_policy";
static const char *outputValueTensorName = "output_value";

class ShogiOnnx
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    ShogiOnnx()
        : mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    bool serialize();

private:
    nvinfer1::Dims mInputDims;        //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputPolicyDims; //!< The dimensions of the output to the network.
    nvinfer1::Dims mOutputValueDims;  //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::map<int, std::shared_ptr<nvinfer1::IExecutionContext>> mContextForProfile;
    std::vector<int> profileForBatchSize;

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                          SampleUniquePtr<nvonnxparser::IParser> &parser);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool ShogiOnnx::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }
    if (!profileBatchSizeRange.length())
    {
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(inputTensorName, OptProfileSelector::kMIN, Dims4{1, 119, 9, 9});
        profile->setDimensions(inputTensorName, OptProfileSelector::kOPT, Dims4{batchSizeMax, 119, 9, 9});
        profile->setDimensions(inputTensorName, OptProfileSelector::kMAX, Dims4{batchSizeMax, 119, 9, 9});
        int profileIdx = config->addOptimizationProfile(profile);
        profileForBatchSize.resize(batchSizeMax + 1);
        for (int b = 1; b <= batchSizeMax; b++)
        {
            profileForBatchSize[b] = profileIdx;
        }
    }
    else
    {
        // profileBatchSizeRange: opt1-max1-opt2-max2...
        // profileBatchSizeRange==10-20-100-200のとき
        // バッチサイズ1~20について、バッチサイズ10に最適化した実行計画を作成
        // バッチサイズ21~200について、バッチサイズ100に最適化した実行計画を作成
        string pbsr = profileBatchSizeRange;
        replace(pbsr.begin(), pbsr.end(), '-', ' ');//ハイフン区切りの文字列をスペース区切りにしてistringstreamでトークンごとに読む
        istringstream iss(pbsr);
        int lastbs = 0;
        profileForBatchSize.resize(batchSizeMax + 1);
        int bs_opt, bs_max;
        string bs_opt_s, bs_max_s;
        while (iss >> bs_opt_s >> bs_max_s) {
            bs_opt = atoi(bs_opt_s.c_str());
            bs_max = atoi(bs_max_s.c_str());
            auto profile = builder->createOptimizationProfile();
            profile->setDimensions(inputTensorName, OptProfileSelector::kMIN, Dims4{lastbs + 1, 119, 9, 9});
            profile->setDimensions(inputTensorName, OptProfileSelector::kOPT, Dims4{bs_opt, 119, 9, 9});
            profile->setDimensions(inputTensorName, OptProfileSelector::kMAX, Dims4{bs_max, 119, 9, 9});
            int profileIdx = config->addOptimizationProfile(profile);
            for (int b = lastbs + 1; b <= bs_max; b++)
            {
                profileForBatchSize[b] = profileIdx;
            }

            lastbs = bs_max;
        }
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

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

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 2);
    mOutputPolicyDims = network->getOutput(0)->getDimensions();
    assert(mOutputPolicyDims.nbDims == 2);
    mOutputValueDims = network->getOutput(1)->getDimensions();
    assert(mOutputValueDims.nbDims == 2);

    return true;
}

bool ShogiOnnx::serialize()
{
    string engineInfoPath(dstDir);
    engineInfoPath.append("/info.bin");

    string enginePath(dstDir);
    enginePath.append("/engine.bin");

    DNNEngineInfo engineInfo;
    IHostMemory *serializedModel = mEngine->serialize();

    ofstream serializedModelFile(enginePath, ios::binary);
    serializedModelFile.write((const char *)serializedModel->data(), serializedModel->size());

    engineInfo.serializedEngineSize = (int)serializedModel->size();
    engineInfo.inputDims[0] = mInputDims.d[0];
    engineInfo.inputDims[1] = mInputDims.d[1];
    engineInfo.inputDims[2] = mInputDims.d[2];
    engineInfo.inputDims[3] = mInputDims.d[3];
    engineInfo.inputSizePerSample = engineInfo.inputDims[1] * engineInfo.inputDims[2] * engineInfo.inputDims[3];
    engineInfo.outputPolicyDims[0] = mOutputPolicyDims.d[0];
    engineInfo.outputPolicyDims[1] = mOutputPolicyDims.d[1];
    engineInfo.outputPolicySizePerSample = engineInfo.outputPolicyDims[1];
    engineInfo.outputValueDims[0] = mOutputValueDims.d[0];
    engineInfo.outputValueDims[1] = mOutputValueDims.d[1];
    engineInfo.outputValueSizePerSample = engineInfo.outputValueDims[1];
    engineInfo.batchSizeMin = batchSizeMin;
    engineInfo.batchSizeMax = batchSizeMax;
    strcpy(engineInfo.inputTensorName, inputTensorName);
    strcpy(engineInfo.outputPolicyTensorName, outputPolicyTensorName);
    strcpy(engineInfo.outputValueTensorName, outputValueTensorName);
    for (int i = 0; i < profileForBatchSize.size(); i++)
    {
        engineInfo.profileForBatchSize[i] = profileForBatchSize[i];
    }

    ofstream infoFile(engineInfoPath, ios::binary);
    infoFile.write((const char *)&engineInfo, sizeof(engineInfo));

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool ShogiOnnx::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                                 SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                                 SampleUniquePtr<nvonnxparser::IParser> &parser)
{
    // [W] [TRT] Calling isShapeTensor before the entire network is constructed may result in an inaccurate result.
    auto parsed = parser->parseFromFile(
        onnxModelPath, static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(batchSizeMax);
    config->setMaxWorkspaceSize(1024_MiB);

    if (fp8)
    {
        gLogInfo << "INT8 mode (scale is not correctly set!)" << std::endl;
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    else if (fp16)
    {
        gLogInfo << "FP16 mode" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else
    {
        gLogInfo << "FP32 mode" << std::endl;
    }

    samplesCommon::enableDLA(builder.get(), config.get(), -1);

    return true;
}

bool tensorrt_engine_builder(const char *onnxModelPath,
                             const char *dstDir,
                             int batchSizeMin,
                             int batchSizeMax,
                             const char *profileBatchSizeRange,
                             int fpbit)
{
    ::onnxModelPath = onnxModelPath;
    ::dstDir = dstDir;
    string mkdir_command("mkdir -p ");
    mkdir_command += dstDir; //セキュリティ的にはよくないが
    if (system(mkdir_command.c_str()) != 0)
    {
        gLogError << "create output directory failed" << std::endl;
        return false;
    }
    ::batchSizeMin = batchSizeMin;
    ::batchSizeMax = batchSizeMax;
    ::profileBatchSizeRange = string(profileBatchSizeRange);
    if (fpbit == 8)
    {
        fp8 = true;
    }
    else if (fpbit == 16)
    {
        fp16 = true;
    }

    auto runner = new ShogiOnnx();
    if (!runner->build())
    {
        gLogError << "build failed" << std::endl;
        return false;
    }
    if (!runner->serialize())
    {
        gLogError << "serialize failed" << std::endl;
        return false;
    }
    return true;
}

#endif
