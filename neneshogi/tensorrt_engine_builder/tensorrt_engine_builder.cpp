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
#include "../../source/engine/user-engine/dnn_engine_info.h"

static int profileBatchSizeMultiplier = 0;
static int batchSizeMin = 1;
static int batchSizeMax = 16;
static bool fp16 = false;
static bool fp8 = false;
static const char *onnxModelPath = nullptr;
static const char *dstDir = nullptr;
static std::vector<std::string> inputTensorNames;
static std::vector<std::string> outputTensorNames;

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
    if (!profileBatchSizeMultiplier)
    {
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMIN, Dims4{1, 119, 9, 9});
        profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kOPT, Dims4{batchSizeMax, 119, 9, 9});
        profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMAX, Dims4{batchSizeMax, 119, 9, 9});
        int profileIdx = config->addOptimizationProfile(profile);
        profileForBatchSize.resize(batchSizeMax + 1);
        for (int b = 1; b <= batchSizeMax; b++)
        {
            profileForBatchSize[b] = profileIdx;
        }
    }
    else
    {
        // profileBatchSizeMultiplier==4のとき
        // バッチサイズ1, 2-4, 5-16, 17-64に対しそれぞれ
        // バッチサイズ1, 4, 16, 64に最適化した実行計画を作成
        int bs = 1;
        int lastbs = 0;
        profileForBatchSize.resize(batchSizeMax + 1);
        while (lastbs < batchSizeMax)
        {
            auto profile = builder->createOptimizationProfile();
            if (bs > batchSizeMax)
            {
                bs = batchSizeMax;
            }
            profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMIN, Dims4{lastbs + 1, 119, 9, 9});
            profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kOPT, Dims4{bs, 119, 9, 9});
            profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMAX, Dims4{bs, 119, 9, 9});
            int profileIdx = config->addOptimizationProfile(profile);
            for (int b = lastbs + 1; b <= bs; b++)
            {
                profileForBatchSize[b] = profileIdx;
            }

            lastbs = bs;
            bs *= profileBatchSizeMultiplier;
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
    strcpy(engineInfo.inputTensorName, inputTensorNames[0].c_str());
    strcpy(engineInfo.outputPolicyTensorName, outputTensorNames[0].c_str());
    strcpy(engineInfo.outputValueTensorName, outputTensorNames[1].c_str());
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

int main(int argc, char **argv)
{
    if (argc != 7)
    {
        std::cerr << "usage: tensorrt_engine_builder onnxModel dstDir batchSizeMin batchSizeMax profileBatchSizeMultiplier fpbit" << std::endl;
        return 1;
    }
    onnxModelPath = argv[1];
    dstDir = argv[2];
    batchSizeMin = atoi(argv[3]);
    batchSizeMax = atoi(argv[4]);
    profileBatchSizeMultiplier = atoi(argv[5]);
    int fpbit = atoi(argv[6]);
    if (fpbit == 8)
    {
        fp8 = true;
    }
    else if (fpbit == 16)
    {
        fp16 = true;
    }
    inputTensorNames.push_back("input");
    outputTensorNames.push_back("output_policy");
    outputTensorNames.push_back("output_value");

    auto runner = new ShogiOnnx();
    if (!runner->build())
    {
        gLogError << "build failed" << std::endl;
        return 1;
    }
    if (!runner->serialize())
    {
        gLogError << "serialize failed" << std::endl;
        return 1;
    }
    return 0;
}
