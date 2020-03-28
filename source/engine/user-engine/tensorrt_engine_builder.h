#ifndef DNN_EXTERNAL
bool tensorrt_engine_builder(const char *onnxModelPath,
                             const char *dstDir,
                             int batchSizeMin,
                             int batchSizeMax,
                             int profileBatchSizeMultiplier,
                             int fpbit);
#endif
