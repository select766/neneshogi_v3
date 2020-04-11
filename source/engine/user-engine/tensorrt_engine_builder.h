#ifndef DNN_EXTERNAL
bool tensorrt_engine_builder(const char *onnxModelPath,
                             const char *dstDir,
                             int batchSizeMin,
                             int batchSizeMax,
                             const char *profileBatchSizeRange,
                             int fpbit);
#endif
