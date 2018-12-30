
#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "dnn_converter.h"
#include "dnn_eval_obj.h"
#include <numeric>
#include <atomic>
#include <vector>
#include <functional>

extern vector<MTQueue<dnn_eval_obj*>*> request_queues;
extern float policy_temperature;
extern float value_temperature;
extern float value_scale;
extern std::atomic_int n_dnn_thread_initalized;
extern size_t batch_size;
extern size_t n_gpu_threads;//GPUスレッドの数(GPU数と必ずしも一致しない)
extern DNNConverter *cvt;
extern std::atomic_int n_dnn_evaled_samples;
extern std::atomic_int n_dnn_evaled_batches;
void start_dnn_threads(string& evalDir, int format_board, int format_move, vector<int>& gpuIds);
