
#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "ipqueue.h"
#include "dnn_converter.h"
#include "dnn_eval_obj.h"
#include <numeric>
#include <functional>

extern ipqueue<dnn_eval_obj> *eval_queue;
extern ipqueue<dnn_result_obj> *result_queue;

extern CNTK::DeviceDescriptor device;
extern CNTK::FunctionPtr modelFunc;
extern shared_ptr<DNNConverter> cvt;
void dnn_thread_main();
