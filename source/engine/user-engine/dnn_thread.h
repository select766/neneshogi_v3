
#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "dnn_converter.h"
#include <numeric>
#include <functional>

extern CNTK::DeviceDescriptor device;
extern CNTK::FunctionPtr modelFunc;
extern shared_ptr<DNNConverter> cvt;
void dnn_thread_main();
