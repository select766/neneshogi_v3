
#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "ipqueue.h"
#include "dnn_converter.h"
#include "dnn_eval_obj.h"
#include <numeric>
#include <atomic>
#include <vector>
#include <functional>

extern ipqueue<dnn_eval_obj> *eval_queue;
extern ipqueue<dnn_result_obj> *result_queue;
extern float policy_temperature;
extern float value_temperature;
extern float value_scale;
extern std::atomic_int n_dnn_thread_initalized;

class DeviceModel
{
public:
	CNTK::DeviceDescriptor device;
	CNTK::FunctionPtr modelFunc;
	DeviceModel(CNTK::DeviceDescriptor _device, CNTK::FunctionPtr _modelFunc) :
		device(_device), modelFunc(_modelFunc)
	{

	}
};
extern vector<DeviceModel> device_models;
extern shared_ptr<DNNConverter> cvt;
void dnn_thread_main(int worker_idx);
