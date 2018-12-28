
#include "../../extra/all.h"
#include "CNTKLibrary.h"
#include "dnn_converter.h"
#include "dnn_eval_obj.h"
#include <numeric>
#include <atomic>
#include <vector>
#include <functional>

extern MTQueue<dnn_eval_obj*> *request_queue;
extern float policy_temperature;
extern float value_temperature;
extern float value_scale;
extern std::atomic_int n_dnn_thread_initalized;
extern int batch_size;

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
extern DNNConverter *cvt;
void dnn_thread_main(int worker_idx);
