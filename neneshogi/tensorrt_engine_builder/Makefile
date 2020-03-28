TARGETS = tensorrt_engine_builder
TENSORRT_DIR = ../../../TensorRT-7.0.0.11
LDFLAGS = -L"/usr/local/cuda/lib64" -L"${TENSORRT_DIR}/lib" -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lcudnn -lcublas -lcudart -lrt -ldl -lpthread -lmyelin -lnvrtc -Wl,--end-group

all: $(TARGETS)

tensorrt_engine_builder: tensorrt_engine_builder.o tensorrt/logger.o
	g++ -o $@ $^ $(LDFLAGS)

.cpp.o:
	g++ -Wall -Wno-deprecated-declarations -std=c++11 -I"/usr/local/cuda/include" -I"/usr/local/cuda/include" -I"$(TENSORRT_DIR)/include" -D_REENTRANT -c -o $@ $<

.PHONY: clean
clean:
	-rm -f *.o $(TARGETS)
