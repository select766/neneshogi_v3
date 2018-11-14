#pragma once
#ifdef PYMODULE
#include "../../extra/all.h"
#include "dnn_converter_py.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// �ՖʁA���Ȃǂ�pretty print���邽�߂̃N���X
class PrintPy {
public:
	static std::string move(int m);
	static std::string piece(int p);
	// DNNConverterPy��set_sfen()������ԂŌĂяo�����ƂőΉ�����Ֆʕ�����𓾂�
	static std::string board(const DNNConverterPy& c);
};
#endif
