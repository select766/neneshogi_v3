#pragma once
#ifdef PYMODULE
#include "../../extra/all.h"
#include "dnn_converter_py.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// 盤面、駒種などをpretty printするためのクラス
class PrintPy {
public:
	static std::string move(int m);
	static std::string piece(int p);
	// DNNConverterPyにset_sfen()した状態で呼び出すことで対応する盤面文字列を得る
	static std::string board(const DNNConverterPy& c);
};
#endif
