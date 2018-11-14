#ifdef PYMODULE
#include "print_py.h"

#ifdef _MSC_VER
// 日本語版コンパイラだと、ソースコードがutf-8であろうと文字列定数がshift-jisで埋め込まれる。
// pretty()の戻り値もそれに準拠している。
// pybind11でpython側に返す文字列はutf-8でないといけないので、変換する。
// 変換サンプルコードを利用 https://qiita.com/javacommons/items/9ea0c8fd43b61b01a8da

#include <windows.h>
#include <string>
#include <vector>

static inline std::wstring cp_to_wide(const std::string &s, UINT codepage)
{
	int in_length = (int)s.length();
	int out_length = MultiByteToWideChar(codepage, 0, s.c_str(), in_length, 0, 0);
	std::vector<wchar_t> buffer(out_length);
	if (out_length) MultiByteToWideChar(codepage, 0, s.c_str(), in_length, &buffer[0], out_length);
	std::wstring result(buffer.begin(), buffer.end());
	return result;
}

static inline std::string wide_to_cp(const std::wstring &s, UINT codepage)
{
	int in_length = (int)s.length();
	int out_length = WideCharToMultiByte(codepage, 0, s.c_str(), in_length, 0, 0, 0, 0);
	std::vector<char> buffer(out_length);
	if (out_length) WideCharToMultiByte(codepage, 0, s.c_str(), in_length, &buffer[0], out_length, 0, 0);
	std::string result(buffer.begin(), buffer.end());
	return result;
}

static inline std::string cp_to_utf8(const std::string &s, UINT codepage)
{
	if (codepage == CP_UTF8) return s;
	std::wstring wide = cp_to_wide(s, codepage);
	return wide_to_cp(wide, CP_UTF8);
}

static std::string convcode(const std::string &s)
{
	return cp_to_utf8(s, 932);
}

#else
// VC以外はutf-8を想定
static std::string convcode(const std::string &s)
{
	return s;
}
#endif

std::string PrintPy::move(int m)
{
	return convcode(pretty((Move)m));
}

std::string PrintPy::piece(int p)
{
	return convcode(pretty((Piece)p));
}

std::string PrintPy::board(const DNNConverterPy & c)
{
	std::stringstream ss;
	ss << c.pos;
	return convcode(ss.str());
}

#endif