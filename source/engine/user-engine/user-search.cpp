#include "../../extra/all.h"
#include "dnn_converter.h"
#include <numeric>
#include <functional>

#ifndef USER_ENGINE
// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos_, istringstream& is)
{
}
#endif
