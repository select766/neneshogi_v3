﻿#ifndef _EVALUATE_LEARN_KPP_KKPT_CPP_
#define _EVALUATE_LEARN_KPP_KKPT_CPP_

// KPP_KKPT評価関数の学習時用のコード
// KPPTの学習用コードのコピペから少しいじった感じ。

#include "../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_KPP_KKPT)

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "learn.h"
#include "learning_tools.h"
#include "../eval/evaluate_io.h"

#include "../evaluate.h"
#include "../eval/evaluate_kpp_kkpt.h"
#include "../eval/kppt_evalsum.h"
#include "../eval/evaluate_io.h"
#include "../position.h"
#include "../misc.h"

using namespace std;

namespace Eval
{
	// 学習のときの勾配配列の初期化
	void init_grad(double eta1, u64 eta_epoch, double eta2, u64 eta2_epoch, double eta3);

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad,bool freeze_kpp);

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	void update_weights(u64 epoch, bool freeze_kk , bool freeze_kkp , bool freeze_kpp);

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name);
}

// --- 以下、定義

namespace Eval
{
	using namespace EvalLearningTools;

	// 評価関数学習用の構造体

	// KK,KKPのWeightを保持している配列
	// 直列化してあるので1次元配列
	std::vector<Weight2> weights;

	// KPPは手番なしなので手番なし用の1次元配列。
	std::vector<Weight> weights_kpp;

	// 学習のときの勾配配列の初期化
	// 引数のetaは、AdaGradのときの定数η(eta)。
	void init_grad(double eta1, u64 eta1_epoch, double eta2, u64 eta2_epoch, double eta3)
	{
		// 学習で使用するテーブル類の初期化
		EvalLearningTools::init();
			
		// 学習用配列の確保
		u64 size = KKP::max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。
		memset(&weights[0], 0, sizeof(Weight2) * weights.size());

		u64 size_kpp = KPP::max_index() - KPP::min_index();
		weights_kpp.resize(size_kpp);
		memset(&weights_kpp[0], 0, sizeof(Weight) * weights_kpp.size());

		// 学習率の設定
		Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);

	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad , bool freeze_kpp)
	{
		// LearnFloatTypeにatomicつけてないが、2つのスレッドが、それぞれx += yと x += z を実行しようとしたとき
		// 極稀にどちらか一方しか実行されなくともAdaGradでは問題とならないので気にしないことにする。
		// double型にしたときにWeight.gが破壊されるケースは多少困るが、double型の下位4バイトが破壊されたところで
		// それによる影響は小さな値だから実害は少ないと思う。
		
		// 勾配に多少ノイズが入ったところで、むしろ歓迎！という意味すらある。
		// (cf. gradient noise)

		// Aperyに合わせておく。
		delta_grad /= 32.0 /*FV_SCALE*/;

		// 勾配
		array<LearnFloatType,2> g =
		{
			// 手番を考慮しない値
			(rootColor == BLACK             ) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad),

			// 手番を考慮する値
			(rootColor == pos.side_to_move()) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad)
		};

		// 180度盤面を回転させた位置関係に対する勾配
		array<LearnFloatType,2> g_flip = { -g[0] , g[1] };

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);

		auto& pos_ = *const_cast<Position*>(&pos);

#if !defined (USE_EVAL_MAKE_LIST_FUNCTION)

		auto list_fb = pos_.eval_list()->piece_list_fb();
		auto list_fw = pos_.eval_list()->piece_list_fw();

#else
		// -----------------------------------
		// USE_EVAL_MAKE_LIST_FUNCTIONが定義されているときは
		// ここでeval_listをコピーして、組み替える。
		// -----------------------------------

		// バッファを確保してコピー
		BonaPiece list_fb[40];
		BonaPiece list_fw[40];
		memcpy(list_fb, pos_.eval_list()->piece_list_fb(), sizeof(BonaPiece) * 40);
		memcpy(list_fw, pos_.eval_list()->piece_list_fw(), sizeof(BonaPiece) * 40);

		// ユーザーは、この関数でBonaPiece番号の自由な組み換えを行なうものとする。
		make_list_function(pos, list_fb, list_fw);
#endif

		// KK
		weights[KK(sq_bk,sq_wk).toIndex()].add_grad(g);

		// flipした位置関係にも書き込む
		//kk_w[Inv(sq_wk)][Inv(sq_bk)].g += g_flip;

		for (int i = 0; i < PIECE_NO_KING; ++i)
		{
			BonaPiece k0 = list_fb[i];
			BonaPiece k1 = list_fw[i];

			if (!freeze_kpp)
			{
				// このループではk0 == l0は出現しない。(させない)
				// それはKPであり、KKPの計算に含まれると考えられるから。
				for (int j = 0; j < i; ++j)
				{
					BonaPiece l0 = list_fb[j];
					BonaPiece l1 = list_fw[j];

					weights_kpp[KPP(sq_bk, k0, l0).toIndex() - KPP::min_index()].add_grad(g[0]);
					weights_kpp[KPP(Inv(sq_wk), k1, l1).toIndex() - KPP::min_index()].add_grad(g_flip[0]);
				}
			}

			// KKP
			weights[KKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
		}
	}

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	// epoch       : 世代カウンター(0から始まる)
	// freeze_kk   : kkは学習させないフラグ
	// freeze_kkp  : kkpは学習させないフラグ
	// freeze_kpp  : kppは学習させないフラグ
	void update_weights(u64 epoch, bool freeze_kk , bool freeze_kkp , bool freeze_kpp)
	{
		u64 vector_length = KPP::max_index();

		// KPPを学習させないなら、KKPのmaxまでだけで良い。あとは数が少ないからまあいいや。
		if (freeze_kpp)
			vector_length = KKP::max_index();

		// epochに応じたetaを設定してやる。
		Weight::calc_eta(epoch);

		// ゼロ定数 手番つき、手番なし
		const auto zero_t = std::array<LearnFloatType, 2> {0, 0};
		const auto zero = LearnFloatType(0);
		
		// 並列化を効かせたいので直列化されたWeight配列に対してループを回す。

#pragma omp parallel
		{

#if defined(_OPENMP)
			// Windows環境下でCPUが２つあるときに、論理64コアまでしか使用されないのを防ぐために
			// ここで明示的にCPUに割り当てる
			int thread_index = omp_get_thread_num();    // 自分のthread numberを取得
			WinProcGroup::bindThisThread(thread_index);
#endif

#pragma omp for schedule(dynamic,20000)
			for (s64 index_ = 0; index_ < (s64)vector_length; ++index_)
			{
				// OpenMPではループ変数は符号型変数でなければならないが
				// さすがに使いにくい。
				u64 index = (u64)index_;

				// 自分が更新すべきやつか？
				// 次元下げしたときのindexの小さいほうが自分でないならこの更新は行わない。
				if (!min_index_flag[index])
					continue;

				if (KK::is_ok(index) && !freeze_kk)
				{
					KK x = KK::fromIndex(index);

					// 次元下げ
					KK a[KK_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					// 次元下げで得た情報を元に、それぞれのindexを得る。
					u64 ids[KK_LOWER_COUNT];
					for (int i = 0; i < KK_LOWER_COUNT; ++i)
						ids[i] = a[i].toIndex();

					// それに基いてvの更新を行い、そのvをlowerDimensionsそれぞれに書き出す。
					// ids[0]==ids[1]==ids[2]==ids[3]みたいな可能性があるので、gは外部で合計する。
					array<LearnFloatType, 2> g_sum = zero_t;

					// inverseした次元下げに関しては符号が逆になるのでadjust_grad()を経由して計算する。
					for (int i = 0; i < KK_LOWER_COUNT; ++i)
						g_sum += a[i].adjust_grad(weights[ids[i]].get_grad());
					
					// 次元下げを考慮して、その勾配の合計が0であるなら、一切の更新をする必要はない。
					if (is_zero(g_sum))
						continue;

					auto& v = kk[a[0].king0()][a[0].king1()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

					for (int i = 1; i < KK_LOWER_COUNT; ++i)
						kk[a[i].king0()][a[i].king1()] = a[i].adjust_grad(v);
					
					// mirrorした場所が同じindexである可能性があるので、gのクリアはこのタイミングで行なう。
					// この場合、毎回gを通常の2倍加算していることになるが、AdaGradは適応型なのでこれでもうまく学習できる。
					for (auto id : ids)
						weights[id].set_grad(zero_t);

				}
				else if (KKP::is_ok(index) && !freeze_kkp)
				{
					// KKの処理と同様

					KKP x = KKP::fromIndex(index);

					KKP a[KKP_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					u64 ids[KKP_LOWER_COUNT];
					for (int i = 0; i < KKP_LOWER_COUNT; ++i)
						ids[i] = a[i].toIndex();

					array<LearnFloatType, 2> g_sum = zero_t;
					for (int i = 0; i <KKP_LOWER_COUNT; ++i)
						g_sum += a[i].adjust_grad(weights[ids[i]].get_grad());
					
					if (is_zero(g_sum))
						continue;

					auto& v = kkp[a[0].king0()][a[0].king1()][a[0].piece()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

					for (int i = 1; i < KKP_LOWER_COUNT; ++i)
						kkp[a[i].king0()][a[i].king1()][a[i].piece()] = a[i].adjust_grad(v);
					
					for (auto id : ids)
						weights[id].set_grad(zero_t);

				}
				else if (KPP::is_ok(index) && !freeze_kpp)
				{
					KPP x = KPP::fromIndex(index);

					KPP a[KPP_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					u64 ids[KPP_LOWER_COUNT];
					for (int i = 0; i < KPP_LOWER_COUNT; ++i)
						ids[i] = a[i].toIndex();

					// KPPに関してはinverseの次元下げがないので、inverseの判定は不要。

					// KPPTとの違いは、ここに手番がないというだけ。
					LearnFloatType g_sum = zero;
					for (auto id : ids)
						g_sum += weights_kpp[id - KPP::min_index()].get_grad();

					if (g_sum == 0)
						continue;

					auto& v = kpp[a[0].king()][a[0].piece0()][a[0].piece1()];
					weights_kpp[ids[0] - KPP::min_index()].set_grad(g_sum);
					weights_kpp[ids[0] - KPP::min_index()].updateFV(v);

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
					for (int i = 1; i < KPP_LOWER_COUNT; ++i)
						kpp[a[i].king()][a[i].piece0()][a[i].piece1()] = v;
#else
					// 三角配列の場合、KPP::toLowerDimensionsで、piece0とpiece1を入れ替えたものは返らないので
					// (同じindexを指しているので)、自分で入れ替えてkpp配列にvの値を反映させる。
					kpp[a[0].king()][a[0].piece1()][a[0].piece0()] = v;
#if KPP_LOWER_COUNT == 2
					kpp[a[1].king()][a[1].piece0()][a[1].piece1()] = v;
					kpp[a[1].king()][a[1].piece1()][a[1].piece0()] = v;
#endif
#endif

					for (auto id : ids)
						weights_kpp[id - KPP::min_index()].set_grad(zero);
				}
			}
		}
	}

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name)
	{
		{
			auto eval_dir = path_combine((string)Options["EvalSaveDir"], dir_name);

			cout << "save_eval() start. folder = " << eval_dir << endl;

			// すでにこのフォルダがあるならmkdir()に失敗するが、
			// 別にそれは構わない。なければ作って欲しいだけ。
			// また、EvalSaveDirまでのフォルダは掘ってあるものとする。

			MKDIR(eval_dir);

			// EvalIOを利用して評価関数ファイルに書き込む。
			// 読み込みのときのinputとoutputとを入れ替えるとファイルに書き込める。EvalIo::eval_convert()マジ優秀。
			auto make_name = [&](std::string filename) { return path_combine(eval_dir, filename); };
			auto input = EvalIO::EvalInfo::build_kpp_kkpt32((void*)kk, (void*)kkp, (void*)kpp);
			auto output = EvalIO::EvalInfo::build_kpp_kkpt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN));

			// 評価関数の実験のためにfe_endをKPPT32から変更しているかも知れないので現在のfe_endの値をもとに書き込む。
			input.fe_end = output.fe_end = Eval::fe_end;

			if (!EvalIO::eval_convert(input, output, nullptr))
				goto Error;

			cout << "save_eval() finished. folder = " << eval_dir << endl;
			return;
		}

	Error:;
		cout << "Error : save_eval() failed" << endl;
	}

	// 現在のetaを取得する。
	double get_eta() {
		return Weight::eta;
	}

}

#endif // EVAL_LEARN
#endif // _EVALUATE_LEARN_KPP_KPPT_CPP_