"""
CNTKでfp32のモデルからfp16のモデルへパラメータをコピーする。
それ以外の構造が同じであることが前提。

fp16のダミーモデルを作成しておくことが必要。model.yamlでkwargsにfp16: trueを設定し、
solver.yamlでval_frequency: 100などと設定してとにかくモデルを保存させる。
これのパラメータを学習済みfp32モデルからコピーする。
"""

import argparse
import numpy as np
import cntk as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_fp32")
    parser.add_argument("src_fp16")
    parser.add_argument("dst_fp16")

    args = parser.parse_args()
    model_fp32 = C.load_model(args.src_fp32)
    model_fp16 = C.load_model(args.src_fp16)
    assert len(model_fp32.parameters) == len(model_fp16.parameters)
    for pf32, pf16 in zip(model_fp32.parameters, model_fp16.parameters):
        assert pf16.name == pf32.name
        assert pf16.shape == pf32.shape
        pf16.value = pf32.value.astype(np.float16)
    for pf32, pf16 in zip(model_fp32.constants, model_fp16.constants):
        assert pf16.name == pf32.name
        assert pf16.shape == pf32.shape
        pf16.value = pf32.value.astype(np.float16)
    model_fp16.save(args.dst_fp16)


if __name__ == '__main__':
    main()
