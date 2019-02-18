"""
局面を学習済みモデルに適用し、その出力を保存する。distillationのターゲットとして使用するため。
"""

import sys
import os
import argparse
from collections import defaultdict
import pickle
import logging

import numpy as np

import cntk as C
from neneshogi import util
from neneshogi.packed_sfen_data_source import PackedSfenDataSource
from neneshogi import models
from neneshogi.train_manager import TrainManager
import neneshogi.log
from neneshogi.mutex_stopper import MutexStopper

logger = logging.getLogger(__name__)

N_TOP_MOVES = 64  # policy上位何要素を保存するか
# 1レコードごとのデータ形式(N_TOP_POLICIES = 64 なら 392bytes)
move_value_pack_dtype = [("move_top_values", np.float32, (N_TOP_MOVES,)),
                         ("move_top_idxs", np.uint16, (N_TOP_MOVES,)),
                         ("value", np.float32, (2,))]


def one_batch(dst_file, model, board_data):
    output = model.eval(board_data)
    move_output = output[model[0]]
    value_output = output[model[1]]
    packed_data = np.zeros((len(board_data),), dtype=move_value_pack_dtype)
    for i in range(len(board_data)):
        move_vec = move_output[i]
        top_idxs = np.argsort(-move_vec)[:N_TOP_MOVES]
        packed_data[i]["move_top_values"] = move_vec[top_idxs]
        packed_data[i]["move_top_idxs"] = top_idxs
    packed_data["value"] = value_output
    dst_file.write(packed_data.tobytes())


def loop(dst_file, model, ds, batchsize):
    processed = 0
    ctr = 0
    stopper = MutexStopper()
    while True:
        if stopper.wait():
            sys.stderr.write("GPU lock released\n")
        minibatch = ds.next_minibatch(batchsize, 1, 0)
        if ds.board_info in minibatch:
            board_data = minibatch[ds.board_info]
            one_batch(dst_file, model, board_data)
            processed += len(board_data)
            ctr += 1
            if ctr % 100 == 0:
                sys.stderr.write(f"processed {processed}\n")
        else:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model cmf file")
    parser.add_argument("kifu_config", help="yaml file of kwargs of PackedSfenDataSource")
    parser.add_argument("dst")
    parser.add_argument("--batchsize", type=int, default=256)
    args = parser.parse_args()
    # FULL_DATA_SWEEP=1周で終わり
    ds = PackedSfenDataSource(max_samples=C.io.FULL_DATA_SWEEP, **util.yaml_load(args.kifu_config))
    model_f = C.load_model(args.model)
    feature_var = C.input_variable(ds.board_shape)
    model = model_f(feature_var)
    with open(args.dst, "wb") as f:
        loop(f, model, ds, args.batchsize)


if __name__ == '__main__':
    main()
