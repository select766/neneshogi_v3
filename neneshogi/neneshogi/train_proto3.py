"""
自前実装の学習ループを使った教師あり学習
CNTK 2.5.1時点で、今のところ自前で書かないと2つのロスを別個に表示することができない。
- lr制御
  - 学習の最初期は規定lrの1/10程度から開始し、1/10epochの間で徐々に上げる。
  - 1/10epochごとにvalidationを行う。
  - 規定の精度より悪いときは発散したとみなして終了。
  - 前回validationより5%以上改善してなければ、lrを1/2にする。
  - lrがもとの1/1000になったら終了。
  - 初回のvalidationで発散判定になった場合は、lrを1/10にして最初からやり直す。
- 中断制御
  - validationごとにモデルおよび学習の状態を保存し、いつでも中断・再開できるようにする。

中断制御機構
学習ループで直接的にインデックスを持たず、1回呼ぶたびに次のアクションを返すオブジェクトを使用。
このオブジェクト（＋datasource）の状態を保存すれば復帰できる。
"""

import os
import argparse
import numpy as np

import cntk as C
from neneshogi import util
from neneshogi.packed_sfen_data_source import PackedSfenDataSource
from neneshogi import models


# Create the network.
def create_shogi_model(board_shape, move_dim, model_config):
    # Input variables denoting the features and label data
    feature_var = C.input_variable(board_shape)
    policy_var = C.input_variable(move_dim)
    value_var = C.input_variable(1)

    model_function = getattr(models, model_config["name"])
    policy, value = model_function(feature_var, board_shape, move_dim, **model_config["kwargs"])

    # loss and metric
    ce = C.cross_entropy_with_softmax(policy, policy_var)
    pe = C.classification_error(policy, policy_var)
    pe5 = C.classification_error(policy, policy_var, topN=5)
    sqe = C.binary_cross_entropy(value, value_var)
    total_error = ce * 1.0 + sqe * 1.0

    return {
        'feature': feature_var,
        'policy': policy_var,
        'value': value_var,
        'ce': ce,
        'pe': pe,
        'sqe': sqe,
        'total_error': total_error,
        'output': C.combine([policy, value])
    }


def shogi_train_and_eval(solver_config, model_config, workdir, restore):
    epochs = solver_config["epoch"]
    ds_train_config = solver_config["dataset"]["train"]
    train_source = PackedSfenDataSource(ds_train_config["path"], count=ds_train_config["count"],
                                        skip=ds_train_config["skip"], format_board=model_config["format_board"],
                                        format_move=model_config["format_move"],
                                        max_samples=ds_train_config["count"] * epochs)
    epoch_size = ds_train_config["count"]
    ds_test_config = solver_config["dataset"]["test"]
    test_source = PackedSfenDataSource(ds_test_config["path"], count=ds_test_config["count"],
                                       skip=ds_test_config["skip"], format_board=model_config["format_board"],
                                       format_move=model_config["format_move"],
                                       max_samples=C.io.FULL_DATA_SWEEP)

    network = create_shogi_model(train_source.board_shape, train_source.move_dim, model_config)
    lr_schedule = C.learning_parameter_schedule(1e-4)
    mm_schedule = C.learners.momentum_schedule(0.9)
    learner = C.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False,
                                      l2_regularization_weight=5e-4)
    batch_size = solver_config["batchsize"]

    # (loss, criterion)のcriterionで評価に必要な変数は含みつつ、スカラーでなければならない(C.combine([sqe, pe])はダメ)
    criterion = network['sqe'] + network["pe"] * 0
    trainer = C.Trainer(network['output'], (network['total_error'], criterion), [learner])

    for i in range(100):
        for j in range(10):
            minibatch = train_source.next_minibatch(batch_size, 1, 0)
            data = {network["feature"]: minibatch[train_source.board_info],
                    network["policy"]: minibatch[train_source.move_info],
                    network["value"]: minibatch[train_source.result_info]}
            _, result = trainer.train_minibatch(data, outputs=[network["sqe"], network["pe"]])
            sqe = result[network["sqe"]]
            pe = result[network["pe"]]
            print(j, "sqe", np.mean(sqe), "pe", np.mean(pe))
        print("VALIDATION")
        for j in range(10):
            minibatch = test_source.next_minibatch(batch_size, 1, 0)
            data = {network["feature"]: minibatch[test_source.board_info],
                    network["policy"]: minibatch[test_source.move_info],
                    network["value"]: minibatch[test_source.result_info]}
            # X.forwardのXを生成する過程にoutputsが入ってないといけない
            _, result = criterion.forward(data, outputs=[network["sqe"], network["pe"]])
            sqe = result[network["sqe"]]
            pe = result[network["pe"]]
            print(j, "V", "sqe", np.mean(sqe), "pe", np.mean(pe))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir")
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)

    args = parser.parse_args()

    if args.device is not None:
        if args.device == -1:
            C.device.try_set_default_device(C.device.cpu())
        else:
            C.device.try_set_default_device(C.device.gpu(args.device))

    solver_config = util.yaml_load(os.path.join(args.workdir, "solver.yaml"))
    model_config = util.yaml_load(os.path.join(args.workdir, "model.yaml"))
    shogi_train_and_eval(solver_config, model_config, args.workdir, restore=not args.restart)
