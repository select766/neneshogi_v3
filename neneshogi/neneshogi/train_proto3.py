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
from collections import defaultdict
import pickle

import numpy as np

import cntk as C
from neneshogi import util
from neneshogi.packed_sfen_data_source import PackedSfenDataSource
from neneshogi import models
from neneshogi.train_manager import TrainManager


def save_checkpoint(workdir, model_config, train_manager, data_sources, model):
    cp_dir = os.path.join(workdir, "checkpoint")
    os.makedirs(cp_dir, exist_ok=True)
    model.save(os.path.join(cp_dir,
                            "nene_{}_{}.cmf".format(model_config["format_board"], model_config["format_move"])))
    status = {"train_manager": train_manager,
              "ds_train_state": data_sources["train"].get_checkpoint_state(),
              "ds_val_state": data_sources["val"].get_checkpoint_state()}
    with open(os.path.join(cp_dir, "status.bin"), "wb") as f:
        pickle.dump(status, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(workdir, model_config, data_sources):
    cp_dir = os.path.join(workdir, "checkpoint")
    with open(os.path.join(cp_dir, "status.bin"), "rb") as f:
        status = pickle.load(f)
    data_sources["train"].restore_from_checkpoint(status["ds_train_state"])
    data_sources["val"].restore_from_checkpoint(status["ds_val_state"])
    return status["train_manager"]


# Create the network.
def create_shogi_model(board_shape, move_dim, model_config, restore_path=None):
    # Input variables denoting the features and label data
    feature_var = C.input_variable(board_shape)
    policy_var = C.input_variable(move_dim)
    value_var = C.input_variable(1)

    if restore_path:
        print(f"restoring from {restore_path}")
        model_function = C.load_model(restore_path)
        combined = model_function(feature_var)
        policy = combined[0]
        value = combined[1]
    else:
        model_function = getattr(models, model_config["name"])
        policy, value = model_function(feature_var, board_shape, move_dim, **model_config["kwargs"])


    # loss and metric
    loss_policy_ce = C.cross_entropy_with_softmax(policy, policy_var)
    loss_policy_cle = C.classification_error(policy, policy_var)
    loss_policy_cle5 = C.classification_error(policy, policy_var, topN=5)
    loss_value_ce = C.binary_cross_entropy(value, value_var)
    total_error = loss_policy_ce * 1.0 + loss_value_ce * 1.0

    return {
        'feature': feature_var,
        'policy': policy_var,
        'value': value_var,
        'losses': {"policy_ce": loss_policy_ce,
                   "policy_cle": loss_policy_cle,
                   "policy_cle5": loss_policy_cle5,
                   "value_ce": loss_value_ce},
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
    ds_val_config = solver_config["dataset"]["val"]
    val_epoch_size = ds_val_config["count"]
    val_source = PackedSfenDataSource(ds_val_config["path"], count=ds_val_config["count"],
                                       skip=ds_val_config["skip"], format_board=model_config["format_board"],
                                       format_move=model_config["format_move"],
                                       max_samples=C.io.INFINITELY_REPEAT)

    restore_path = None
    if restore:
        restore_path = os.path.join(workdir, "checkpoint",
                                    "nene_{}_{}.cmf".format(model_config["format_board"], model_config["format_move"]))
        if not os.path.exists(restore_path):
            restore_path = None
    network = create_shogi_model(train_source.board_shape, train_source.move_dim, model_config, restore_path)
    lr_schedule = C.learning_parameter_schedule(1e-4)
    mm_schedule = C.learners.momentum_schedule(0.9)
    learner = C.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False,
                                      l2_regularization_weight=5e-4)
    batch_size = solver_config["batchsize"]
    val_batchsize = solver_config.get("val_batchsize", batch_size)

    # (loss, criterion)のcriterionで評価に必要な変数は含みつつ、スカラーでなければならない(C.combine([sqe, pe])はダメ)
    losses = network["losses"]
    criterions = list(losses.keys())
    criterion = losses[criterions[0]]
    for c in criterions[1:]:
        criterion = criterion + losses[c]
    trainer = C.Trainer(network['output'], (network['total_error'], criterion), [learner])

    val_frequency = solver_config["val_frequency"]
    if restore_path:
        manager = load_checkpoint(workdir, model_config, {"train": train_source, "val": val_source})
    else:
        manager = TrainManager(epoch_size=epoch_size, batch_size=batch_size, val_frequency=val_frequency,
                               initial_lr=solver_config["lr"], min_lr=solver_config["lr"] / 1000,
                               first_diverge_check_size=val_frequency,
                               diverge_criterion={"policy_cle": 0.95, "value_ce": 0.6})
    last_lr = None
    i = 0
    while True:
        action = manager.get_next_action()
        if action["action"] == "quit":
            break
        if action["action"] == "train":
            # train 1 batch
            if action["lr"] != last_lr:
                print(f"updating lr from {last_lr} to {action['lr']}")
                last_lr = action["lr"]
                learner.reset_learning_rate(C.learning_parameter_schedule(last_lr))
            minibatch = train_source.next_minibatch(batch_size, 1, 0)
            data = {network["feature"]: minibatch[train_source.board_info],
                    network["policy"]: minibatch[train_source.move_info],
                    network["value"]: minibatch[train_source.result_info]}
            _, result = trainer.train_minibatch(data, outputs=[losses[c] for c in criterions])
            mean_criterions = {c: float(np.mean(result[losses[c]])) for c in criterions}
            i += 1
            if i % 100 == 0:
                print(i, "criterions", mean_criterions)
            manager.put_train_result(mean_criterions)
        if action["action"] == "val":
            # val 1 epoch
            n_validated_samples = 0
            sum_criterions = defaultdict(float)
            while n_validated_samples < val_epoch_size:
                minibatch = val_source.next_minibatch(val_batchsize, 1, 0)
                data = {network["feature"]: minibatch[val_source.board_info],
                        network["policy"]: minibatch[val_source.move_info],
                        network["value"]: minibatch[val_source.result_info]}
                # X.forwardのXを生成する過程にoutputsが入ってないといけない
                _, result = criterion.forward(data, outputs=[losses[c] for c in criterions])
                for c in criterions:
                    sum_criterions[c] += float(np.sum(result[losses[c]]))
                n_validated_samples += val_batchsize
            mean_criterions = {}
            for k, v in sum_criterions.items():
                mean_criterions[k] = v / n_validated_samples
            print("val_criterions", mean_criterions)
            manager.put_val_result(mean_criterions)
            print("saving")
            save_checkpoint(workdir, model_config, manager, {"train": train_source, "val": val_source},
                            network["output"])


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
