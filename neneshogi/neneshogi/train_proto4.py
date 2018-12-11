"""
自前実装の学習ループを使った教師あり学習
multipv学習に対応。
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
import logging

import numpy as np

import cntk as C
from neneshogi import util
from neneshogi.packed_sfen_data_source import PackedSfenDataSource
from neneshogi import models
from neneshogi.train_manager import TrainManager
import neneshogi.log

logger = logging.getLogger(__name__)


def find_latest_checkpoint_dir(workdir):
    cp_base_dir = os.path.join(workdir, "checkpoint")
    if not os.path.exists(cp_base_dir):
        return None
    latest_num = -1
    latest_dir = None
    for obj in os.listdir(cp_base_dir):
        if obj.startswith("train_"):
            num = int(obj[6:])
            if num > latest_num:
                latest_num = num
                latest_dir = obj
    if latest_dir is not None:
        return os.path.join(cp_base_dir, latest_dir)
    return None


def save_checkpoint(workdir, model_config, train_manager, data_sources, model):
    cp_dir = os.path.join(workdir, "checkpoint", f"train_{train_manager.trained_samples}")
    os.makedirs(cp_dir, exist_ok=True)
    model.save(os.path.join(cp_dir,
                            "nene_{}_{}.cmf".format(model_config["format_board"], model_config["format_move"])))
    status = {"train_manager": train_manager,
              "ds_train_state": data_sources["train"].get_checkpoint_state(),
              "ds_val_state": data_sources["val"].get_checkpoint_state()}
    with open(os.path.join(cp_dir, "status.bin"), "wb") as f:
        pickle.dump(status, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(checkpoint_dir, model_config, data_sources):
    with open(os.path.join(checkpoint_dir, "status.bin"), "rb") as f:
        status = pickle.load(f)
    data_sources["train"].restore_from_checkpoint(status["ds_train_state"])
    data_sources["val"].restore_from_checkpoint(status["ds_val_state"])
    return status["train_manager"]


# Create the network.
def create_shogi_model(board_shape, move_dim, model_config, loss_config, restore_path=None):
    # Input variables denoting the features and label data
    feature_var = C.input_variable(board_shape)
    policy_var = C.input_variable(move_dim)
    result_var = C.input_variable(2)
    winrate_var = C.input_variable(2)
    multi_policy_var = C.input_variable(move_dim)

    if restore_path:
        logging.info(f"restoring from {restore_path}")
        model_function = C.load_model(restore_path)
        combined = model_function(feature_var)
        policy = combined[0]
        value = combined[1]
    else:
        model_function = getattr(models, model_config["name"])
        policy, value = model_function(feature_var, board_shape, move_dim, **model_config["kwargs"])

    # loss and metric
    # loss_config["multipv_temperature"] -> 0ならhard, ->infならsoft
    multi_policy_prob = C.softmax(multi_policy_var * (1.0 / loss_config["multipv_temperature"]))
    loss_policy_ce = C.cross_entropy_with_softmax(policy, policy_var)
    loss_multi_policy_ce = C.cross_entropy_with_softmax(policy, multi_policy_prob)
    loss_policy_cle = C.classification_error(policy, policy_var)  # classification errorはPVに対して計算
    loss_policy_cle5 = C.classification_error(policy, policy_var, topN=5)
    loss_result_ce = C.cross_entropy_with_softmax(value, result_var)
    loss_winrate_ce = C.cross_entropy_with_softmax(value, winrate_var)
    loss_result_cle = C.classification_error(value, result_var)  # 勝敗の識別率
    total_error = loss_policy_ce * loss_config["weight_policy"] \
                  + loss_multi_policy_ce * loss_config["weight_multi_policy"] \
                  + loss_result_ce * loss_config["weight_result"] \
                  + loss_winrate_ce * loss_config["weight_winrate"]

    return {
        'feature': feature_var,
        'policy': policy_var,
        'result': result_var,
        'winrate': winrate_var,
        'multi_policy': multi_policy_var,
        'losses': {"policy_ce": loss_policy_ce,
                   "multi_policy_ce": loss_multi_policy_ce,
                   "policy_cle": loss_policy_cle,
                   "policy_cle5": loss_policy_cle5,
                   "result_ce": loss_result_ce,
                   "winrate_ce": loss_winrate_ce,
                   "result_cle": loss_result_cle},
        'total_error': total_error,
        'output': C.combine([policy, value])
    }


def shogi_train_and_eval(solver_config, model_config, workdir, restore):
    epochs = solver_config["epoch"]
    ds_train_config = solver_config["dataset"]["train"]
    train_source = PackedSfenDataSource(ds_train_config["path"], count=ds_train_config["count"],
                                        skip=ds_train_config["skip"], format_board=model_config["format_board"],
                                        format_move=model_config["format_move"],
                                        max_samples=ds_train_config["count"] * epochs,
                                        multi_pv=ds_train_config["multi_pv"],
                                        multi_pv_top=ds_train_config["multi_pv_top"])
    epoch_size = ds_train_config["count"]
    ds_val_config = solver_config["dataset"]["val"]
    val_epoch_size = ds_val_config["count"]
    val_source = PackedSfenDataSource(ds_val_config["path"], count=ds_val_config["count"],
                                      skip=ds_val_config["skip"], format_board=model_config["format_board"],
                                      format_move=model_config["format_move"],
                                      max_samples=C.io.INFINITELY_REPEAT,
                                      multi_pv=ds_val_config["multi_pv"],
                                      multi_pv_top=ds_val_config["multi_pv_top"])

    restore_path = None
    restore_cp_dir = find_latest_checkpoint_dir(workdir)
    if restore and restore_cp_dir is not None:
        restore_path = os.path.join(restore_cp_dir,
                                    "nene_{}_{}.cmf".format(model_config["format_board"], model_config["format_move"]))
        if not os.path.exists(restore_path):
            restore_path = None
    network = create_shogi_model(train_source.board_shape, train_source.move_dim, model_config, solver_config["loss"],
                                 restore_path)
    lr_schedule = C.learning_parameter_schedule(1e-4)
    mm_schedule = C.learners.momentum_schedule(0.9)
    learner = C.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False,
                                      l2_regularization_weight=solver_config.get("l2_regularization", 0.0))
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
        manager = load_checkpoint(restore_cp_dir, model_config, {"train": train_source, "val": val_source})
    else:
        manager = TrainManager(epoch_size=epoch_size, batch_size=batch_size, val_frequency=val_frequency,
                               initial_lr=solver_config["lr"], min_lr=solver_config["lr"] / 1000,
                               first_diverge_check_size=val_frequency,
                               diverge_criterion={"policy_cle": 0.95, "result_cle": 0.45},
                               lr_reduce_ratio=solver_config["lr_reduce"]["ratio"],
                               lr_reduce_average_count=solver_config["lr_reduce"]["average_count"],
                               lr_reduce_threshold=solver_config["lr_reduce"]["threshold"])
    last_lr = None
    i = 0
    while True:
        action = manager.get_next_action()
        if action["action"] == "quit":
            break
        if action["action"] == "train":
            # train 1 batch
            if action["lr"] != last_lr:
                logging.info(f"updating lr from {last_lr} to {action['lr']}")
                last_lr = action["lr"]
                learner.reset_learning_rate(C.learning_parameter_schedule(last_lr))
            minibatch = train_source.next_minibatch(batch_size, 1, 0)
            data = {network["feature"]: minibatch[train_source.board_info],
                    network["policy"]: minibatch[train_source.move_info],
                    network["result"]: minibatch[train_source.result_info],
                    network["winrate"]: minibatch[train_source.winrate_info],
                    network["multi_policy"]: minibatch[train_source.multi_move_info]}
            _, result = trainer.train_minibatch(data, outputs=[losses[c] for c in criterions])
            mean_criterions = {c: float(np.mean(result[losses[c]])) for c in criterions}
            i += 1
            if i % 100 == 0:
                logger.info(f"{manager.trained_samples}, lr, {action['lr']}, criterions, {mean_criterions}")
            manager.put_train_result(mean_criterions)
        if action["action"] == "val":
            # val 1 epoch
            logging.info("validating")
            n_validated_samples = 0
            sum_criterions = defaultdict(float)
            while n_validated_samples < val_epoch_size:
                minibatch = val_source.next_minibatch(val_batchsize, 1, 0)
                data = {network["feature"]: minibatch[val_source.board_info],
                        network["policy"]: minibatch[val_source.move_info],
                        network["result"]: minibatch[val_source.result_info],
                        network["winrate"]: minibatch[val_source.winrate_info],
                        network["multi_policy"]: minibatch[val_source.multi_move_info]}
                # X.forwardのXを生成する過程にoutputsが入ってないといけない
                _, result = criterion.forward(data, outputs=[losses[c] for c in criterions])
                for c in criterions:
                    sum_criterions[c] += float(np.sum(result[losses[c]]))
                n_validated_samples += val_batchsize
            mean_criterions = {}
            for k, v in sum_criterions.items():
                mean_criterions[k] = v / n_validated_samples
            logger.info(f"val_criterions, {mean_criterions}")
            manager.put_val_result(mean_criterions)
            logging.info("saving")
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
    neneshogi.log.init(os.path.join(args.workdir, "train.log"))
    try:
        shogi_train_and_eval(solver_config, model_config, args.workdir, restore=not args.restart)
    except Exception as ex:
        logger.exception("Aborted with exception")
