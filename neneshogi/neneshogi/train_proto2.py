# based on CNTK official example
# https://github.com/Microsoft/CNTK/blob/81b57aa5ddff3c6c9e144c1604623f5cf914463e/Examples/Image/Classification/VGG/Python/VGG16_ImageNet_Distributed.py

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

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

    if model_config["name"] == "ResNet":
        policy, value = models.ResNet(feature_var, board_shape, move_dim, **model_config["kwargs"])
    else:
        raise NotImplementedError

    # loss and metric
    ce = C.cross_entropy_with_softmax(policy, policy_var)
    pe = C.classification_error(policy, policy_var)
    # pe5 = C.classification_error(policy, label_var, topN=5)
    sqe = C.binary_cross_entropy(value, value_var)
    total_error = ce * 0.01 + sqe * 1.0

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


# Create trainer
def create_trainer(network, epoch_size, progress_printer):
    # Set learning parameters
    lr_per_mb = [0.01] * 20 + [0.001] * 20 + [0.0001] * 20 + [0.00001] * 10 + [0.000001]
    lr_schedule = C.learning_parameter_schedule(lr_per_mb, epoch_size=epoch_size)
    mm_schedule = C.learners.momentum_schedule(0.9)
    l2_reg_weight = 0.0005  # CNTK L2 regularization is per sample, thus same as Caffe

    # Create learner
    local_learner = C.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False,
                                            l2_regularization_weight=l2_reg_weight)

    # Create trainer
    # 複数のmetricでの評価はまだサポートされてないようだ。
    # https://github.com/Microsoft/CNTK/issues/2522
    # しかしロス計算に使われない計算結果があると下記のwarningが出るので0をかけてごまかす。
    # WARNING: Function::Forward provided values for (1) extra arguments which are not required for evaluating the specified Function outputs!
    return C.Trainer(network['output'], (network['total_error'], network['sqe'] + network["pe"] * 0), [local_learner],
                     progress_printer)


# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, checkpoint_path, restore):
    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.board_info,
        network['policy']: train_source.move_info,
        network['value']: train_source.result_info
    }

    test_input_map = {
        network['feature']: test_source.board_info,
        network['policy']: test_source.move_info,
        network['value']: test_source.result_info
    }

    # Train all minibatches
    C.training_session(
        trainer=trainer, mb_source=train_source,
        model_inputs_to_streams=input_map,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config=C.CheckpointConfig(filename=checkpoint_path, restore=restore, frequency=epoch_size // 10),
        test_config=C.TestConfig(minibatch_source=test_source, minibatch_size=minibatch_size,
                                 model_inputs_to_streams=test_input_map)
    ).train()

def create_trainer_warmup(network, epoch_size, progress_printer):
    # Set learning parameters
    lr_schedule = C.learning_parameter_schedule(1e-4)
    mm_schedule = C.learners.momentum_schedule(0.9)
    l2_reg_weight = 0.0005  # CNTK L2 regularization is per sample, thus same as Caffe

    # Create learner
    local_learner = C.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False,
                                            l2_regularization_weight=l2_reg_weight)

    # Create trainer
    # 複数のmetricでの評価はまだサポートされてないようだ。
    # https://github.com/Microsoft/CNTK/issues/2522
    # しかしロス計算に使われない計算結果があると下記のwarningが出るので0をかけてごまかす。
    # WARNING: Function::Forward provided values for (1) extra arguments which are not required for evaluating the specified Function outputs!
    return C.Trainer(network['output'], (network['total_error'], network['sqe'] + network["pe"] * 0), [local_learner],
                     progress_printer)


def warmup_train(network, trainer, train_source, test_source, minibatch_size, epoch_size, checkpoint_path, restore):
    input_map = {
        network['feature']: train_source.board_info,
        network['policy']: train_source.move_info,
        network['value']: train_source.result_info
    }
    C.training_session(
        trainer=trainer, mb_source=train_source,
        model_inputs_to_streams=input_map,
        mb_size=minibatch_size,
        progress_frequency=epoch_size
    ).train()


# Train and evaluate the network.
def shogi_train_and_eval(solver_config, model_config, workdir, restore):
    epochs = solver_config["epoch"]
    progress_printer = C.logging.ProgressPrinter(
        freq=100,
        tag='Training',
        # log_to_file=log_to_file,
        num_epochs=epochs)

    ds_train_config = solver_config["dataset"]["train"]
    warmup_size = max(ds_train_config["count"] // 100, 10000)
    warmup_train_source = PackedSfenDataSource(ds_train_config["path"], count=warmup_size,
                                               skip=ds_train_config["skip"], format_board=model_config["format_board"],
                                               format_move=model_config["format_move"],
                                               max_samples=C.io.FULL_DATA_SWEEP)
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
    trainer = create_trainer(network, epoch_size, progress_printer)
    checkpoint_dir = os.path.join(workdir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir,
                                   f"nene_{model_config['format_board']}_{model_config['format_board']}.cmf")
    progress_printer_warmup = C.logging.ProgressPrinter(
        freq=10,
        tag='Training-Warmup',
        # log_to_file=log_to_file,
        num_epochs=1)

    if not (restore and os.path.exists(checkpoint_path)):
        warmup_trainer = create_trainer_warmup(network, warmup_size, progress_printer_warmup)
        print("start warmup")
        warmup_train(network, warmup_trainer, warmup_train_source, test_source, solver_config["batchsize"], warmup_size, None, False)
    print("start training")
    train_and_test(network, trainer, train_source, test_source, solver_config["batchsize"], epoch_size, checkpoint_path,
                   restore=restore)


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
