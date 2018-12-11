"""
MultiPV学習対応ResNet
"""

import numpy as np
import cntk as C
from cntk.layers import Convolution2D, Dense, BatchNormalization


def res_block(input_var, count, ch):
    h = input_var
    for i in range(count):
        h = BatchNormalization()(h)
        h = C.relu(h)
        h = Convolution2D(3, ch, pad=True, bias=False)(h)
    return h + input_var


def MPResNet(feature_var, board_shape, move_dim, *, ch=16, depth=4, block_depth=3, spatial_bias=False):
    with C.layers.default_options(init=C.glorot_uniform()):
        h = feature_var
        h = Convolution2D(5, ch, pad=True, bias=False)(h)
        h = BatchNormalization()(h)
        h = C.relu(h)
        for b in range(depth):
            h = res_block(h, block_depth, ch)
        policy = res_block(h, block_depth, ch)
        policy = C.relu(policy)
        policy = Convolution2D(3, (move_dim // 81), pad=True, bias=not spatial_bias)(policy)
        if spatial_bias:
            spatial_bias_param = C.parameter((move_dim // 81, 9, 9))
            policy = C.plus(policy, spatial_bias_param)
        policy = C.reshape(policy, shape=(move_dim,))
        value = res_block(h, block_depth, ch)
        value = C.relu(value)
        value = Dense(2, name="value")(value)

    return policy, value


def res_block_az(input_var, count, ch):
    h = input_var
    for i in range(count):
        h = Convolution2D(3, ch, pad=True, bias=False)(h)
        h = BatchNormalization()(h)
        if i < count - 1:
            h = C.relu(h)
    return C.relu(h + input_var)


def MPResNetAZ(feature_var, board_shape, move_dim, *, ch=16, depth=4, block_depth=2, move_hidden=256, fp16=False):
    """
    AlphaGoZeroに似せたモデル
    :param feature_var:
    :param board_shape:
    :param move_dim:
    :param ch:
    :param depth:
    :param block_depth:
    :param move_hidden:
    :return:
    """
    dtype = np.float16 if fp16 else np.float32
    with C.layers.default_options(init=C.glorot_uniform(), dtype=dtype):
        if fp16:
            h = C.cast(feature_var, dtype)
        else:
            h = feature_var
        h = Convolution2D(3, ch, pad=True, bias=False)(h)
        h = BatchNormalization()(h)
        h = C.relu(h)
        for b in range(depth):
            h = res_block_az(h, block_depth, ch)
        policy = Convolution2D(1, 2, pad=False, bias=False)(h)
        policy = BatchNormalization()(policy)
        policy = C.relu(policy)
        policy = Dense(move_dim)(policy)
        value = Convolution2D(1, 1, pad=False, bias=False)(h)
        value = BatchNormalization()(value)
        value = C.relu(value)
        value = Dense(move_hidden)(value)
        value = C.relu(value)
        value = Dense(2)(value)

    if fp16:
        policy = C.cast(policy, np.float32)
        value = C.cast(value, np.float32)
    return policy, value
