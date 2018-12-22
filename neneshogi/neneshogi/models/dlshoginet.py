"""
dlshogi(201812時点) のモデル構造移植実験
https://github.com/TadaoYamaoka/DeepLearningShogi/blob/a46c3d85644f490497227c9b530e4f92ba6343ff/dlshogi/policy_value_network.py

入力は駒配置62chの後ろに持ち駒57chがあり、これを最初にsplitして別個にconvolution
出力は移動元方向を表す27ch*9*9
"""

import cntk as C
from cntk.layers import Convolution2D, Dense, BatchNormalization


def res_block(x, k, dropout_ratio):
    h = x
    h = BatchNormalization()(h)
    h = C.relu(h)
    h = Convolution2D(3, k, pad=True, bias=False)(h)
    h = BatchNormalization()(h)
    h = C.relu(h)
    h = C.dropout(h, dropout_ratio)
    h = Convolution2D(3, k, pad=True, bias=False)(h)
    return h + x


def DlshogiNet(feature_var, board_shape, move_dim, *, k=192, dropout_ratio=0.1, fc1=256):
    with C.layers.default_options(init=C.glorot_uniform()):
        features1 = C.slice(feature_var, 0, 0, 62)
        features2 = C.slice(feature_var, 0, 62, 119)
        l1_1_1 = Convolution2D(3, k, pad=True, bias=False)(features1)
        l1_1_2 = Convolution2D(1, k, bias=False)(features1)
        l1_2 = Convolution2D(1, k, bias=False)(features2)
        h = l1_1_1 + l1_1_2 + l1_2
        for b in range(10):
            h = res_block(h, k, dropout_ratio)
        h = BatchNormalization()(h)
        h = C.relu(h)

        policy = Convolution2D(1, 27, pad=False, bias=False)(h)
        policy = C.flatten(policy)
        spatial_bias_param = C.parameter((27 * 9 * 9,))
        policy = C.plus(policy, spatial_bias_param)
        value = Convolution2D(1, 27, pad=False, bias=False)(h)
        value = BatchNormalization()(value)
        value = C.relu(value)
        value = Dense(fc1)(value)
        value = C.relu(value)
        value = Dense(2)(value)

    return policy, value
