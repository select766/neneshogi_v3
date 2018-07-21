import cntk as C
from cntk.layers import Convolution2D, Dense, BatchNormalization


def res_block(input_var, count, ch):
    h = input_var
    for i in range(count):
        h = BatchNormalization()(input_var)
        h = C.relu(h)
        h = Convolution2D(3, ch, pad=True, bias=False)(h)
    return h + input_var


def ResNet(feature_var, board_shape, move_dim, *, ch=16, depth=4, block_depth=3, spatial_bias=False):
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
        value = Dense(1, activation=C.sigmoid, name="value")(value)

    return policy, value


def res_block2(input_var, ch):
    h = input_var
    h = BatchNormalization()(input_var)
    h = C.relu(h)
    h = Convolution2D(3, ch, pad=True, bias=False)(h)
    h = BatchNormalization()(h)
    h = C.relu(h)
    h = C.dropout(h, dropout_rate=0.1)
    h = Convolution2D(3, ch, pad=True, bias=False)(h)
    return h + input_var


def ResNet2(feature_var, board_shape, move_dim, *, ch=192, depth=4):
    """
    dlshogiのネットワーク構造(入力部は異なる)
    https://github.com/TadaoYamaoka/DeepLearningShogi/blob/b6e7538aebf2a5f41bd408b711b7d38c605b310e/dlshogi/policy_value_network.py
    :param feature_var:
    :param board_shape:
    :param move_dim:
    :param ch:
    :param depth:
    :param block_depth:
    :param spatial_bias:
    :return:
    """
    move_ch = move_dim // 81
    with C.layers.default_options(init=C.glorot_uniform()):
        h = feature_var
        h = Convolution2D(3, ch, pad=True, bias=False)(h)
        for b in range(depth):
            h = res_block2(h, ch)
        h = BatchNormalization()(h)
        h = C.relu(h)

        policy = h
        policy = Convolution2D(1, move_ch, pad=True, bias=False)(policy)
        spatial_bias_param = C.parameter((move_ch, 9, 9))
        policy = C.plus(policy, spatial_bias_param)
        policy = C.reshape(policy, shape=(move_dim,))

        value = h
        value = Convolution2D(1, move_ch, pad=True, bias=False)(value)
        spatial_bias_param = C.parameter((move_ch, 9, 9))
        value = C.plus(value, spatial_bias_param)
        value = C.reshape(value, shape=(move_dim,))
        value = BatchNormalization()(value)
        value = C.relu(value)
        value = Dense(256, activation=C.relu, name="value")(value)
        value = Dense(1, activation=C.sigmoid, name="value")(value)

    return policy, value
