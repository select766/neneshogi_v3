import cntk as C
from cntk.layers import Convolution2D, Dense, BatchNormalization


def res_block(input_var, count, ch):
    h = input_var
    for i in range(count):
        h = BatchNormalization()(input_var)
        h = C.relu(h)
        h = Convolution2D(3, ch, pad=True, bias=False)(h)
    return h + input_var


def ResNet(feature_var, board_shape, move_dim, *, ch=16, depth=4, block_depth=3):
    with C.layers.default_options(init=C.glorot_uniform()):
        h = feature_var
        h = Convolution2D(5, ch, pad=True, bias=False)(h)
        h = BatchNormalization()(h)
        h = C.relu(h)
        for b in range(depth):
            h = res_block(h, block_depth, ch)
        policy = res_block(h, block_depth, ch)
        policy = C.relu(policy)
        policy = Convolution2D(3, (move_dim // 81), pad=True, bias=True)(policy)
        policy = C.reshape(policy, shape=(move_dim,))
        value = res_block(h, block_depth, ch)
        value = C.relu(value)
        value = Dense(1, activation=C.tanh, name="value")(value)

    return policy, value
