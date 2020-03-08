"""
学習済みモデルをONNX形式でエクスポート
"""
import argparse

import numpy as np
import torch
from neneshogi.model_loader import load_model

BOARD_SHAPE = (119, 9, 9)
BOARD_SIZE = 119 * 9 * 9
MOVE_DIM = 27 * 9 * 9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("dst")
    args = parser.parse_args()
    device = "cpu"
    model = load_model(args.checkpoint_dir, device)
    torch.onnx.export(model, torch.randn(1, *BOARD_SHAPE), args.dst, export_params=True, opset_version=10,
                      verbose=True, do_constant_folding=True, input_names=["input"], output_names=["output_policy", "output_value"],
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output_policy' : {0 : 'batch_size'},
                                    'output_value' : {0 : 'batch_size'}})


if __name__ == '__main__':
    main()
