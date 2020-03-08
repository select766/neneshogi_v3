"""
ダミーデータでの計算速度測定
"""

import argparse

import numpy as np
import time
import torch
from collections import defaultdict
from neneshogi.model_loader import load_model

BOARD_SHAPE = (119, 9, 9)
BOARD_SIZE = 119 * 9 * 9
MOVE_DIM = 27 * 9 * 9


def make_input(batch_size, device):
    return torch.zeros(batch_size, *BOARD_SHAPE).to(device)


def run_once(model, device, batch_size):
    with torch.no_grad():
        predicted = model(make_input(batch_size, device))
    policy_data = predicted[0].cpu().numpy()
    value_data = predicted[1].cpu().numpy()


def bench(model, device, min_bs, max_bs, count):
    times = defaultdict(list)
    for i in range(count):
        bs = i % (max_bs - min_bs + 1) + min_bs
        time_start = time.time()
        run_once(model, device, bs)
        time_end = time.time()
        times[bs].append(time_end - time_start)
    print("batch_size,avg_ms,nps")
    for bs in range(min_bs, max_bs + 1):
        timelist = times[bs]
        if len(timelist) > 0:
            avg_sec = np.mean(timelist)
            nps = int(bs / avg_sec)
            print(f"{bs},{avg_sec * 1000},{nps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("gpu_id", type=int)
    parser.add_argument("min_bs", type=int)
    parser.add_argument("max_bs", type=int)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
    model = load_model(args.checkpoint_dir, device)
    run_once(model, device, args.max_bs)
    bench(model, device, args.min_bs, args.max_bs, 1000)


if __name__ == '__main__':
    main()
