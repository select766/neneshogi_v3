# TCPによるプロセス間通信で、探索エンジンから呼ばれてpytorchモデルを実行する

r"""
nenefwd.bat バッチファイル例

@echo off
call C:\Users\foo\Anaconda3\Scripts\activate.bat C:\Users\foo\Anaconda3\envs\neneshogi2020
python -m neneshogi.nenefwd.nenefwd %*

"""
import argparse
import struct
import socket

import numpy as np
import torch
from neneshogi.model_loader import load_model

BOARD_SHAPE = (119, 9, 9)
BOARD_SIZE = 119 * 9 * 9
MOVE_DIM = 27 * 9 * 9


def read_batch_size(sock):
    # バッチサイズを読み取る
    number_len = 4  # sizeof(int32)
    buf = sock.recv(number_len)
    if len(buf) == 0:
        return 0
    while len(buf) < number_len:
        extbuf = sock.recv(number_len - len(buf))
        if len(extbuf) == 0:
            return 0
        buf += extbuf
    return struct.unpack("i", buf)[0]  # int32をパース


def read_input_array(sock, batch_size):
    buf = b""
    total_size = BOARD_SIZE * 4 * batch_size  # float32
    while len(buf) < total_size:
        extbuf = sock.recv(min(4096, total_size - len(buf)))
        if len(extbuf) == 0:
            raise ValueError
        buf += extbuf
    return np.frombuffer(buf, dtype=np.float32).reshape((batch_size,) + BOARD_SHAPE)


def request_loop(device, model, sock):
    while True:
        batch_size = read_batch_size(sock)
        if batch_size == 0:
            return
        board_array = read_input_array(sock, batch_size)
        with torch.no_grad():
            predicted = model(torch.from_numpy(board_array).to(device))
        policy_data = predicted[0].cpu().numpy()
        value_data = predicted[1].cpu().numpy()
        send_data = struct.pack("i", batch_size)
        for i in range(batch_size):
            send_data += policy_data[i].tobytes()
            send_data += value_data[i].tobytes()
        sock.sendall(send_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("gpu_id", type=int)
    parser.add_argument("hostname")
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
    model = load_model(args.checkpoint_dir, device)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.hostname, args.port))
    request_loop(device, model, sock)


if __name__ == '__main__':
    main()
