"""
棋譜学習データをCNTKのCTF形式に変換
|labels 0 0 0 1 0 0 0 0 0 0 |features 0 255 0 123 ...
"""

import argparse
import struct
import numpy as np
from neneshogi_cpp import DNNConverter

RECORD_SIZE = 40


def vec2str(vec: np.ndarray) -> str:
    return " ".join(map(str, vec.tolist()))


def conv_record(cvt: DNNConverter, record: bytes) -> str:
    sfen, score, move, gamePly, game_result = struct.unpack("32shHHbx", record)
    cvt.set_packed_sfen(sfen)
    board_vec = cvt.get_board_array().flatten()
    move_vec = np.zeros(np.prod(cvt.move_shape()), dtype=np.uint8)
    move_vec[cvt.get_move_index(move)] = 1
    return f"|labels {vec2str(move_vec)} |features {vec2str(board_vec)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packed_sfen_value")
    parser.add_argument("dst")
    parser.add_argument("--count", type=int)
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    count = 0
    cvt = DNNConverter(0, 0)
    with open(args.packed_sfen_value, "rb") as inf:
        with open(args.dst, "w") as outf:
            if args.skip is not None:
                inf.seek(RECORD_SIZE * args.skip)
            while args.count is None or count < args.count:
                record = inf.read(RECORD_SIZE)
                if len(record) < RECORD_SIZE:
                    break
                line = conv_record(cvt, record)
                outf.write(line + "\n")
                count += 1


if __name__ == "__main__":
    main()
