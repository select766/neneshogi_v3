"""
PackedSfenファイルのシャッフル
複数ファイルを受け取り、全域にわたりシャッフル。
record_size単位で1レコードとみなす。ファイル末尾の余りは捨てる。

phase 1:
入力1ファイルずつに対して、各レコードを中間複数ファイルのいずれかにランダムに書きだす。
ファイルサイズ1つはbuffer_size程度になるようにする（ファイル数で調節）。

phase 2:
中間ファイル1つずつをオンメモリでシャッフルし、最終ファイルに追記していく。
"""

import os
import sys
import argparse
import numpy as np


def get_count(path, record_size):
    return os.path.getsize(path) // record_size


def write_tmp_files(src, tmp_files, buffer_size, record_size, item_count, n_tmp_files):
    """
    入力ファイル1つを複数ファイルにランダムに振り分ける
    :param src:
    :param tmp_files:
    :param buffer_size:
    :param record_size:
    :param item_count:
    :param n_tmp_files:
    :return:
    """
    # TODO buffer_size以下に分割して読む
    read_count = item_count * record_size  # 末尾削除
    records = np.fromfile(src, dtype=np.uint8, count=read_count).reshape(-1, record_size)
    assignments = np.random.randint(0, n_tmp_files, size=len(records))
    for i in range(n_tmp_files):
        ext_records = records[assignments == i]
        ext_records.tofile(tmp_files[i])


def write_dst_file(tmp_paths, dst, record_size):
    """
    全テンポラリファイルをシャッフルして出力ファイルに書き込む
    :param tmp_files:
    :param dst:
    :param record_size:
    :return:
    """
    with open(dst, "wb") as f:
        for tmp_path in tmp_paths:
            records = np.fromfile(tmp_path, dtype=np.uint8).reshape(-1, record_size)
            perm = np.random.permutation(len(records))
            shuffled = records[perm]
            del records
            del perm
            shuffled.tofile(f)
            del shuffled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dst")
    parser.add_argument("src", nargs="+")
    parser.add_argument("--record_size", type=int, default=72)
    parser.add_argument("--buffer_size", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()
    record_size = args.record_size
    buffer_size = args.buffer_size
    item_counts = [get_count(path, args.record_size) for path in args.src]
    total_count = sum(item_counts)
    item_per_tmp_file = buffer_size // record_size
    n_tmp_files = (total_count + item_per_tmp_file + 1) // item_per_tmp_file

    tmp_paths = [args.dst + f"_tmp{i}.bin" for i in range(n_tmp_files)]

    tmp_files = [open(path, "wb") for path in tmp_paths]
    for src_path, item_count in zip(args.src, item_counts):
        write_tmp_files(src_path, tmp_files, buffer_size, record_size, item_count, n_tmp_files)
    [f.close() for f in tmp_files]

    write_dst_file(tmp_paths, args.dst, record_size)

    [os.remove(path) for path in tmp_paths]


if __name__ == "__main__":
    main()
