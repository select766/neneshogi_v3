"""
sfen棋譜からランダムに次の一手評価用局面を抽出する。
1局から1局面、最初と最後の20手は除く。

startpos moves xxxx yyyy zzzz
の場合、yyyyまで指したあとの局面を入力として、bestmoveがzzzzであるかを評価する。
"""

import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--trim_start", type=int, default=20)
    parser.add_argument("--trim_end", type=int, default=20)
    args = parser.parse_args()
    with open(args.src) as f:
        all_kifus = f.readlines()
    random_kifus = random.sample(all_kifus, args.n_samples)
    results = []
    for kifu in random_kifus:
        moves = kifu.rstrip().split(" ")
        # trim_start==0なら初手の予想、trim_end==0なら1手詰みの予想問題が入る。
        # +2は"startpos moves"の分
        pos = random.randrange(args.trim_start + 3, len(moves) - args.trim_end)
        results.append(" ".join(moves[:pos]) + "\n")
    with open(args.dst, "w") as f:
        f.writelines(results)


if __name__ == "__main__":
    main()
