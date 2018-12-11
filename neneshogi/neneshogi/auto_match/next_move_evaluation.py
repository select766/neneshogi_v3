"""
エンジンを1つ起動し、次の一手の正解率を評価する。
"""
import os
import sys
import argparse
import subprocess
import re
from collections import OrderedDict
from typing import Dict, List
import time
from datetime import datetime
import random
import threading

from neneshogi.auto_match.auto_match import AutoMatch
from neneshogi.util import yaml_load, yaml_dump

import shogi
from .auto_match_objects import Rule, EngineConfig, MatchResult, AutoMatchResult


class NextMoveEvaluator(AutoMatch):
    def __init__(self, engine_config: EngineConfig):
        super().__init__(Rule(), [engine_config])  # Ruleはダミー

    def _run_single_evaluation(self, kifu: str) -> str:
        board = shogi.Board()
        for move in kifu.rstrip().split(" ")[2:-1]:  # startpos movesと最後の手以外
            board.push_usi(move)
        bestmove = self._get_bestmove(board, 0)
        return bestmove

    def run_evaluation(self, log_prefix: str, kifus: List[str]) -> List[str]:
        self.engine_handles = []
        bestmoves = []
        self._log_file = open(log_prefix + ".log", "a")
        cleanup_ctr = 0
        try:
            for i, ec in enumerate(self.engine_config_list):
                self._log(f"Initializing engine {i}")
                self.engine_handles.append(self._exec_engine(ec))
                self._init_engine(i, ec)
                self._isready_engine(i)
                self._engine_write(i, "usinewgame")
            for kifu in kifus:
                bestmove = self._run_single_evaluation(kifu)
                bestmoves.append(bestmove)
                cleanup_ctr += 1
                if cleanup_ctr >= 100:
                    # 置換表フルを避けるため、100手思考したところでリセット
                    self._engine_write(0, f"gameover win")
                    self._isready_engine(0)
                    self._engine_write(0, "usinewgame")
                    cleanup_ctr = 0
            for i in range(len(self.engine_config_list)):
                self._log(f"Closing engine {i}")
                self._quit_engine(i)
            self._log("Finished task")
        except Exception as ex:
            self._log(f"Exception: {ex}")
            raise
        finally:
            self._log_file.close()
            self._log_file = None
            yaml_dump({"bestmoves": bestmoves}, log_prefix + ".yaml")
        return bestmoves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kifu")
    parser.add_argument("engine")
    parser.add_argument("--log_prefix")
    args = parser.parse_args()
    log_prefix = args.log_prefix or f"data/next_move_evaluation/next_move_evaluation_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
    engine_config = EngineConfig.load(args.engine)
    with open(args.kifu) as f:
        kifus = f.readlines()
    auto_match = NextMoveEvaluator(engine_config)
    auto_match.run_evaluation(log_prefix, kifus)


if __name__ == "__main__":
    main()
