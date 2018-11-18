"""
測定対象ソフトを、強さの分かっている複数の相手と戦わせてレートを推定する。

"""

import os
import sys
import argparse
from neneshogi.util import yaml_load, yaml_dump
from neneshogi.auto_match.auto_match import AutoMatch
from neneshogi.auto_match.auto_match_objects import Rule, EngineConfig, MatchResult, AutoMatchResult


def match(rule: Rule, base_config: EngineConfig, target_config: EngineConfig, log_prefix: str):
    """
    特定の基準ソフトと1回(ruleに依存し、複数対局)戦わせて、対戦成績を取得する
    :param rule:
    :param base_config:
    :param target_config:
    :param log_prefix:
    :return: base, targetの勝ち数
    """
    auto_match = AutoMatch(rule, [base_config, target_config])
    match_results = auto_match.run_matches(log_prefix)
    wins = [0, 0]
    for mr in match_results:
        if not mr.draw:
            wins[mr.winner] += 1
    return wins


def iter_match(rule: Rule, base_engines_rates, target_config, count: int, dst: str):
    os.makedirs(dst)
    cur_base = len(base_engines_rates) // 2
    vs_results = []
    for i in range(count):
        print(f"iter {i}: match to rate {base_engines_rates[cur_base]['rate']}")
        wins = match(rule, base_engines_rates[cur_base]["config"], target_config, os.path.join(dst, f"am_{i}"))
        print(f"base:target={wins[0]}:{wins[1]}")
        # winsはbase:targetなので注意
        vs_results.append({"base_idx": cur_base, "base_rate": base_engines_rates[cur_base]['rate'], "wins": wins})

        if wins[0] > wins[1]:
            cur_base -= 1
        elif wins[0] < wins[1]:
            cur_base += 1
        cur_base = max(0, min(len(base_engines_rates) - 1, cur_base))
        # クラッシュ対策に毎回保存しておく
        yaml_dump(vs_results, os.path.join(dst, "multi_measure_result.yaml"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rule")
    parser.add_argument("base_info")
    parser.add_argument("target_engine")
    parser.add_argument("count", type=int)
    parser.add_argument("dst")
    args = parser.parse_args()
    rule = Rule.load(args.rule)
    base_info = yaml_load(args.base_info)
    base_engines_rates = []
    for er in base_info["engines"]:
        base_engines_rates.append({"config": EngineConfig.load_obj(er["config"]), "rate": er["rate"]})
    target_config = EngineConfig.load(args.target_engine)
    iter_match(rule, base_engines_rates, target_config, args.count, args.dst)


if __name__ == "__main__":
    main()
