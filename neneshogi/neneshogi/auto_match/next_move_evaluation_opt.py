"""
次の一手問題のパラメータを最適化する。
optuna利用。

ベースとなる設定ファイルの一部を書き換えて一致率測定を繰り返す。
"""

import os
import sys
import argparse
import copy
import subprocess
import re
from collections import OrderedDict
from typing import Dict, List, Tuple
import subprocess
import uuid
import optuna
from neneshogi.util import yaml_load, yaml_dump

base_config = {}


def call_evaluation(engine_config, out_dir, kifu_path):
    # サブディレクトリに設定ファイルとログを出力する
    trial_dir = os.path.join(out_dir, str(uuid.uuid4()))
    os.makedirs(trial_dir)
    trial_engine_config_path = os.path.join(trial_dir, "engine.yaml")
    yaml_dump(engine_config, trial_engine_config_path)
    stdout = subprocess.check_output(
        ["python", "-m", "neneshogi.auto_match.next_move_evaluation", kifu_path, trial_engine_config_path,
         "--log_prefix", os.path.join(trial_dir, "next_move_evaluation")])
    accuracy = None
    for line in stdout.decode("utf-8").splitlines():
        if line.startswith("accuracy:"):
            accuracy_str = line[len("accuracy:"):].strip()
            accuracy = float(accuracy_str)
    assert accuracy is not None
    return accuracy


def objective(trial):
    engine_config = copy.deepcopy(base_config["engine"])
    # suggestがスカラーのndarrayを出してくるので、そのままyamlに通すと可読性がないダンプがされてしまう
    for k, params in base_config["value_ranges"].items():
        v = getattr(trial, params["method"])(k, *params["args"])
        if params["method"] in ["suggest_uniform", "suggest_loguniform"]:
            v = float(v)
        elif params["method"] in ["suggest_int"]:
            v = int(v)
        engine_config["options"][k] = v
    accuracy = call_evaluation(engine_config, base_config["out_dir"], base_config["kifu"])
    return -accuracy  # minimize objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kifu")
    parser.add_argument("engine")
    parser.add_argument("value_ranges")
    parser.add_argument("out_dir")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_config["engine"] = yaml_load(args.engine)
    base_config["value_ranges"] = yaml_load(args.value_ranges)
    base_config["kifu"] = args.kifu
    base_config["out_dir"] = args.out_dir

    os.makedirs(base_config["out_dir"], exist_ok=True)
    study_name = 'next_move_evaluation_opt'  # Unique identifier of the study.
    # sqliteの指定は絶対パスでもokのようだ
    if args.resume:
        study = optuna.Study(study_name=study_name,
                             storage='sqlite:///' + os.path.join(base_config["out_dir"], "optuna.db"))
    else:
        study = optuna.create_study(study_name=study_name,
                                    storage='sqlite:///' + os.path.join(base_config["out_dir"], "optuna.db"))
    study.optimize(objective, n_trials=args.trials)


if __name__ == "__main__":
    main()
