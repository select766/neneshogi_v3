from typing import Dict, List
import hashlib
from neneshogi.util import yaml_load


class Rule:
    n_match: int
    max_moves: int
    max_go_time: float
    init_book: str

    def __init__(self):
        self.n_match = 1
        self.max_moves = 256
        self.max_go_time = 60
        self.init_book = None

    @classmethod
    def load(cls, rule_file) -> "Rule":
        item_dict = yaml_load(rule_file)
        inst = cls()
        inst.__dict__.update(item_dict)
        return inst


class EngineConfig:
    config_id: str  # 設定を一意に識別する文字列。設定ファイルのsha1sumをデフォルトで割り当てる。
    path: str
    go: str
    options: Dict[str, str]
    env: Dict[str, str]

    def __init__(self):
        self.config_id = None
        self.path = None
        self.go = "go btime 0 wtime 0 byoyomi 1000"
        self.options = {"USI_Ponder": "false", "USI_Hash": "256"}
        self.env = {}

    @classmethod
    def load(cls, engine_file) -> "EngineConfig":
        config_id_cand = hashlib.sha1(open(engine_file, "rb").read()).hexdigest()
        item_dict = yaml_load(engine_file)
        inst = cls()
        inst.__dict__.update(item_dict)
        if not inst.config_id:
            inst.config_id = config_id_cand
        return inst


class MatchResult:
    draw: bool
    winner: int
    gameover_reason: str
    kifu: List[str]

    def __init__(self, draw, winner, gameover_reason, kifu):
        self.draw = draw
        self.winner = winner
        self.gameover_reason = gameover_reason
        self.kifu = kifu


class AutoMatchResult:
    rule: Rule
    engine_config_list: List[EngineConfig]
    match_results: List[MatchResult]
