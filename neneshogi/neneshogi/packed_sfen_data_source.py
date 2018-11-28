"""
棋譜のPackedSfenValueデータから学習データをロードするUserMinibatchSource
"""
from typing import Optional
import os
import struct
import numpy as np
import cntk as C
from cntk.io import UserMinibatchSource, StreamInformation, MinibatchData, FULL_DATA_SWEEP, INFINITELY_REPEAT
from neneshogi_cpp import DNNConverter


class PackedSfenDataSource(UserMinibatchSource):
    def __init__(self, path: str, count: Optional[int] = None, skip: int = 0,
                 format_board: int = 0, format_move: int = 0, max_samples: int = INFINITELY_REPEAT,
                 multi_pv: int = 0, multi_pv_top: int = 0):
        fsize = os.path.getsize(path)
        self.multi_pv = multi_pv  # ファイルフォーマットでいくつのmulti_pvが格納されるか
        self.multi_pv_top = multi_pv_top if multi_pv_top > 0 else multi_pv  # multi_pvのうち実際に使う数
        self.record_size = 40 + 4 * self.multi_pv
        total_record_count = fsize // self.record_size
        if count is None:
            count = total_record_count - skip
        self._file = open(path, "rb")
        self.skip = skip
        self.count = count
        self.record_idx = 0
        self.max_samples = max_samples
        self.total_generated_samples = 0
        self.no_next_sweep = False
        self._seek_to_head()

        self._cvt = DNNConverter(format_board, format_move)
        self.board_shape = self._cvt.board_shape()
        self.board_dim = int(np.prod(self.board_shape))
        self.board_info = StreamInformation("board", 0, "dense", np.float32, self.board_shape)
        self.move_dim = int(np.prod(self._cvt.move_shape()))
        self.move_info = StreamInformation("move", 1, "dense", np.float32, (self.move_dim,))
        self.result_info = StreamInformation("result", 2, "dense", np.float32, (2,))
        self.winrate_info = StreamInformation("winrate", 3, "dense", np.float32, (2,))
        self.multi_move_info = StreamInformation("multi_move", 4, "dense", np.float32, (self.move_dim,))

        self._struct_psv = struct.Struct("32shHHbx")
        if self.multi_pv:
            self._struct_mpv = struct.Struct(f"{self.multi_pv}H{self.multi_pv}h")

        super().__init__()  # it references self.stream_infos

    def _seek_to_head(self):
        self._file.seek(self.skip * self.record_size)

    def _read_record(self):
        raw = self._file.read(self.record_size)
        packed_sfen, score, move, game_ply, game_result = self._struct_psv.unpack_from(raw)
        self._cvt.set_packed_sfen(packed_sfen)
        board = self._cvt.get_board_array()
        move_ary = np.zeros((self.move_dim,), dtype=np.float32)
        move_ary[self._cvt.get_move_index(move)] = 1
        multi_move_ary = np.full((self.move_dim,), -np.inf, dtype=np.float32)  # 指し手ごとの評価値(600で割ってある)
        if self.multi_pv:
            # // MultiPVで得られた指し手リスト。無効部分はMOVE_NONEとする。
            # u16 multipv_moves[LEARN_GENSFEN_MULTIPV];
            # // MultiPVの指し手ごとに対応する評価値。
            # s16 multipv_scores[LEARN_GENSFEN_MULTIPV];
            mp_move_score = self._struct_mpv.unpack_from(raw, 40)
            for i in range(self.multi_pv_top):
                mp_move = mp_move_score[i]
                if mp_move == 0:
                    break
                mp_score = mp_move_score[self.multi_pv + i]
                multi_move_ary[self._cvt.get_move_index(mp_move)] = mp_score / 600.0
        else:
            # 通常のPVを代入
            multi_move_ary[self._cvt.get_move_index(move)] = score / 600.0
        result_ary = np.zeros((2,), dtype=np.float32)  # [勝ちなら1, 負けなら1]
        if game_result < 0:
            result_ary[1] = 1.0
        elif game_result > 0:
            result_ary[0] = 1.0
        else:
            # 引き分けはないことを想定しているが、存在しても2値分類をクラッシュさせない
            result_ary[0] = 1.0
        winrate_ary = np.zeros((2,), dtype=np.float32)  # [勝率, 1-勝率]
        winrate = (np.tanh(score / 600.0 / 2) + 1) * 0.5  # 600で割ってsigmoid
        winrate_ary[0] = winrate
        winrate_ary[1] = 1.0 - winrate
        return board, move_ary, result_ary, winrate_ary, multi_move_ary

    def stream_infos(self):
        return [self.board_info, self.move_info, self.result_info]

    def next_minibatch(self, num_samples: int, number_of_workers: int, worker_rank: int, device=None):
        assert number_of_workers == 1 and worker_rank == 0, "multi worker not supported"
        boards = []
        moves = []
        results = []
        winrates = []
        multi_moves = []
        sweep_end = False
        while len(boards) < num_samples and self.total_generated_samples < self.max_samples and (
                not self.no_next_sweep):
            board, move, result, winrate, multi_move = self._read_record()
            boards.append(board)
            moves.append(move)
            results.append(result)
            winrates.append(winrate)
            multi_moves.append(multi_move)

            self.record_idx += 1
            self.total_generated_samples += 1
            if self.record_idx >= self.count:
                self._seek_to_head()
                self.record_idx = 0
                sweep_end = True
                if self.max_samples == FULL_DATA_SWEEP:
                    self.no_next_sweep = True
                    break

        n_items = len(boards)
        if n_items == 0:
            # When the maximum number of epochs/samples is exhausted, the return value is an empty dict.
            return {}

        board_data = C.Value(batch=np.array(boards), device=device)
        move_data = C.Value(batch=np.array(moves), device=device)
        result_data = C.Value(batch=np.array(results), device=device)
        winrate_data = C.Value(batch=np.array(winrates), device=device)
        multi_move_data = C.Value(batch=np.array(multi_moves), device=device)
        res = {
            self.board_info: MinibatchData(board_data, n_items, n_items, sweep_end),
            self.move_info: MinibatchData(move_data, n_items, n_items, sweep_end),
            self.result_info: MinibatchData(result_data, n_items, n_items, sweep_end),
            self.winrate_info: MinibatchData(winrate_data, n_items, n_items, sweep_end),
            self.multi_move_info: MinibatchData(multi_move_data, n_items, n_items, sweep_end),
        }
        return res

    def get_checkpoint_state(self):
        return {"record_idx": self.record_idx,
                "total_generated_samples": self.total_generated_samples,
                "no_next_sweep": self.no_next_sweep}

    def restore_from_checkpoint(self, state):
        self.record_idx = state["record_idx"]
        self.total_generated_samples = state["total_generated_samples"]
        self.no_next_sweep = state["no_next_sweep"]
        assert isinstance(self.record_idx, int)
        assert isinstance(self.total_generated_samples, int)
        self._file.seek((self.skip + self.record_idx) * self.record_size)
