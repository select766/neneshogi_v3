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
    RECORD_SIZE = 40

    def __init__(self, path: str, count: Optional[int] = None, skip: int = 0,
                 format_board: int = 0, format_move: int = 0, max_samples: int = INFINITELY_REPEAT):
        fsize = os.path.getsize(path)
        total_record_count = fsize // PackedSfenDataSource.RECORD_SIZE
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

        super().__init__()  # it references self.stream_infos

    def _seek_to_head(self):
        self._file.seek(self.skip * PackedSfenDataSource.RECORD_SIZE)

    def _read_record(self):
        raw = self._file.read(PackedSfenDataSource.RECORD_SIZE)
        packed_sfen, score, move, game_ply, game_result = struct.unpack("32shHHbx", raw)
        self._cvt.set_packed_sfen(packed_sfen)
        board = self._cvt.get_board_array()
        move_ary = np.zeros((self.move_dim,), dtype=np.float32)
        move_ary[self._cvt.get_move_index(move)] = 1
        return board, move_ary

    def stream_infos(self):
        return [self.board_info, self.move_info]

    def next_minibatch(self, num_samples: int, number_of_workers: int, worker_rank: int, device=None):
        assert number_of_workers == 1 and worker_rank == 0, "multi worker not supported"
        boards = []
        moves = []
        sweep_end = False
        while len(boards) < num_samples and self.total_generated_samples < self.max_samples and (
                not self.no_next_sweep):
            board, move = self._read_record()
            boards.append(board)
            moves.append(move)

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
        res = {
            self.board_info: MinibatchData(board_data, n_items, n_items, sweep_end),
            self.move_info: MinibatchData(move_data, n_items, n_items, sweep_end),
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
        self._file.seek((self.skip + self.record_idx) * PackedSfenDataSource.RECORD_SIZE)
