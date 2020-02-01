import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from neneshogi_cpp import DNNConverter


class PackedSfenDataset(Dataset):
    """
    やねうら王形式のPacked Sfen棋譜を読み取るデータセット
    """
    def __init__(self, path, count, skip=0):
        self._record_size = 40
        self._file = open(path, "rb")
        self._current_file_offset = 0
        self.count = count
        self.skip = skip
        self._cvt = DNNConverter(1, 1)  # format_board, format_move
        self.board_shape = self._cvt.board_shape()
        self.board_dim = int(np.prod(self.board_shape))
        self.move_dim = int(np.prod(self._cvt.move_shape()))
        self._struct_psv = struct.Struct("32shHHbx")

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO: idxが配列の時
        board, move_index, game_result = self._read_record(idx)
        return {'board': board, 'move_index': move_index, 'game_result': game_result}

    def _read_record(self, idx):
        ofs = (self.skip + idx) * self._record_size
        if ofs != self._current_file_offset:
            self._file.seek(ofs)
        rec = self._file.read(self._record_size)
        self._current_file_offset = ofs + self._record_size
        packed_sfen, score, move, game_ply, game_result = self._struct_psv.unpack_from(rec)
        self._cvt.set_packed_sfen(packed_sfen)
        board = self._cvt.get_board_array()
        move_index = self._cvt.get_move_index(move)
        # game_resultは勝ち負け引き分けが1,0,-1になっているがここでは勝ちをラベル0、それ以外をラベル1としておく
        game_result_binary = 0 if game_result >= 1 else 1
        return board, move_index, game_result_binary
