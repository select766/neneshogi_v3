# Stubs for neneshogi_cpp (Python 3.6)
#
# NOTE: Manually implemented

from typing import Any, Tuple
import numpy as np

class Color:
    BLACK = ... # type: int
    WHITE = ... # type: int
    COLOR_NB = ... # type: int

class DNNConverter:
    def __init__(self, format_board: int, format_move: int):
        ...

    def board_shape(self) -> Tuple[int, int, int]:
        ...

    def move_shape(self) -> Tuple[int, int, int]:
        ...

    def set_packed_sfen(self, packed_sfen: bytes) -> bool:
        ...

    def set_sfen(self, sfen: str) -> None:
        ...

    def get_board_array(self) -> np.ndarray:
        ...

    def get_move_index(self, move: int) -> int:
        ...

    def reverse_move_index(self, move_index: int) -> int:
        ...
