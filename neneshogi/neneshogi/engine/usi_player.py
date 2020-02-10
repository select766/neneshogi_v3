# 将棋所で使えるプレイヤーの基底クラス。respond_goだけ実装すればいい。
from typing import Dict

import shogi


class UsiPlayer:
    board: shogi.Board
    options: Dict[str, str]

    def __init__(self):
        self.options = {}

    def respond_usi(self, tokens):
        print(f"id name neneshogi.{self.__class__.__name__}")
        print("id author select766")
        self.respond_option()
        print("usiok")

    def respond_option(self):
        # print("option name BookFile type string default public.bin")
        pass

    def respond_isready(self, tokens):
        print("readyok")

    def respond_position(self, tokens):
        # position startpos moves 7g7f 8c8d 7i6h
        self.board = shogi.Board()
        assert tokens[1] == "startpos"
        for move_usi in tokens[3:]:
            self.board.push_usi(move_usi)

    def respond_setoption(self, tokens):
        # setoption name BookFile value public.bin
        self.options[tokens[2]] = " ".join(tokens[4:])

    def respond_go(self, tokens):
        raise NotImplementedError

    def usi_loop(self):
        while True:
            try:
                msg = input()
            except EOFError:
                break
            tokens = msg.split(" ")
            if tokens[0] == "usi":
                self.respond_usi(tokens)
            elif tokens[0] == "setoption":
                self.respond_setoption(tokens)
            elif tokens[0] == "isready":
                self.respond_isready(tokens)
            elif tokens[0] == "usinewgame":
                pass
            elif tokens[0] == "position":
                self.respond_position(tokens)
            elif tokens[0] == "go":
                self.respond_go(tokens)
            elif tokens[0] == "quit":
                break
            else:
                pass
