# python-shogiによる将棋所で使えるランダムプレイヤー

import random
import shogi


class RandomPlayer:
    board: shogi.Board

    def respond_usi(self, tokens):
        print("id name random_player")
        print("id author select766")
        print("usiok")

    def respond_isready(self, tokens):
        print("readyok")

    def respond_position(self, tokens):
        # position startpos moves 7g7f 8c8d 7i6h
        self.board = shogi.Board()
        assert tokens[1] == "startpos"
        for move_usi in tokens[3:]:
            self.board.push_usi(move_usi)

    def respond_go(self, tokens):
        bestmove = "resign"
        moves = list(self.board.legal_moves)
        if len(moves) > 0:
            move = random.choice(moves)
            bestmove = move.usi()
        print(f"bestmove {bestmove}")

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
                pass
            elif tokens[0] == "isready":
                self.respond_isready(tokens)
            elif tokens[0] == "usinewgame":
                pass
            elif tokens[0] == "position":
                self.respond_position(tokens)
            elif tokens[0] == "go":
                self.respond_go(tokens)
            else:
                pass


def main():
    player = RandomPlayer()
    player.usi_loop()


if __name__ == '__main__':
    main()
