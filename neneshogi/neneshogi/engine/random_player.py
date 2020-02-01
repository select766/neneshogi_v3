# python-shogiによる将棋所で使えるランダムプレイヤー

import random

from neneshogi.engine.usi_player import UsiPlayer


class RandomPlayer(UsiPlayer):
    def respond_go(self, tokens):
        bestmove = "resign"
        moves = list(self.board.legal_moves)
        if len(moves) > 0:
            move = random.choice(moves)
            bestmove = move.usi()
        print(f"bestmove {bestmove}")


def main():
    player = RandomPlayer()
    player.usi_loop()


if __name__ == '__main__':
    main()
