# python-shogiによる将棋所で使える、方策モデルそのままのプレイヤー

import numpy as np
import torch

from neneshogi.engine.usi_player import UsiPlayer
from neneshogi.model_loader import load_model
from neneshogi_cpp import DNNConverter


class PolicyPlayer(UsiPlayer):
    device: torch.device
    model: torch.nn.Module
    cvt: DNNConverter

    def respond_option(self):
        print("option name Checkpoint type string default <empty>")
        print("option name Device type string default cpu")

    def respond_isready(self, tokens):
        self.device = torch.device(self.options["Device"])
        self.model = load_model(self.options["Checkpoint"], self.device)
        self.cvt = DNNConverter(1, 1)
        print("readyok")

    def respond_go(self, tokens):
        bestmove = "resign"
        moves = list(self.board.legal_moves)
        if len(moves) > 0:
            self.cvt.set_sfen(self.board.sfen())
            board_array = self.cvt.get_board_array()
            with torch.no_grad():
                predicted = self.model(torch.from_numpy(board_array[np.newaxis, ...]).to(self.device))
            policy_vec = predicted[0].cpu().numpy()[0]
            value_vec = predicted[1].cpu().numpy()[0]
            move_usis = []
            policy_logits = []
            for move in moves:
                policy_logits.append(policy_vec[self.cvt.get_move_index(self.cvt.move_from_usi(move.usi()))])
                move_usis.append(move.usi())
            choice_idx = int(np.argmax(policy_logits))
            bestmove = move_usis[choice_idx]
            # value_vecをsoftmaxした結果のindex=0が勝率
            cp = int(value_vec[0] - value_vec[1] * 600.0)
            print(f"info score cp {cp} pv {bestmove}")
        print(f"bestmove {bestmove}")


def main():
    player = PolicyPlayer()
    player.usi_loop()


if __name__ == '__main__':
    main()
