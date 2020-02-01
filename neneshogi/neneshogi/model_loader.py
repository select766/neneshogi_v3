# 学習済みモデルのロード
import os
import torch
from neneshogi import models
from neneshogi.util import yaml_load


def load_model(checkpoint_dir, device):
    # checkpoint_dir: path/to/training/checkpoints/train_012345
    model_config = yaml_load(os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), "model.yaml"))
    model_class = getattr(models, model_config["model"])
    model = model_class(board_shape=(119, 9, 9), move_dim=27 * 9 * 9, **model_config.get("kwargs", {}))
    model.eval()
    model.to(device)
    saved = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=str(device))
    model.load_state_dict(saved["model_state_dict"])
    return model
