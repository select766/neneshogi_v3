from collections import defaultdict
from typing import Dict, List
from logging import getLogger
import numpy as np

logger = getLogger(__name__)


class TrainManager:
    next_action: str
    exit_lr: float  # schedulerにより、lrがこの値未満になったら終了
    quit_reason: str
    batch_size: int
    trained_samples: int
    val_frequency: int

    def __init__(self, batch_size: int, val_frequency: int, exit_lr: float = 0.0):
        self.next_action = "train"
        self.quit_reason = None
        self.batch_size = batch_size
        self.exit_lr = exit_lr
        self.trained_samples = 0
        self.val_frequency = val_frequency
        self.average_mean_criterions = defaultdict(float)

    def get_next_action(self):
        if self.quit_reason is not None:
            logger.warning(f"quit: {self.quit_reason}")
            return {"action": "quit", "reason": self.quit_reason}
        return {"action": self.next_action}

    def put_train_result(self):
        last_val_cycle = self.trained_samples // self.val_frequency
        self.trained_samples += self.batch_size
        do_val = self.trained_samples // self.val_frequency > last_val_cycle
        if do_val:
            self.next_action = "val"

    def put_val_result(self, new_lr: float):
        if new_lr < self.exit_lr:
            self.quit_reason = "lr below exit_lr"
        self.next_action = "train"
