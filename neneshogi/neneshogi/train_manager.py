from collections import defaultdict
from typing import Dict, List
from logging import getLogger
import numpy as np

logger = getLogger(__name__)


class TrainManager:
    next_action: str
    main_criterion: str
    lr: float
    lr_reduce_ratio: float
    lr_reduce_average_count: int
    lr_reduce_threshold: float
    lr_reduce_val_results: List[float]
    min_lr: float
    quit_reason: str
    epoch_size: int
    batch_size: int
    trained_samples: int
    val_frequency: int
    first_diverge_check_size: int
    diverge_criterion: Dict[str, float]
    average_mean_criterions: Dict[str, float]  # train errorの移動平均

    def __init__(self, epoch_size: int, batch_size: int, val_frequency: int,
                 initial_lr: float, min_lr: float, first_diverge_check_size: int,
                 diverge_criterion: Dict[str, float], lr_reduce_ratio: float,
                 lr_reduce_average_count: int, lr_reduce_threshold: float):
        self.next_action = "train"
        self.main_criterion = "policy_cle"
        self.last_val_main_criterion = None
        self.lr = initial_lr
        self.lr_reduce_ratio = lr_reduce_ratio
        self.lr_reduce_average_count = lr_reduce_average_count
        self.lr_reduce_threshold = lr_reduce_threshold
        self.lr_reduce_val_results = []
        self.min_lr = min_lr
        self.quit_reason = None
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.trained_samples = 0
        self.val_frequency = val_frequency
        self.first_diverge_check_size = first_diverge_check_size
        self.diverge_criterion = diverge_criterion
        self.average_mean_criterions = defaultdict(float)

    def get_next_action(self):
        if self.quit_reason is not None:
            logger.warning(f"quit: {self.quit_reason}")
            return {"action": "quit", "reason": self.quit_reason}
        return {"action": self.next_action, "lr": self.lr}

    def _check_diverge(self, mean_criterions: Dict[str, float]) -> bool:
        """
        発散基準を満たすかどうかチェックする。
        指定されたエラー率が閾値以上なら発散したとみなす。
        :param mean_criterions:
        :return:
        """
        for key, thres in self.diverge_criterion.items():
            if mean_criterions[key] > thres:
                return True
        return False

    def put_train_result(self, mean_criterions: Dict[str, float]):
        last_val_cycle = self.trained_samples // self.val_frequency
        self.trained_samples += self.batch_size
        for k, v in mean_criterions.items():
            self.average_mean_criterions[k] = self.average_mean_criterions[k] * 0.99 + v * 0.01
        if self.trained_samples >= self.first_diverge_check_size:
            # 発散チェック
            # サンプルによっては特別に悪い場合があるので、移動平均で判定
            if self._check_diverge(self.average_mean_criterions):
                self.quit_reason = "diverge"
        do_val = self.trained_samples // self.val_frequency > last_val_cycle
        if do_val:
            logger.info(f"average_mean_criterions, {self.average_mean_criterions}")
            self.next_action = "val"

    def put_val_result(self, mean_criterions: Dict[str, float]):
        main_score = mean_criterions[self.main_criterion]
        self.lr_reduce_val_results.append(main_score)
        if len(self.lr_reduce_val_results) >= self.lr_reduce_average_count * 2 and \
                len(self.lr_reduce_val_results) % self.lr_reduce_average_count == 0:
            # x = scores[-10:], y = scores[-20:-10]として、
            # (y - x) / y <  lr_reduce_thresholdならlrを下げる
            x = np.mean(self.lr_reduce_val_results[-self.lr_reduce_average_count:])
            y = np.mean(self.lr_reduce_val_results[-self.lr_reduce_average_count * 2:-self.lr_reduce_average_count])
            improve_ratio = 1.0 - x / y
            logger.info(f"val score improvement: {improve_ratio}")
            if improve_ratio < self.lr_reduce_threshold:
                # 改善がほとんどない
                # lrを下げる
                logger.info("reduce lr")
                self.lr /= self.lr_reduce_ratio
                if self.lr < self.min_lr:
                    # 学習終了
                    self.quit_reason = "lr_below_min"

        self.last_val_main_criterion = main_score
        self.next_action = "train"
