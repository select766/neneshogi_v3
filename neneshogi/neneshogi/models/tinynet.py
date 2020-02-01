# 動作確認用小規模CNN
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    def __init__(self, board_shape, move_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(board_shape[0], 64, 5, padding=2)  # in,out,ksize
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.policy1 = nn.Conv2d(64, 64, 5, padding=2)
        self.policy1_bn = nn.BatchNorm2d(64)
        self.policy2 = nn.Conv2d(64, move_dim // 81, 1)
        self.value1 = nn.Linear(64 * 9 * 9, 256)
        self.value1_bn = nn.BatchNorm1d(256)
        self.value2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        xp = F.relu(self.policy1_bn(self.policy1(x)))
        xp = self.policy2(xp)
        xp = xp.view(xp.shape[0], -1)
        xv = x.view(x.shape[0], -1)
        xv = F.relu(self.value1_bn(self.value1(xv)))
        xv = self.value2(xv)
        return xp, xv
