import torch.nn as nn
import torch.nn.functional as F


class ResBlockAZ(nn.Module):
    def __init__(self, count, ch):
        super().__init__()
        self.count = count
        self.ch = ch
        layers = []
        for i in range(count):
            layers.append(nn.Conv2d(ch, ch, 3, padding=1, bias=False))  # in,out,ksize
            layers.append(nn.BatchNorm2d(ch))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for i in range(self.count):
            h = self.layers[i * 2](h)
            h = self.layers[i * 2 + 1](h)
            if i < self.count - 1:
                h = F.relu(h)
        h = F.relu(h + x)
        return h


class ResNetAZ(nn.Module):
    def __init__(self, board_shape, move_dim, *, ch=16, depth=4, block_depth=2, move_hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(board_shape[0], ch, 3, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(ch)
        self.convs = nn.ModuleList([ResBlockAZ(block_depth, ch) for _ in range(depth)])
        self.p_conv1 = nn.Conv2d(ch, 2, 1, bias=False)
        self.p_conv1_bn = nn.BatchNorm2d(2)
        self.p_fc2 = nn.Linear(2 * 9 * 9, move_dim)
        self.v_conv1 = nn.Conv2d(ch, 1, 1, bias=False)
        self.v_conv1_bn = nn.BatchNorm2d(1)
        self.v_fc2 = nn.Linear(1 * 9 * 9, move_hidden)
        self.v_fc3 = nn.Linear(move_hidden, 2)

    def forward(self, x):
        h = x
        h = F.relu(self.conv1_bn(self.conv1(h)))
        for i in range(len(self.convs)):
            h = self.convs[i](h)
        hp = h
        hp = F.relu(self.p_conv1_bn(self.p_conv1(hp)))
        hp = hp.view(hp.shape[0], -1)
        hp = self.p_fc2(hp)
        hv = h
        hv = F.relu(self.v_conv1_bn(self.v_conv1(hv)))
        hv = hv.view(hv.shape[0], -1)
        hv = F.relu(self.v_fc2(hv))
        hv = self.v_fc3(hv)
        return hp, hv
