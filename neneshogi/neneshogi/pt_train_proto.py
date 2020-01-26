import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from neneshogi_cpp import DNNConverter


class PackedSfenDataset(Dataset):
    def __init__(self, path, count, skip=0):
        self._record_size = 40
        self._file = open(path, "rb")
        self._current_file_offset = 0
        self.count = count
        self.skip = skip
        self._cvt = DNNConverter(1, 1)  # format_board, format_move
        self.board_shape = self._cvt.board_shape()
        self.board_dim = int(np.prod(self.board_shape))
        self.move_dim = int(np.prod(self._cvt.move_shape()))
        self._struct_psv = struct.Struct("32shHHbx")

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO: idxが配列の時
        board, move_index = self._read_record(idx)
        return {'board': board, 'move_index': move_index}

    def _read_record(self, idx):
        ofs = (self.skip + idx) * self._record_size
        if ofs != self._current_file_offset:
            self._file.seek(ofs)
        rec = self._file.read(self._record_size)
        self._current_file_offset = ofs + self._record_size
        packed_sfen, score, move, game_ply, game_result = self._struct_psv.unpack_from(rec)
        self._cvt.set_packed_sfen(packed_sfen)
        board = self._cvt.get_board_array()
        move_index = self._cvt.get_move_index(move)
        return board, move_index


trainset = PackedSfenDataset(r"D:\tmp\kifudecode\ALN_293_0.bin", 100000, skip=10000)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=False, num_workers=0)

testset = PackedSfenDataset(r"D:\tmp\kifudecode\ALN_293_0.bin", 10000, skip=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(119, 64, 3, padding=1)  # in,out,ksize
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.policy = nn.Conv2d(64, 27, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.policy(x)
        x = x.view(-1, 9 * 9 * 27)
        return x


net = PolicyNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['board']
        labels = data['move_index']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
