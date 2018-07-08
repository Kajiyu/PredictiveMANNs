#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, size, dim=64, z_dim=128):
        super(Generator, self).__init__()
        self._w = size[1] / 8
        self._h = size[2] / 8
        self.img_size = size
        self.dim = dim
        self.z_dim = z_dim
        preprocess = nn.Sequential(
            nn.Linear(z_dim, int(self._w * self._h * 4 * self.dim)),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(int(4 * self.dim), int(2 * self.dim), 2, stride=2),
            nn.BatchNorm2d(int(2 * self.dim)),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(int(2 * self.dim), self.dim, 2, stride=2),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.dim, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, self._w, self._h)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, self.img_size[1], self.img_size[2])