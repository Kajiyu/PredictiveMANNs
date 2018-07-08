#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, size, dim=64):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self._w = size[1] / 8
        self._h = size[2] / 8
        self.dim = dim
        self.linear = nn.Linear(self._w*self._h*4*dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self._w*self._h*4*self.dim)
        output = self.linear(output)
        return output