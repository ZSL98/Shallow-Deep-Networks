import torch
import math

import torch.nn as nn


def fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class ee_module(nn.Module):
    def __init__(self):
        super(ee_module, self).__init__()
        self.layer1 = fc_layer(7*7*512, 4096)
        self.layer2 = fc_layer(4096, 4096)
        self.layer3 = nn.Linear(4096, 1000)
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out

