import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

class _Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels, kernel_size=2,stride=2)
    def forward(self,x):
        x1 = self.up(x)
        return x1

class Up(nn.Module):
    def __init__(self,in_channels):
        super(Up, self).__init__()
        self.up1 = (_Up(in_channels, in_channels // 2))
        self.up2 = (_Up(in_channels // 2, in_channels // 4))
        self.up3 = (_Up(in_channels // 4, in_channels // 8))
        self.up4 = (_Up(in_channels // 8, 1))
    def forward(self,x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x
        
        