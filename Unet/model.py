import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import DoubleConv, Up, Down, OutConv


class Unet(nn.Module):
    def __init__(self, n_channel, n_class, bilinear=True):
        super(Unet, self).__init__()
        self.n_channel = n_channel
        self.n_class = n_class
        self.bilinear = bilinear

        self.in_layer = DoubleConv(n_channel, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out_layer = OutConv(64, n_class)

    def forward(self, x):
        x1 = self.in_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_layer(x)

        return logits
