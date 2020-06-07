import torch
import torch.nn as nn
import torch.nn.functional as F

from Unet.model import Unet
from CRF.crfrnn import CrfRnn


class UCRFNet(nn.Module):
    def __init__(self, n_channel, n_class, num_iteration, bilinear=True, crf_init_params=None):
        super(UCRFNet, self).__init__()

        self.unet = Unet(n_channel, n_class, bilinear)
        self.crfrnn = CrfRnn(n_class, num_iteration, crf_init_params)

    def forward(self, x):

        logits = self.unet(x)
        logits = self.crfrnn(x, logits)

        return logits
