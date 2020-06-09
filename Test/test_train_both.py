import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import datetime
import numpy as np
import logging
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision

from Unet.model import Unet
from CRF.crfrnn import CrfRnn
from Loader.Loader2d import create_loader_2d
from Loader.Dataset3d import Dataset3d
from Utils.util import create_logger
from Utils.model_util import load_unet_checkpoint, save_unet_checkpoint, val, load_unet_checkpoint_pred

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = '/home/kunhan/workspace/UCRFNet/Ckpt'
    ckpt_unet_fn = 'unet_ckpt_2020-06-08-22-56_Epoch_6.ckpt'

    n_class = 7
    n_slice = 5
    unet = Unet(n_slice, n_class, bilinear=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        unet = nn.DataParallel(unet)
    unet.to(device)

    unt = load_unet_checkpoint_pred(unet, ckpt_dir, ckpt_unet_fn, device)
    unt2 = Unet(n_slice, n_class, bilinear=True).to(device)
    unt2.load_state_dict(unt.module.state_dict())



if __name__ == '__main__':
    main()