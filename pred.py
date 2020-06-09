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
import re

from Unet.model import Unet
from CRF.crfrnn import CrfRnn
from Loader.Loader2d import create_loader_2d
from Loader.Dataset3d import Dataset3d
from Utils.util import create_logger
from Utils.model_util import load_unet_checkpoint_pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_config_path = '/home/kunhan/workspace/UCRFNet/Config/config_data.yml'
    pred_config_path = '/home/kunhan/workspace/UCRFNet/Config/config_pred.yml'
    data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
    pred_config = yaml.load(open(pred_config_path, 'r'), Loader=yaml.FullLoader)
    pred_config = pred_config['model']

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M")
    logging_path = os.path.join(pred_config['logging_dir'], 'logging_pred_' + date_time)
    logger = create_logger(logging_path)

    ###################################################################################
    # channel, class configuration
    ###################################################################################
    n_rois = len(data_config['dataset']['roi_names'])
    if n_rois > 1:
        # add background as the first class
        n_class = n_rois + 1
    else:
        n_class = 1
    n_channel_unet = data_config['dataset']['n_slice_unet']
    n_channel_crfrnn = data_config['dataset']['n_slice_crfrnn']

    ###################################################################################
    # construct net
    ###################################################################################
    logger.info("Creating net...")
    unet = Unet(n_channel_unet, n_class, bilinear=True)
    # crfrnn = CrfRnn(n_class, num_iterations=5).to('cpu')

    ###################################################################################
    # parallel model and data
    ###################################################################################
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        unet = nn.DataParallel(unet)
    unet.to(device)

    ###################################################################################
    # load model
    ###################################################################################
    unet = load_unet_checkpoint_pred(unet=unet,
                                     ckpt_dir=pred_config['ckpt_dir'],
                                     ckpt_fn_unet=pred_config['checkpoint_unet'],
                                     device=device)

    ###################################################################################
    # pred
    ###################################################################################
    os.makedirs(os.path.join(data_config['dataset']['pred_save_dir'], f'pred_{date_time}'), exist_ok=True)
    pred_phase = 'pred'
    datateset3d_pred = Dataset3d(data_config['dataset'], pred_phase)
    n_pred = len(datateset3d_pred)
    with tqdm(total=n_pred, desc="Pred execution", unit='batch') as pbar:
        for data in datateset3d_pred:
            print()
            logger.info(f"Processing {data['pid']}...")
            loader2d = create_loader_2d(data, data_config, pred_phase)
            n_batch = len(loader2d)
            mask_pred = torch.zeros_like(data['raw']).to(device=device, dtype=torch.int64)
            mask_gt = torch.zeros_like(data['raw']).to(device=device, dtype=torch.int64)
            for batch_id, batch in enumerate(loader2d):
                img_batch = batch['img_patch'].to(device=device, dtype=torch.float32)  # [N, n_channel_unet, H, W]
                mask_type = torch.float32 if n_class == 1 else torch.long
                mask_gt_batch = batch['mask'].to(device=device, dtype=mask_type)  # [N, H, W]
                target_slice_batch = batch['target_slice']
                with torch.no_grad():
                    mask_pred_batch = unet(img_batch)

                mask_gt[target_slice_batch] = mask_gt_batch
                mask_pred_batch = torch.argmax(mask_pred_batch, dim=1)
                mask_pred[target_slice_batch] = mask_pred_batch
                pbar.set_postfix(**{'pid': data['pid'], 'batch': f'{batch_id}/{n_batch}'})

            img_numpy = data['raw'].detach().cpu().numpy()
            mask_gt_numpy = mask_gt.detach().cpu().numpy()
            mask_pred_numpy = mask_pred.detach().cpu().numpy()
            np.save(os.path.join(data_config['dataset']['pred_save_dir'], f'pred_{date_time}', data['pid'] + '_img.npy'),
                    img_numpy)
            np.save(os.path.join(data_config['dataset']['pred_save_dir'], f'pred_{date_time}', data['pid'] + '_mask_gt.npy'),
                    mask_gt_numpy)
            np.save(os.path.join(data_config['dataset']['pred_save_dir'], f'pred_{date_time}', data['pid'] + '_mask_pred.npy'),
                    mask_pred_numpy)
            pbar.update()


if __name__ == '__main__':
    main()
