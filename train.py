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
from Utils.model_util import load_unet_checkpoint, save_unet_checkpoint, val, ckpt_parallel2single


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_config_path = '/home/kunhan/workspace/UCRFNet/Config/config_data.yml'
    train_config_path = '/home/kunhan/workspace/UCRFNet/Config/config_train.yml'
    data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, 'r'), Loader=yaml.FullLoader)
    train_config = train_config['train']

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M")
    logging_path = os.path.join(train_config['logging_dir'], 'logging_train_' + date_time)
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

    for name in os.listdir(train_config['ckpt_dir_unet']):
        if os.path.isfile(os.path.join(train_config['ckpt_dir_unet'], name)):
            ckpt_parallel2single(n_channel_unet, n_class, train_config['ckpt_dir_unet'], train_config['ckpt_dir_unet_crfrnn'], name, device)
    return

    ###################################################################################
    # construct net
    ###################################################################################
    logger.info("Creating net...")
    unet = Unet(n_channel_unet, n_class, bilinear=True)
    crfrnn = CrfRnn(n_class, num_iterations=5).to('cpu')

    ###################################################################################
    # parallel model and data
    ###################################################################################
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        unet = nn.DataParallel(unet)
    unet.to(device)

    ###################################################################################
    # criterion, optimizer, scheduler
    ###################################################################################
    if n_class > 1:
        # input: (N, C) or (N, C, H, W)
        # target: (N) or (N, H, W)
        criterion = nn.CrossEntropyLoss()
    else:
        # input and target have the same shape. (N, *)
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Creating optimizer...")
    optimizer_unet = optim.RMSprop(unet.parameters(), lr=train_config['lr'], weight_decay=1e-8, momentum=0.9)
    # optimizer_crfrnn = optim.RMSprop(unet.parameters(), lr=train_config['lr'], weight_decay=1e-8, momentum=0.9)
    logger.info("Creating scheduler...")
    scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'min', patience=2)
    # scheduler_crfrnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_crfrnn, 'min', patience=2)

    ###################################################################################
    # SummaryWriter
    ###################################################################################
    logger.info("Creating writer")
    writer = SummaryWriter(comment=f"LR_{train_config['lr']}_BS_{train_config['n_epoch']}")

    ###################################################################################
    # train setup
    ###################################################################################
    global_step = 0
    best_loss = np.inf
    train_phase = 'train'
    epoch_start = 0

    ###################################################################################
    # load previous model
    ###################################################################################
    if train_config['load_checkpoint'] is not None:
        print("Loading net...")
        logging.info("Loading net...")
        # load unet from checkpoint
        unet, optimizer_unet, scheduler_unet, epoch_loss, epoch_start, global_step = load_unet_checkpoint(unet=unet,
                                                                                                          optimizer_unet=optimizer_unet,
                                                                                                          scheduler_unet=scheduler_unet,
                                                                                                          ckpt_dir_unet=
                                                                                                          train_config[
                                                                                                              'ckpt_dir_unet'],
                                                                                                          ckpt_fn_unet=
                                                                                                          train_config[
                                                                                                              'checkpoint_unet'],
                                                                                                          device=device)
        logger.info(f"Start from epoch: {epoch_start}, global_step: {global_step}")
        print(f"Start from epoch: {epoch_start}, global step{global_step}")
    ###################################################################################
    # train
    ###################################################################################
    for epoch in range(epoch_start, train_config['n_epoch']):
        epoch_loss = 0
        dataset3d = Dataset3d(data_config['dataset'], train_phase)
        n_train = len(dataset3d)
        logger.info(f"Epoch: {epoch}/{train_config['n_epoch']}")
        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{train_config['n_epoch']}", unit='batch') as pbar:
            for data in dataset3d:
                print()
                logger.info(f"Processing {data['pid']}...")
                loader2d = create_loader_2d(data, data_config, train_phase)
                for batch_id, batch in enumerate(loader2d):
                    img = batch['img_patch'].to(device=device, dtype=torch.float32)  # [N, n_channel_unet, H, W]
                    mask_type = torch.float32 if n_class == 1 else torch.long
                    mask_gt = batch['mask'].to(device=device, dtype=mask_type)  # [N, H, W]
                    mask_pred = unet(img)

                    loss_unet = criterion(mask_pred, mask_gt)
                    loss = loss_unet
                    optimizer_unet.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(unet.parameters(), 0.1)
                    optimizer_unet.step()
                    global_step += 1

                    loss_scalar = loss.item()
                    epoch_loss += loss_scalar

                    logger.info(f"\tBatch: {batch_id}/{len(loader2d)}, Batch Loss: {loss_scalar}")
                    pbar.set_postfix(**{'loss (batch)': loss_scalar, 'pid': data['pid']})

                    if (global_step + 1) % train_config['write_summary_loss_batch_step'] == 0:
                        writer.add_scalar('Loss_unet_batch/train', loss_scalar, global_step)
                    if (global_step + 1) % train_config['write_summary_2d_batch_step'] == 0:
                        images_grid = torchvision.utils.make_grid(torch.unsqueeze(img[:, n_channel_unet // 2], 1))
                        gt_masks_grid = torchvision.utils.make_grid(torch.unsqueeze(mask_gt, 1))
                        pred_mask_grid = torchvision.utils.make_grid(torch.argmax(mask_pred, dim=1, keepdim=True))
                        writer.add_image('images', images_grid, global_step)
                        writer.add_image('gt_masks', gt_masks_grid, global_step)
                        writer.add_image('pred_masks', pred_mask_grid, global_step)

                pbar.update()
        # logging
        if (epoch + 1) % train_config['logging_epoch_step'] == 0:
            writer.add_scalar('Loss_unet_epoch/train', epoch_loss, global_step)
            print(f"Epoch: {epoch}/{train_config['n_epoch']}, Epoch Loss: {epoch_loss}")
            logger.info(f"Epoch: {epoch}/{train_config['n_epoch']}, Epoch Loss: {epoch_loss}")
        # do validation and save model
        if (epoch + 1) % train_config['save_model_epoch_step'] == 0:
            # validation
            val_loss = val(unet, crfrnn, criterion, data_config, n_class, device, logger)
            writer.add_scalar('Loss_unet_epoch/val', val_loss, global_step)
            print(f"Epoch: {epoch}/{train_config['n_epoch']}, Validation Loss: {val_loss}")
            logger.info(f"Epoch: {epoch}/{train_config['n_epoch']}, Validation Loss: {val_loss}")

            # save model
            scheduler_unet.step(val_loss)
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                save_unet_checkpoint(unet=unet, optimizer_unet=optimizer_unet, scheduler_unet=scheduler_unet,
                                     epoch_loss=epoch_loss, epoch=epoch, global_step=global_step,
                                     ckpt_dir_unet=train_config['ckpt_dir_unet'],
                                     ckpt_fn_unet='best_unet_ckpt' + date_time + '.ckpt')
            save_unet_checkpoint(unet=unet, optimizer_unet=optimizer_unet, scheduler_unet=scheduler_unet,
                                 epoch_loss=epoch_loss, epoch=epoch, global_step=global_step,
                                 ckpt_dir_unet=train_config['ckpt_dir_unet'],
                                 ckpt_fn_unet=f'unet_ckpt_{date_time}_Epoch_{epoch}.ckpt')
            print(f"Epoch: {epoch}/{train_config['n_epoch']}, Save model.")
            logger.info(f"Epoch: {epoch}/{train_config['n_epoch']}, Save model.")
    writer.close()


if __name__ == '__main__':
    main()
