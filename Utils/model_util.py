import torch
import os
import torch.nn as nn

from Loader.Dataset3d import Dataset3d
from Loader.Loader2d import create_loader_2d
from tqdm import tqdm
from Unet.model import Unet


def load_unet_checkpoint(unet, optimizer_unet, scheduler_unet, ckpt_dir_unet, ckpt_fn_unet, device):
    print(os.path.join(ckpt_dir_unet, ckpt_fn_unet))
    state_dict = torch.load(os.path.join(ckpt_dir_unet, ckpt_fn_unet), map_location=device)
    unet.load_state_dict(state_dict['unet_state_dict'])
    optimizer_unet.load_state_dict(state_dict['optimizer_unet_state_dict'])
    scheduler_unet.load_state_dict(state_dict['scheduler_unet_state_dict'])
    epoch_loss = state_dict['epoch_loss']
    epoch = state_dict['epoch']
    global_step = state_dict['global_step']
    return unet, optimizer_unet, scheduler_unet, epoch_loss, epoch, global_step


def save_unet_checkpoint(unet, optimizer_unet, scheduler_unet, epoch_loss, epoch, global_step, ckpt_dir_unet, ckpt_fn_unet):
    state_dict = {
        'unet_state_dict': unet.state_dict(),
        'optimizer_unet_state_dict': optimizer_unet.state_dict(),
        'scheduler_unet_state_dict': scheduler_unet.state_dict(),
        'epoch_loss': epoch_loss,
        'epoch': epoch,
        'global_step': global_step
    }
    torch.save(state_dict, os.path.join(ckpt_dir_unet, ckpt_fn_unet))


def load_unet_checkpoint_pred(unet, ckpt_dir_unet, ckpt_fn_unet, device):
    print(os.path.join(ckpt_dir_unet, ckpt_fn_unet))
    state_dict = torch.load(os.path.join(ckpt_dir_unet, ckpt_fn_unet), map_location=device)
    unet.load_state_dict(state_dict['unet_state_dict'])
    return unet


def ckpt_parallel2single(n_channel_unet, n_class, ckpt_dir_unet, ckpt_dir_unet_crfrnn, ckpt_fn_unet, device):
    """
    when training, if we save model use, model.module.state_dict(), this could be avoided
    :param n_channel_unet:
    :param n_class:
    :param device:
    :return:
    """
    print("Parallel to single...")
    unet_parallel = Unet(n_channel_unet, n_class, bilinear=True)
    unet_parallel = nn.DataParallel(unet_parallel)
    unet_parallel.to(device)
    state_dict = torch.load(os.path.join(ckpt_dir_unet, ckpt_fn_unet), map_location=device)
    unet_parallel.load_state_dict(state_dict['unet_state_dict'])
    state_dict['unet_state_dict'] = unet_parallel.module.state_dict()
    torch.save(state_dict, os.path.join(ckpt_dir_unet_crfrnn, ckpt_fn_unet))




def val(unet, crfrnn, criterion, data_config, n_class, device, logger):
    print('Start validation')
    val_phase = 'val'
    unet.eval()
    # crfrnn.eval()

    dataset3d = Dataset3d(data_config['dataset'], phase='val')
    n_train = len(dataset3d)
    val_loss = 0
    with tqdm(total=n_train, desc="Validation execution.", unit='batch') as pbar:
        for data in dataset3d:
            print()
            logger.info(f"Processing {data['pid']}...")
            loader2d = create_loader_2d(data, data_config, val_phase)
            for batch_id, batch in enumerate(loader2d):
                img = batch['img_patch'].to(device=device, dtype=torch.float32)  # [N, n_channel_unet, H, W]
                mask_type = torch.float32 if n_class == 1 else torch.long
                mask_gt = batch['mask'].to(device=device, dtype=mask_type)  # [N, H, W]

                with torch.no_grad():
                    mask_pred = unet(img)

                loss_unet = criterion(mask_pred, mask_gt)
                loss = loss_unet
                loss_scalar = loss.item()
                val_loss += loss_scalar
                pbar.set_postfix(**{'loss (batch)': loss_scalar, 'pid': data['pid']})
            pbar.update()
    unet.train()
    # crfrnn.train()
    return val_loss
