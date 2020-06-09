import numpy as np
import importlib
import os
import nrrd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import Augment.transforms as transforms
from Utils.data_util import mask_dict2mask


class Dataset3d(Dataset):
    def __init__(self, dataset_config, phase):

        self.phase = phase
        self.data_dir = dataset_config['data_dir']
        self.pid_cutoff_pair_csv_path = dataset_config['pid_cutoff_pair']
        self.roi_names = dataset_config['roi_names']
        self.roi_names_dict = dataset_config['roi_names_dict']
        if self.phase == 'train':
            self.used_pids_csv_path = dataset_config['train_used_pids']
            self.transformer_config = dataset_config['transformer']['3d']['train']
        elif self.phase == 'val':
            self.used_pids_csv_path = dataset_config['val_used_pids']
            self.transformer_config = dataset_config['transformer']['3d']['val']
        elif self.phase == 'pred':
            self.used_pids_csv_path = dataset_config['pred_used_pids']
            self.transformer_config = dataset_config['transformer']['3d']['pred']
        else:
            print("Not implement test phase dataset. -> NRRDDataset.__init__")
            exit()

        self.used_pids = np.genfromtxt(self.used_pids_csv_path, dtype=str, delimiter='\n')
        self.pid_cutoff_dict = self._get_pid_cutoff_dict()
        self.transformer = transforms.Transformer(self.transformer_config).play_transform()

    def _get_pid_cutoff_dict(self):
        pid_cutoff_pair = np.genfromtxt(self.pid_cutoff_pair_csv_path, dtype=str, delimiter='\n')
        if self.phase in ['train', 'val'] and pid_cutoff_pair is not None:
            pid_cutoff_dict = {}
            for pid_cutoff in pid_cutoff_pair:
                pid_cutoff_dict[pid_cutoff.split(',')[0]] = [int(pid_cutoff.split(',')[1]),
                                                             int(pid_cutoff.split(',')[2])]
        else:
            pid_cutoff_dict = None
        return pid_cutoff_dict

    def __len__(self):
        return len(self.used_pids)

    def __getitem__(self, idx):
        if self.phase in ['train', 'val']:
            self.pid = self.used_pids[idx]
            # load the raw img
            self.img, _ = nrrd.read(os.path.join(self.data_dir, '%s_img.nrrd' % self.pid))
            # load the mask and output the rough mask weight
            self.mask = self._load_mask(self.pid)
            # remove the regions out of cutoffs along z axis
            if self.pid_cutoff_dict is not None:
                self._cut_along_z()

            data = {'pid': self.pid,
                    'raw': self.img,
                    'label': self.mask
                    }
            data = self.transformer(data)

            # print(f"Extract 3d: {self.pid}: {self.img.shape}, {torch.max(data['label'])} -> 3d datatset")
            if torch.max(data['label']) > (len(self.roi_names) + 1):
                print("Augmentation make wrong labels")
                exit(1)

        elif self.phase == 'pred':
            self.pid = self.used_pids[idx]
            # load the raw img
            self.img, _ = nrrd.read(os.path.join(self.data_dir, '%s_img.nrrd' % self.pid))
            # load the mask and output the rough mask weight
            self.mask = self._load_mask(self.pid)
            data = {'pid': self.pid,
                    'raw': self.img,
                    'label': self.mask
                    }
            data = self.transformer(data)

        else:
            data = None
            print("Not implement test phase dataset. -> NRRDDataset.__init__")
            exit()

        return data

    def _load_mask(self, pid):
        mask = {}
        sel_n_roi = 0
        for j, roi in enumerate(self.roi_names):
            if os.path.isfile(os.path.join(self.data_dir, '%s_%s.nrrd' % (pid, roi))):
                sel_n_roi += 1
                m, _ = nrrd.read(os.path.join(self.data_dir, '%s_%s.nrrd' % (pid, roi)))
                mask[roi] = m
        # print(f"{pid} select {sel_n_roi} rois.")
        if len(mask) > 0:
            mask = mask_dict2mask(self.img.shape, mask, self.roi_names)
        else:
            mask = np.zeros_like(self.img)

        return mask

    def _cut_along_z(self):
        cutoff_list = self.pid_cutoff_dict[self.pid]
        self.img = self.img[cutoff_list[0]:cutoff_list[1], ...]
        if self.phase in ['train', 'val']:
            self.mask = self.mask[cutoff_list[0]:cutoff_list[1], ...]




