import torch
from torch.utils.data import Dataset, DataLoader


class Dataset2d(Dataset):
    def __init__(self, data, dataset_config, phase):
        self.phase = phase
        self.n_slice = dataset_config['n_slice_unet']
        self.roi_names = dataset_config['roi_names']
        self.data = data
        self.img = data['raw']
        self.mask = data['label']
        self.z_slice = self._get_z_slice()

    def _get_z_slice(self):
        D, H, W = self.img.shape
        if self.phase in ['train', 'val']:
            z_slice = torch.randint(D-self.n_slice, (D,))
            z_slice[-1] = D - self.n_slice
        elif self.phase == 'pred':
            z_slice = torch.arange(D-self.n_slice+1)
        return z_slice

    def __len__(self):
        return len(self.z_slice)

    def __getitem__(self, idx):
        if self.phase in ['train', 'val']:
            img_slice_idx_range = slice(self.z_slice[idx], self.z_slice[idx] + self.n_slice)
            mask_slice_idx = self.z_slice[idx] + self.n_slice//2
            img_patch = self.img[img_slice_idx_range]
            mask = self.mask[mask_slice_idx]
            data = {'img_patch': img_patch,
                    'mask': mask
                    }

            # print(f"Extract 2d: {self.z_slice[idx]}:{self.z_slice[idx]+5}, {torch.max(mask)}")
            if torch.max(mask) > (len(self.roi_names) + 1):
                print("Augmentation make wrong labels: 2d dataset")
                exit(1)

        elif self.phase == 'pred':
            img_slice_idx_range = slice(self.z_slice[idx], self.z_slice[idx] + self.n_slice)
            target_slice = self.z_slice[idx] + self.n_slice//2
            img_patch = self.img[img_slice_idx_range]
            mask = self.mask[target_slice]
            data = {'img_patch': img_patch,
                    'mask': mask,
                    'target_slice': target_slice
                    }

        else:
            data = None
            print("Not implement test phase dataset. -> NRRDDataset.__init__")
            exit()

        return data
