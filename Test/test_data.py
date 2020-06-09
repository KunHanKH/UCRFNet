import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import nrrd
import numpy as np
import os
import yaml
from Loader.Dataset3d import Dataset3d


def main():
    config = yaml.load(open('/home/kunhan/workspace/UCRFNet/Config/config_data.yml', 'r'),
                       Loader=yaml.FullLoader)
    dataset_3d = Dataset3d(config['dataset'])
    print(len(dataset_3d))
    res = dataset_3d[0]
    torch.save(res, 'res.pt')


if __name__ == '__main__':
    main()
