from torch.utils.data import DataLoader
from Loader.Dataset2d import Dataset2d


def create_loader_2d(data, data_config, phase):
    """

    :param phase:
    :param data: {'raw': 3d_img,    # (Z, H, W)
                  'label': 3d_mask  @ (Z, H, W)
                  }
    :param data_config -> dataset, loader
    :return:
    """
    dataset_config = data_config['dataset']
    loader_config = data_config['loader']['2d_loader']

    dataset2d = Dataset2d(data, dataset_config, phase)
    if phase == 'train':
        loader = DataLoader(dataset2d, batch_size=loader_config['train_batch_size'], shuffle=True,
                            num_workers=loader_config['num_workers'])
    elif phase == 'val':
        loader = DataLoader(dataset2d, batch_size=loader_config['val_batch_size'], shuffle=True,
                            num_workers=loader_config['num_workers'])
    elif phase == 'pred':
        loader = DataLoader(dataset2d, batch_size=loader_config['pred_batch_size'], shuffle=False,
                            num_workers=loader_config['num_workers'])

    else:
        loader = None
        print("Not implement test phase dataset. -> create_loader_2d")
        exit()

    return loader
