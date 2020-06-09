import numpy as np


def mask_dict2mask(img_shape, mask_dict, roi_names):
    D, H, W = img_shape
    mask = np.zeros([D, H, W])
    for i, roi in enumerate(roi_names):
        if roi in mask_dict:
            # background is class 0
            mask[mask_dict[roi] > 0] = i+1
    return mask
