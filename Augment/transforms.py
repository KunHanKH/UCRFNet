import importlib

import numpy as np
import time
import math
import numbers
import torch
import cv2
from scipy.ndimage import rotate, gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from skimage.transform import warp, AffineTransform, resize
from torchvision.transforms import Compose

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(int(time.time()))


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > 0.5:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        self.random_state = random_state

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, (1, 2))
        else:
            channels = [np.rot90(m[c], k, (1, 2)) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
        Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m


class RandomAffineXYZ:
    @staticmethod
    def swap3dvolume(m, axis):
        assert m.ndim in [3, 4]
        if axis in ['xz', 'zx']:
            m = np.swapaxes(m, -1, -3)
        elif axis in ['yz', 'zy']:
            m = np.swapaxes(m, -2, -3)
        return m


class RandomAffineXYZ_skimage(RandomAffineXYZ):
    def __init__(self, random_state, raw_order=1, raw_fillcolor=-1, label_fillcolor=0,
                 scale=None, translate=None, degrees=None, shear=None, axis='xy',
                 execution_probability=0.1, **kwargs):

        if degrees is not None:
            if isinstance(degrees, numbers.Number):
                if degrees < 0:
                    raise ValueError("If degrees is a single number, it must be positive.")
                self.degrees = (-degrees * math.pi / 180, degrees * math.pi / 180)
            else:
                assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                    "degrees should be a list or tuple and it must be of length 2."
                self.degrees = degrees * math.pi / 180
        else:
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.random_state = random_state
        self.raw_fillcolor = raw_fillcolor
        self.label_fillcolor = label_fillcolor
        self.raw_order = raw_order
        self.axis= axis
        self.execution_probability = execution_probability

    @staticmethod
    def get_params(random_state, scale_ranges, degrees, translate, shears, m_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        if scale_ranges is not None:
            scale = (random_state.uniform(*scale_ranges),
                     random_state.uniform(*scale_ranges))
        else:
            scale = (1.0, 1.0)

        if translate is not None:
            max_dx = translate[0] * m_size[0]
            max_dy = translate[1] * m_size[1]
            translations = (np.round(random_state.uniform(-max_dx, max_dx)),
                            np.round(random_state.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if degrees is not None:
            rotation = random_state.uniform(*degrees)
        else:
            rotation = 0

        if shears is not None:
            shear = random_state.uniform(*shears)
        else:
            shear = 0.0

        params = {'scale': scale,
                  'shear': shear,
                  'rotation': rotation,
                  'translation': translations}

        return params

    @staticmethod
    def warp3d(mm, af, order, fillcolor):
        if mm.ndim == 2:
            if np.any(mm):
                return warp(mm, af.inverse, order=order, mode='constant', cval=fillcolor)
            else:
                return mm
        else:
            return np.stack([RandomAffineXYZ_skimage.warp3d(mmm, af, order, fillcolor) for mmm in mm], axis=0)

    def __call__(self, data):
        if self.random_state.uniform() < self.execution_probability:
            assert data['raw'].ndim in [2, 3]
            if 'label' in data:
                assert data['label'].ndim in [3, 4]

            data_size = data.shape[-2:]
            params = self.get_params(self.random_state, self.scale, self.degrees, self.translate, self.shear, data_size)
            af = AffineTransform(scale=params['scale'],
                                 rotation=params['rotation'],
                                 shear=params['shear'],
                                 translation=params['translation'])

            data['raw'] = self.warp3d(data['raw'], af, self.raw_order, self.raw_fillcolor)

            if 'label' in data:
                if data['label'].ndim == 3:
                    data['label'] = self.warp3d(data['label'], af, 0, self.label_fillcolor)
                else:
                    data['label'] = self.swap3dvolume(data['label'], self.axis)

                    new_m = []
                    for mm_z in data['label']:
                        new_m.append(self.warp3d(mm_z, af, 0, self.label_fillcolor))
                    m = np.stack(new_m, axis=0)
                    data['label'] = self.swap3dvolume(m, self.axis)
                    del m, new_m

        return data


class RandomAffineXYZ_cv(RandomAffineXYZ):
    def __init__(self, random_state, affine_alpha=10,
                 raw_borderMode='BORDER_CONSTANT', label_borderMode='BORDER_TRANSPARENT',
                 raw_fillcolor=-1, label_fillcolor=0, axis='xy',
                 execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.affine_alpha = affine_alpha
        self.raw_borderMode = getattr(cv2, raw_borderMode)
        self.label_borderMode = getattr(cv2, label_borderMode)
        self.raw_fillcolor = raw_fillcolor
        self.label_fillcolor = label_fillcolor
        self.axis = axis

        self.execution_probability = execution_probability

    @staticmethod
    def warp3d(mm, M, size, borderMode, borderValue):
        if mm.ndim == 2:
            if np.any(mm):
                return cv2.warpAffine(mm, M, size[::-1], borderMode=borderMode, borderValue=borderValue)
            else:
                return mm
        else:
            return np.stack([RandomAffineXYZ_cv.warp3d(mmm, M, size, borderMode, borderValue) for mmm in mm], axis=0)

    def __call__(self, data):
        if self.random_state.uniform() < self.execution_probability:
            assert data['raw'].ndim in [2, 3]
            assert 'raw' in data
            if 'label' in data:
                assert data['label'].ndim in [3, 4]

            # z, y, x
            data_shape = data['raw'].shape[-2:]

            # Random affine
            center_square = np.float32(data_shape) // 2
            square_size = min(data_shape) // 3
            pts1 = np.float32(
                [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                 center_square - square_size])
            pts2 = pts1 + self.random_state.uniform(-self.affine_alpha, self.affine_alpha, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)

            data['raw'] = self.warp3d(data['raw'], M, data_shape, self.raw_borderMode, self.raw_fillcolor)

            if 'label' in data:
                if data['label'].ndim == 3:
                    data['label'] = self.warp3d(data['label'], M, data_shape,
                                                self.label_borderMode, self.label_fillcolor)
                else:
                    data['label'] = self.swap3dvolume(data['label'], self.axis)

                    new_m = []
                    for mm_z in data['label']:
                        new_m.append(self.warp3d(mm_z, M, data_shape,
                                                 self.label_borderMode, self.label_fillcolor))
                    m = np.stack(new_m, axis=0)
                    data['label'] = self.swap3dvolume(m, self.axis)
                    del m, new_m

                if np.amax(data['label']) >= data['label'].shape[-3]:
                    print(self.label_borderMode, self.label_fillcolor)
                    print("Rotation fill label wrongly. ", np.amax(data['label']))
                    exit(1)

        return data


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=3 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, raw_spline_order=1, label_spline_order=0,
                 alpha=15, sigma=3, raw_fillcolor=-1, label_fillcolor=0,
                 execution_probability=0.1, **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.raw_spline_order = raw_spline_order
        self.label_spline_order = label_spline_order
        self.grid_scale = 4
        self.alpha = alpha // self.grid_scale
        self.sigma = sigma
        self.raw_fillcolor = raw_fillcolor
        self.label_fillcolor = label_fillcolor
        self.execution_probability = execution_probability

    @staticmethod
    def _map(m, indices, spline_order, fillcolor):
        if m.ndim == 3:
            if np.any(m):
                return map_coordinates(m, indices, order=spline_order, mode='constant', cval=fillcolor)
            else:
                return m
        else:
            channels = [map_coordinates(c, indices, order=spline_order, mode='constant', cval=fillcolor) for c in m]
            return np.stack(channels, axis=0)

    def __call__(self, data):
        if self.random_state.uniform() < self.execution_probability:
            assert data['raw'].ndim == 3
            if 'label' in data:
                assert data['label'].ndim in [3, 4]

            volume_shape = data['raw'].shape[-3:]

            grid_shape = (volume_shape[0] // self.grid_scale,
                          volume_shape[1] // self.grid_scale,
                          volume_shape[2] // self.grid_scale)

            dz = gaussian_filter(self.random_state.rand(*grid_shape) * 2 - 1, self.sigma) * self.alpha * 0.4
            dy = gaussian_filter(self.random_state.rand(*grid_shape) * 2 - 1, self.sigma) * self.alpha
            dx = gaussian_filter(self.random_state.rand(*grid_shape) * 2 - 1, self.sigma) * self.alpha

            if self.grid_scale > 1:
                dx = resize(dx, volume_shape)
                dz = resize(dz, volume_shape)
                dy = resize(dy, volume_shape)

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            data['raw'][3:-3] = self._map(data['raw'], indices, self.raw_spline_order, self.raw_fillcolor)[3:-3]
            if 'label' in data:
                if data['label'].ndim == 4:
                    data['label'][:, 3:-3, ...] = self._map(data['label'], indices, self.label_spline_order, self.label_fillcolor)[:, 3:-3, ...]
                else:
                    data['label'][3:-3] = self._map(data['label'], indices, self.label_spline_order, self.label_fillcolor)[3:-3]
                if np.amax(data['label']) >= z_dim:
                    print("Elastic fill label wrongly. ", np.amax(data['label']))
                    exit(1)
        return data


def blur_boundary(boundary, sigma):
    boundary = gaussian(boundary, sigma=sigma)
    boundary[boundary >= 0.5] = 1
    boundary[boundary < 0.5] = 0
    return boundary


class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)  # Z
    ]

    def __init__(self, ignore_index=None, aggregate_affinities=False, append_label=False, **kwargs):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        :param aggregate_affinities: aggregate affinities with the same offset across Z,Y,X axes
        :param append_label: if True append the orignal ground truth labels to the last channel
        :param blur: Gaussian blur the boundaries
        :param sigma: standard deviation for Gaussian kernel
        """
        self.ignore_index = ignore_index
        self.aggregate_affinities = aggregate_affinities
        self.append_label = append_label

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()
        boundary_arr = [np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels]
        channels = np.stack(boundary_arr)
        results = []
        if self.aggregate_affinities:
            assert len(kernels) % 3 == 0, "Number of kernels must be divided by 3 (one kernel per offset per Z,Y,X axes"
            # aggregate affinities with the same offset
            for i in range(0, len(kernels), 3):
                # merge across X,Y,Z axes (logical OR)
                xyz_aggregated_affinities = np.logical_or.reduce(channels[i:i + 3, ...]).astype(np.int)
                # recover ignore index
                xyz_aggregated_affinities = _recover_ignore_index(xyz_aggregated_affinities, m, self.ignore_index)
                results.append(xyz_aggregated_affinities)
        else:
            results = [_recover_ignore_index(channels[i], m, self.ignore_index) for i in range(channels.shape[0])]

        if self.append_label:
            # append original input data
            results.append(m)

        # stack across channel dim
        return np.stack(results, axis=0)

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def get_kernels(self):
        raise NotImplementedError


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, blur=False, sigma=1, mode='thick', blobs=False, **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.blur = blur
        self.sigma = sigma
        self.mode = mode
        self.blobs = blobs

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        if self.blur:
            boundaries = blur_boundary(boundaries, self.sigma)

        results = []
        if self.blobs:
            blobs = (m > 0).astype('uint8')
            results.append(_recover_ignore_index(blobs, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-6, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / np.clip(self.std, a_min=self.eps, a_max=None)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value=-1000, max_value=3076, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, data):
        m = data['raw']
        data['raw'] = np.clip((m - self.min_value), a_min=0, a_max=None) / max(1, self.value_range)
        return data


class AdditiveGaussianNoise:
    def __init__(self, random_state, scale=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise
        return m


class AdditivePoissonNoise:
    def __init__(self, random_state, lam=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, raw_expand_dims=True, label_expand_dims=True,
                 raw_dtype=np.float32, label_dtype=bool, **kwargs):
        self.raw_expand_dims = raw_expand_dims
        self.label_expand_dims = label_expand_dims
        self.raw_dtype = raw_dtype
        self.label_dtype = label_dtype

    def __call__(self, data):
        assert data['raw'].ndim in [3, 4]    #'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        if 'label' in data:
            assert data['label'].ndim in [3, 4]

        # add channel dimension
        if self.raw_expand_dims and data['raw'].ndim == 3:
            data['raw'] = np.expand_dims(data['raw'], axis=0)
        data['raw'] = torch.from_numpy(data['raw'].astype(dtype=self.raw_dtype))

        if 'label' in data:
            if self.label_expand_dims and data['label'].ndim == 3:
                data['label'] = np.expand_dims(data['label'], axis=0)
            data['label'] = torch.from_numpy(data['label'].astype(dtype=self.label_dtype))

        return data


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m


def get_transformer(config, phase):
    if phase == 'val':
        phase = 'test'

    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    return Transformer(phase_config)


class Transformer:
    def __init__(self, phase_config):
        self.phase_config = phase_config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def play_transform(self):
        return self._create_transform()

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('Augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self):
        return Compose([
            self._create_augmentation(c, self.seed + i*100) for i, c in enumerate(self.phase_config)])

    def _create_augmentation(self, c, seed):
        config = c
        config['random_state'] = np.random.RandomState(seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
