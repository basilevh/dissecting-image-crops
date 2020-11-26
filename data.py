'''
Image crop detection by absolute patch localization.
Data loading, processing, and transformation logic.
Basile Van Hoorick, Fall 2020.
'''

# Library imports.
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import random
import scipy
import seaborn as sns
import shutil
import tempfile
import time
import torch
import torch.nn
import torch.nn.functional
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import torchvision.utils
import tqdm
import typing

# Repository imports.
import aberrations


# https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
def _string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)


def _sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])


def _pack_sequences(seqs: typing.Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets


def _unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


def _read_image_robust(img_path):
    ''' Returns an image that meets conditions along with a success flag, in order to avoid crashing. '''
    try:
        image = plt.imread(img_path).copy()
        success = True
        if (image.ndim != 3 or image.shape[2] != 3
                or np.any(np.array(image.strides) < 0)):
            # Either not RGB or has negative stride, so discard.
            success = False

    except IOError as e:
        # Probably corrupt file.
        image = None
        success = False
        print('Failed to read image file:', img_path)
        print(e)

    return image, success


def _apply_rotation(image, angle):
    if angle == 90:
        image = np.flip(image.swapaxes(0, 1), 0).copy()
    elif angle == 180:
        image = np.flip(image, [0, 1]).copy()
    elif angle == 270:
        image = np.flip(image.swapaxes(0, 1), 1).copy()
    return image


def _get_random_rotation(image, landscape_only=False):
    '''
    Rotate a numpy array by a random multiple of 90° (could be identity).
    '''
    # Select angle
    if landscape_only:
        if image.shape[0] > image.shape[1]:
            # Source is portrait
            angle = np.random.choice([90, 270])
        else:
            # Source is landscape
            angle = np.random.choice([0, 180])
    else:
        # Both landscape and portrait allowed
        angle = np.random.choice([0, 90, 180, 270])

    # Apply rotation & return
    image = _apply_rotation(image, angle)
    return image, angle


def _ensure_landscape(image):
    ''' Returns image, angle. '''
    if image.shape[0] > image.shape[1]:
        # Source is portrait => rotate left or right
        return _get_random_rotation(image, landscape_only=True)
    else:
        # Source is landscape => leave unmodified
        return image, 0


def _ensure_aspect_ratio_center_crop(image, desired_ar):
    '''
    Crops to desired aspect ratio around image center while preserving landscape / portrait.
    '''
    is_portrait = (image.shape[0] > image.shape[1])
    if is_portrait:
        current_ar = image.shape[0] / image.shape[1]
    else:
        current_ar = image.shape[1] / image.shape[0]

    if abs(current_ar - desired_ar) < 0.02:
        # Close enough (within ~1.5%).
        return image

    elif current_ar < desired_ar:
        # Make image more narrow.
        if is_portrait:
            new_w = round(image.shape[0] / desired_ar)
            delta_w = (image.shape[1] - new_w) // 2
            return image[:, delta_w:delta_w+new_w, :]
        else:
            new_h = round(image.shape[1] / desired_ar)
            delta_h = (image.shape[0] - new_h) // 2
            return image[delta_h:delta_h+new_h, :, :]

    else:
        # Make image wider.
        if is_portrait:
            new_h = round(image.shape[1] * desired_ar)
            delta_h = (image.shape[0] - new_h) // 2
            return image[delta_h:delta_h+new_h, :, :]
        else:
            new_w = round(image.shape[0] * desired_ar)
            delta_w = (image.shape[1] - new_w) // 2
            return image[:, delta_w:delta_w+new_w, :]


def _extract_crop_edge(image, size_factor, imposed_crop_rectangle, multiple_8pxl):
    width, height = image.shape[1], image.shape[0]

    if imposed_crop_rectangle is None:
        x1 = np.random.uniform(0, 1 - size_factor)
        y1 = np.random.uniform(0, 1 - size_factor)
        stick_edge = np.random.randint(0, 4)
        if stick_edge == 0:  # left
            x1 = 0.0
        elif stick_edge == 1:  # right
            x1 = 1.0 - size_factor
        elif stick_edge == 2:  # top
            y1 = 0.0
        else:  # bottom
            y1 = 1.0 - size_factor
        x2 = x1 + size_factor
        y2 = y1 + size_factor

    else:
        x1, x2, y1, y2 = imposed_crop_rectangle

    x1_pxl = max(0, int(x1 * width))
    y1_pxl = max(0, int(y1 * height))
    x2_pxl = min(width, int(x2 * width))
    y2_pxl = min(height, int(y2 * height))

    if multiple_8pxl:
        x1_pxl = max(0, int(round(x1_pxl / 8) * 8))
        y1_pxl = max(0, int(round(y1_pxl / 8) * 8))
        x2_pxl = min(width, int(round(x2_pxl / 8) * 8))
        y2_pxl = min(height, int(round(y2_pxl / 8) * 8))
        x1 = x1_pxl / width
        x2 = x2_pxl / width
        y1 = y1_pxl / height
        y2 = y2_pxl / height

    return image[y1_pxl:y2_pxl, x1_pxl:x2_pxl], (x1, x2, y1, y2), (x1_pxl, x2_pxl, y1_pxl, y2_pxl)


def _extract_random_crop_edge(image, min_factor, max_factor, imposed_crop_rectangle, multiple_8pxl):
    '''
    Returns a random crop that sticks to an edge or corner.
    The aspect ratio is preserved in order to make for a believable photograph.
    '''
    size_factor = np.random.uniform(min_factor, max_factor)
    crop, bounds, bounds_pxl = _extract_crop_edge(
        image, size_factor, imposed_crop_rectangle, multiple_8pxl)
    return crop, bounds, bounds_pxl, size_factor


def _get_patch_xy_clear(current_cell, full_size, crop_rectangle, grid_size,
                        patch_dim, random_patch_positions, patch_jitter):

    # Describes the potentially cropped (= full) image, not the original uncropped one.
    # Since global may have been resized already, infer original size from rectangle instead.
    x, y = current_cell
    width, height = full_size
    crop_x1, crop_x2, crop_y1, crop_y2 = crop_rectangle
    orig_width = int(width / (crop_x2 - crop_x1))
    orig_height = int(height / (crop_y2 - crop_y1))
    grid_size_x, grid_size_y = grid_size
    not_cropped = (orig_width == width and orig_height == height)

    if not random_patch_positions:

        # Select center of full image grid cells, with some optional jitter.

        # Get horizontal coordinates.
        image_center_x = (x + 0.5) / grid_size_x
        orig_center_x = image_center_x * (crop_x2 - crop_x1) + crop_x1
        index_x = np.floor(orig_center_x * grid_size_x)
        x1_pxl = int((x + 0.5) / grid_size_x *
                     width - patch_dim / 2.0)
        if patch_jitter:
            # Make every offset modulo 8 equally likely => [-8, 7].
            x1_pxl = x1_pxl + np.random.randint(-8, 8)
            x1_pxl = np.clip(x1_pxl, 0, width - patch_dim)
        x2_pxl = x1_pxl + patch_dim

        # Get vertical coordinates.
        image_center_y = (y + 0.5) / grid_size_y
        orig_center_y = image_center_y * (crop_y2 - crop_y1) + crop_y1
        index_y = np.floor(orig_center_y * grid_size_y)
        y1_pxl = int(image_center_y * height - patch_dim / 2.0)
        if patch_jitter:
            # Make every offset modulo 8 equally likely => [-8, 7].
            y1_pxl = y1_pxl + np.random.randint(-8, 8)
            y1_pxl = np.clip(y1_pxl, 0, height - patch_dim)
        y2_pxl = y1_pxl + patch_dim

    else:

        # Select uniformly random patch within full image grid cells.
        # However, keep trying to avoid grid border crossings of the *uncropped* image.
        # At least 75% of the patch area must belong to one source cell.
        # This ensures non-ambiguity in the predictions. Jitter is irrelevant here.

        # Get horizontal coordinates.
        uncropped_crossing = True
        for _ in range(32):  # Try a limited number of times.
            low = np.ceil(x / grid_size_x * width)
            high = np.floor((x + 1) / grid_size_x * width - patch_dim)
            x1_pxl = np.random.randint(low, high)  # Pixel offset within full.
            x2_pxl = x1_pxl + patch_dim
            # Get fractional positions within original image (at 25% and 75%).
            x1_orig = (crop_x1 * orig_width + x1_pxl + patch_dim / 4) / orig_width
            x2_orig = (crop_x1 * orig_width + x2_pxl - patch_dim / 4) / orig_width
            check_index_x1 = np.floor(x1_orig * grid_size_x)  # Class within original.
            check_index_x2 = np.floor(x2_orig * grid_size_x)
            if check_index_x1 == check_index_x2:
                # Both borders almost belong to the same class.
                uncropped_crossing = False
                break
        if uncropped_crossing:
            print('WARNING: failed to find a horizontal patch pixel range that covers just a single cell!')

        # Get vertical coordinates.
        uncropped_crossing = True
        for _ in range(32):  # Try a limited number of times.
            low = np.ceil(y / grid_size_y * height)
            high = np.floor((y + 1) / grid_size_y * height - patch_dim)
            y1_pxl = np.random.randint(low, high)  # Pixel offset within full.
            y2_pxl = y1_pxl + patch_dim
            # Get fractional positions within original image (at 25% and 75%).
            y1_orig = (crop_y1 * orig_height + y1_pxl + patch_dim / 4) / orig_height
            y2_orig = (crop_y1 * orig_height + y2_pxl - patch_dim / 4) / orig_height
            check_index_y1 = np.floor(y1_orig * grid_size_y)  # Class within original.
            check_index_y2 = np.floor(y2_orig * grid_size_y)
            if check_index_y1 == check_index_y2:
                # Both borders almost belong to the same class.
                uncropped_crossing = False
                break
        if uncropped_crossing:
            print('WARNING: failed to find a vertical patch pixel range that covers just a single cell!')

        # Look at cell index of the actual center.
        x_center_orig = (crop_x1 * orig_width + (x1_pxl + x2_pxl) / 2.0) / orig_width
        y_center_orig = (crop_y1 * orig_height + (y1_pxl + y2_pxl) / 2.0) / orig_height
        index_x = np.floor(x_center_orig * grid_size_x)
        index_y = np.floor(y_center_orig * grid_size_y)

        # Sanity check.
        if not_cropped:
            assert(index_x == x)
            assert(index_y == y)

    return x1_pxl, x2_pxl, y1_pxl, y2_pxl, index_x, index_y


def _resize_random_interpol(image, width, height):
    ip_index = np.random.choice([0, 1, 2, 3, 4])
    if ip_index == 0:
        interpol = cv2.INTER_NEAREST
    elif ip_index == 1:
        interpol = cv2.INTER_LINEAR
    elif ip_index == 2:
        interpol = cv2.INTER_AREA
    elif ip_index == 3:
        interpol = cv2.INTER_CUBIC
    elif ip_index == 4:
        interpol = cv2.INTER_LANCZOS4
    else:
        raise Exception('Unknown interpolation method index')
    # interpol = cv2.INTER_LINEAR
    image = cv2.resize(image, (width, height), interpol)
    return image


def _consecutive_random_resize(image, steps, max_width, dest_size):
    temp = image
    width, height = image.shape[1], image.shape[0]
    aspect_ratio = width / height
    for _ in range(steps - 1):
        cur_w = np.random.randint(max_width // 2, max_width)
        cur_h = int(cur_w / aspect_ratio * np.random.uniform(0.8, 1.2))
        temp = _resize_random_interpol(temp, cur_w, cur_h)
    final = _resize_random_interpol(temp, *dest_size)
    return final


def _get_rectangle_from_file_name(file_name):
    param_split = file_name.split('.jpg')[0].split('_')
    factor = float(param_split[1])
    x1_pxl = int(param_split[2])
    x2_pxl = int(param_split[3])
    y1_pxl = int(param_split[4])
    y2_pxl = int(param_split[5])
    if x2_pxl - x1_pxl < y2_pxl - y1_pxl:
        # Portrait.
        # Impossible to infer rectangle without angle => simply state cropped or not.
        return factor, None, None, None, None
    else:
        # Landscape.
        return factor, x1_pxl, x2_pxl, y1_pxl, y2_pxl


class ImageCropPatchScaleDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class that crops 50% of images and extracts both thumbnails and patches.
    '''

    def __init__(self, root_dir: str, size_factor_range: tuple, use_patches: bool, use_global: bool,
                 grid_size: tuple, patch_dim: int, global_size: tuple,
                 aberration_config: aberrations.AberrationConfig, random_patch_positions: bool,
                 patch_jitter: bool, crop_probability: float, force_random_index: bool,
                 resize_max_width: int,
                 imposed_crop_rectangle: tuple = None,
                 precrop_aspect_ratio: float = None,
                 preresize_width: int = None,
                 silent_initialization: bool = False,
                 crop_multiple_8pxl: bool = False,
                 double_count: bool = False,
                 load_dir_fraction: float = 1.0):
        '''
        Initializes the dataset.

        Args:
            root_dir: The directory path within which image files (.JPG and .PNG) are directly
                contained. Alternatively, path to a single image file for focused evaluation.
            size_factor_range: (min_factor, max_factor) defined as the bounds for crop factor
                within which a uniformly random selection is made per image. NOTE: It is recommended
                to make max_factor small enough such that at least one patch will be classified to
                an unexpected position most of the time, for example 0.85 for a 4x4 grid size.
            use_patches: Whether to return patches extracted from every (possibly cropped but never
                resampled) image.
            use_global: Whether to return the downscaled, possibly cropped image. Shortcut fuzzing
                against resampling detection is always used here.
            grid_size: (x, y) indicating number of grid cells in each dimension. The number of
                classes for the patch location classifier must equal x*y.
            patch_dim: Image patch square size.
            global_size: (width, height) indicating destination size for the possibly cropped
                image, after shortcut fuzzing.
            aberration_config: Object with customized lens aberration pipeline settings.
            random_patch_positions: If False, select the patch strictly at the center of every grid
                cell of the possibly cropped image. If True, extract uniformly randomly within the
                grid cell bounds, while ensuring that the true label (with respect to the uncropped
                grid cells) covers at least 75% of the patch area.
            patch_jitter: If True and if random_patch_positions = False, randomly jitter source
                positions of patches by [-7, 8] pixels in both dimensions to avoid potential JPEG
                block artefact alignment.
            crop_probability: Probability of cropping a raw input image; typically 0.5.
            force_random_index: Select every file index using system time-dependent random seed.
            resize_max_width: Shortcut fuzzing will happen by resizing within
                [resize_max_width // 2, resize_max_width].
            imposed_crop_rectangle: If not None, (x1, x2, y1, y2) specifies the desired crop
                rectangle parameters to test.
            precrop_aspect_ratio: If not None, crop all non-conforming incoming images according to
                this width to height ratio.
            preresize_width: if not None, resize (potentially cropped) global images before
                extracting patches or downscaling. This is to simulate that real-world images can
                have any size. If 'random', select uniformly random value in
                [resize_max_width // 2, resize_max_width].
            silent_initialization: If True, do not print informational messages.
            crop_multiple_8pxl: If True, ensure that x1 and y1 are multiples of 8 pixels to avoid
                reliance on JPEG block artefacts.
            double_count: If True, load every image once original and once cropped.
                If None, auto decide based on dataset size. If False, always use default mode.
            load_dir_fraction: If < 1.0, load a subset of the dataset only for faster evaluation.
        '''
        self._root_dir = root_dir
        self._min_size_factor = size_factor_range[0]
        self._max_size_factor = size_factor_range[1]
        self._use_patches = use_patches
        self._use_global = use_global
        self._grid_size_x = grid_size[0]
        self._grid_size_y = grid_size[1]
        self._patch_dim = patch_dim
        self._global_size = global_size
        self._aberration_config = aberration_config
        self._random_patch_positions = random_patch_positions
        self._patch_jitter = patch_jitter
        self._crop_probability = crop_probability
        self._force_random_index = force_random_index
        self._resize_max_width = resize_max_width
        self._imposed_crop_rectangle = imposed_crop_rectangle
        self._precrop_aspect_ratio = precrop_aspect_ratio
        self._preresize_width = preresize_width
        self._crop_multiple_8pxl = crop_multiple_8pxl
        self._resize_steps = 4
        # To ensure random_patch_positions never gets stuck; typical value = 768.
        self._min_image_dim = int(patch_dim * np.max(grid_size) / np.min(size_factor_range))
        if not silent_initialization:
            print('Minimum image dimension: ', self._min_image_dim)

        # Load list of input image file names.
        if os.path.isdir(root_dir):
            self._all_files = os.listdir(root_dir)
            self._all_files = [fn for fn in self._all_files
                               if fn.lower().endswith('.jpg') or fn.lower().endswith('.png')]

            # Select subset if desired.
            if load_dir_fraction < 1.0:
                print(f'Selecting fraction: {load_dir_fraction:.3f}, old file count: {len(self._all_files)}')
                self._all_files = self._all_files[:int(load_dir_fraction*len(self._all_files))]
                double_count = False

            self._all_files.sort()

        elif os.path.isfile(root_dir):
            if not silent_initialization:
                print('Single file was selected, ignoring all others.')
            self._all_files = [os.path.basename(root_dir)]
            self._root_dir = pathlib.Path(root_dir).parent

        else:
            raise ValueError(f'{root_dir} is neither a directory nor a file.')

        # Avoid memory leak (NOTE: doesn't appear to fully resolve the issue).
        # https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
        # self._all_files = np.array(self._all_files)

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
        seqs = [_string_to_sequence(s) for s in self._all_files]
        self._all_files_v, self._all_files_o = _pack_sequences(seqs)

        self._file_count = len(self._all_files)
        if not silent_initialization:
            print('Image file count:', self._file_count)

        # Address double count option.
        self._ensure_min_count = 1000
        if double_count is None:
            if self._file_count <= self._ensure_min_count:
                self._double_count = True
                print('===> This is a tiny test set, so we DOUBLE COUNT every image (once original, once cropped).')
            else:
                self._double_count = False
        else:
            self._double_count = double_count
        if self._double_count:
            if self._force_random_index:
                print('===> WARNING: force_random_index is True but this setting will be IGNORED.')
            if self._crop_probability != 0.5:
                print('===> WARNING: crop_probability is not 0.5 but this setting will be IGNORED.')
            self._repeat_all = int(self._ensure_min_count / self._file_count)
            print(f'===> The whole dataset will be iterated {self._repeat_all} times.')
        else:
            self._repeat_all = 1  # Just once.

        # Instantiate lens aberration preprocessing pipeline.
        if aberration_config is not None:
            self._aberration_pipeline = aberrations.ImageAberrationPipeline(aberration_config)
        else:
            self._aberration_pipeline = None

        # Define final transforms.
        self._patch_tf = torchvision.transforms.ToTensor()
        self._global_tf = torchvision.transforms.ToTensor()

    def __len__(self):
        return self._file_count * (2 if self._double_count else 1) * self._repeat_all

    def __getitem__(self, index):
        success = False

        if not self._double_count:
            # Default loading mode.
            # Force randomness for image retrieval if desired.
            if self._force_random_index:
                if np.random.uniform(0.0, 1.0) < 0.1:
                    random_state = index * 183 + int((time.time() * 123456789.0) % 321654987.0)
                    np.random.seed(random_state)
                file_index = np.random.choice(self._file_count)
            else:
                file_index = index

            for _ in range(32):
                # Get source image name from index.
                # img_name = self._all_files[file_index]
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
                seq = _unpack_sequence(self._all_files_v, self._all_files_o, file_index)
                img_name = _sequence_to_string(seq)

                # Get source image path, load into array, and verify dimensions.
                img_path = os.path.join(self._root_dir, img_name)
                orig_image, success = _read_image_robust(img_path)
                if success:
                    success = (orig_image.shape[0] >= self._min_image_dim and
                            orig_image.shape[1] >= self._min_image_dim)
                else:
                    pass

                if success:
                    break
                else:
                    # Retry or crash depending on dataset configuration.
                    if self._force_random_index:
                        file_index = np.random.choice(self._file_count)
                    else:
                        raise RuntimeError(f'Could not load image: {img_name} (index {index})')

            if not success:
                raise RuntimeError('Could not load a suitable input image after 32 tries.')

        elif self._double_count:
            # Double count mode.
            file_index = (index // 2) % self._file_count
            img_name = self._all_files[file_index]
            img_path = os.path.join(self._root_dir, img_name)
            orig_image, success = _read_image_robust(img_path)
            if success:
                success = (orig_image.shape[0] >= self._min_image_dim and
                        orig_image.shape[1] >= self._min_image_dim)
            if not success:
                raise RuntimeError('This file could not be loaded: ' + img_path)

        # Ensure landscape.
        orig_image, angle = _ensure_landscape(orig_image)

        # Apply some uncropped preprocessing (unless it could be already cropped).
        # Ensure aspect ratio by precropping source image if desired.
        if self._precrop_aspect_ratio is not None:
            orig_image = _ensure_aspect_ratio_center_crop(orig_image, self._precrop_aspect_ratio)
        orig_width, orig_height = orig_image.shape[1], orig_image.shape[0]

        # Apply specified perturbations onto *uncropped* image.
        if self._aberration_pipeline is not None:
            aber_image = self._aberration_pipeline.process_image(orig_image)
            assert(aber_image.shape == orig_image.shape)
        else:
            aber_image = orig_image

        # Determine whether to crop or not.
        if not self._double_count:
            # Default data loading mode => probability-based.
            if np.random.uniform(0.0, 1.0) < 1.0 - self._crop_probability:
                is_cropped = False
            else:
                is_cropped = True
        
        elif self._double_count:
            # Double count mode => deterministic.
            is_cropped = (index % 2 == 1)

        # Extract random crop, or leave intact.
        if not is_cropped:
            # Original.
            image = aber_image
            x1 = y1 = 0.0
            x2 = y2 = 1.0
            size_factor = 1.0

        else:
            # Cropped.
            image, (x1, x2, y1, y2), _, size_factor = _extract_random_crop_edge(
                aber_image, self._min_size_factor, self._max_size_factor,
                self._imposed_crop_rectangle, self._crop_multiple_8pxl)
        
        # Ensure sophisticated shortcut mitigation by resizing global first if desired.
        if self._preresize_width is not None:
            if self._preresize_width == 'random':
                current_preresize_width = int(np.random.uniform(
                    self._resize_max_width // 2, self._resize_max_width))
            else:
                current_preresize_width = self._preresize_width
            image = _resize_random_interpol(
                image, current_preresize_width,
                int(current_preresize_width * image.shape[0] / image.shape[1]))
        width, height = image.shape[1], image.shape[0]

        # Extract patches from centers of grid cells.
        # NOTE: Images adhere to cropped grid, but labels correspond to original grid.
        patches_list = []
        labels_list = []
        if self._use_patches:
            for y in range(self._grid_size_y):
                for x in range(self._grid_size_x):

                    # Obtain offsets in a smart way that avoids crossing original grid cells
                    # to ensure non-ambiguous supervision when classifying locations.
                    x1_pxl, x2_pxl, y1_pxl, y2_pxl, index_x, index_y = _get_patch_xy_clear(
                        (x, y), (width, height),
                        (x1, x2, y1, y2), (self._grid_size_x, self._grid_size_y),
                        self._patch_dim, self._random_patch_positions, self._patch_jitter)

                    # Extract patch and label.
                    patch = image[y1_pxl:y2_pxl, x1_pxl:x2_pxl]
                    patch = self._patch_tf(patch)
                    label = index_x + index_y * self._grid_size_x

                    patches_list.append(patch)
                    labels_list.append(label)

            # Avoid lists due to memory leaks.
            patches_tensor = torch.stack(patches_list)  # (16, 3, 96, 96)
            labels_tensor = torch.tensor(labels_list)  # (16)
            del patches_list, labels_list

        else:
            patches_tensor = -1
            labels_tensor = -1

        # Shortcut fuzzing for global image by consecutive random scaling.
        if self._use_global:
            max_width = self._resize_max_width
            global_image = _consecutive_random_resize(
                image, self._resize_steps, max_width, self._global_size)
            global_image = self._global_tf(global_image)

        else:
            global_image = -1.0

        del orig_image, aber_image, image

        is_cropped_value = (1.0 if is_cropped else 0.0)
        rectangle = np.array([x1, x2, y1, y2], dtype=np.float32)
        result = {'global': global_image, 'orig_res': (orig_width, orig_height),
                  'cur_res': (width, height), 'path': img_path, 'angle': angle,
                  'size_factor': size_factor, 'cropped': is_cropped_value,
                  'patches': patches_tensor, 'labels': labels_tensor,
                  'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                  'rectangle': rectangle}
        return result


def roll_per_batch_index(tensor, shifts):
    '''
    Performs cyclic shift of subtensors at every row.
    tensor: (B, cell, ...).
    Example: 6 cells, each corresponding to a batch with 3 elements.
    one column = one batch; one row = one image.
    tensor([[0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5]])
    tensor([[0, 1, 2, 3, 4, 5],
            [5, 0, 1, 2, 3, 4],
            [4, 5, 0, 1, 2, 3]])
    '''
    for i, shift in enumerate(shifts):
        tensor[i] = torch.roll(tensor[i], shift, dims=0)
    return tensor
