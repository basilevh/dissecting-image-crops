'''
Image crop detection by absolute patch localization.
Test loop and performance evaluation logic.
Basile Van Hoorick, Fall 2020.
'''

# Library imports.
import argparse
import copy
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import platform
import random
import scipy
import seaborn as sns
import shutil
import sys
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

# Repository imports.
import aberrations
import data
import logistics
import model


# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')

_DEFAULT_BACKBONE_GLOBAL = 'resnet34'
_DEFAULT_BACKBONE_PATCH = 'resnet18'
_DEFAULT_GLOBAL_DIM = 224
_DEFAULT_PATCH_DIM = 96
_DEFAULT_ASPECT_RATIO = 1.5
_DEFAULT_SIZE_FACTOR_RANGE = [0.5, 0.9]
_DEFAULT_GRID_SIZE = [4, 4]
_DEFAULT_LAMBDA_PATCH = 0.15
_DEFAULT_LAMBDA_RECT = 3.0

_DEFAULT_TEST_DIR = r'data/test/'
_DEFAULT_RESULT_DIR = r'results/'
_DEFAULT_BATCH_SIZE = 8


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Process arguments.
parser = argparse.ArgumentParser()

parser.add_argument('--custom_tag', default='', type=str,
                    help='Name of current testing run to prepend to tag.')

# Paths.
parser.add_argument('--test_dir', default=_DEFAULT_TEST_DIR,
                    type=str, help='Path to test data directory.')
parser.add_argument('--result_dir', default=_DEFAULT_RESULT_DIR,
                    type=str, help='Path to directory where test results should be stored.')
parser.add_argument('--model_path', default='', type=str,
                    help='Path to existing PyTorch model file, or parent directory, to test.')

# Execution options.
parser.add_argument('--batch_size', default=_DEFAULT_BATCH_SIZE, type=int,
                    help='Mini-batch size (default: 8)')
parser.add_argument('--gpus', nargs='+', default=[],
                    type=int, help='GPU IDs to use (default: all)')
parser.add_argument('--store_detailed', default=False, type=str2bool,
                    help='If True, store all partial input batches, model outputs, and loss information.')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of PyTorch data loader multiprocessing workers (default: 0)')

# Architectural & data-related options.
parser.add_argument('--backbone_global', default=_DEFAULT_BACKBONE_GLOBAL, type=str,
                    help='Backbone model for global images (resnet18 / resnet34 / none). '
                    'If none, do not feed in the downscaled global image. (default: resnet18)')
parser.add_argument('--backbone_patch', default=_DEFAULT_BACKBONE_PATCH, type=str,
                    help='Backbone model for small patches (resnet18 / resnet34 / none). '
                    'If none, do not extract and feed in any image patch. (default: resnet18)')
parser.add_argument('--global_dim', default=_DEFAULT_GLOBAL_DIM, type=int,
                    help='Patch image size (height and width) (default: 224).')
parser.add_argument('--patch_dim', default=_DEFAULT_PATCH_DIM, type=int,
                    help='Patch image size (height and width) (default: 96).')
parser.add_argument('--aspect_ratio', default=_DEFAULT_ASPECT_RATIO, type=float,
                    help='Typical ratio of width to height (default: 1.5).')
parser.add_argument('--grid_size', nargs='+', default=_DEFAULT_GRID_SIZE, type=int,
                    help='Uniformly sized grid cell counts for patch location classification '
                    '(width x height) (default: 4 x 4).')
parser.add_argument('--add_coordinates_global', default=True, type=str2bool,
                    help='If True, add fractional (x, y) coordinates to input global images to '
                    'encourage semantic features to be linked to their position.')

# Loss options.
parser.add_argument('--lambda_patch', default=_DEFAULT_LAMBDA_PATCH, type=float,
                    help='Weight for patch localization loss term (default: 0.15).')
parser.add_argument('--lambda_rect', default=_DEFAULT_LAMBDA_RECT, type=float,
                    help='Weight for crop rectangle regression loss term (default: 3.0).')
parser.add_argument('--mix_patch_batch', default=True, type=str2bool,
                    help='If True, mix patch location labels within a batch by using cyclic shifts '
                    'to counter the peculiar batch norm / memory effect during train mode.')

# Options that can be toggled independently (to some extent) of training configuration.
parser.add_argument('--random_patch_positions', default=True, type=str2bool,
                    help='If True, select patches from uniformly random locations within sub-grid cells instead of picking the exact center.')
parser.add_argument('--disable_patch_jitter', default=False, type=str2bool,
                    help='If True, do not randomly jitter extracted patch positions by +/-8 pixels.')
parser.add_argument('--size_factor_range', nargs='+', default=_DEFAULT_SIZE_FACTOR_RANGE, type=int,
                    help='Image crop factor range (default: 0.5 to 0.9).')
parser.add_argument('--patch_localization_only', default=False, type=str2bool,
                    help='If True, test absolute patch localization network only, '
                    'ignoring globals and the rectangle and classification heads.')
parser.add_argument('--post_crop_random_resize', default=True, type=str2bool,
                    help='If True, resize all images to random but sufficiently large dimensions '
                    'just BEFORE extracting patches as well, to simulate lack of size knowledge '
                    'at test time.')

# Image perturbation options.
parser.add_argument('--red_chroma_aber', default=0.0, type=float,
                    help='Percentage of outward red transversal chromatic aberration to simulate (default: 0.0).')
parser.add_argument('--green_chroma_aber', default=0.0, type=float,
                    help='Percentage of outward green transversal chromatic aberration to simulate (default: 0.0).')
parser.add_argument('--blue_chroma_aber', default=0.0, type=float,
                    help='Percentage of outward blue transversal chromatic aberration to simulate (default: 0.0).')
parser.add_argument('--vignetting_strength', default=0.0, type=float,
                    help='Fraction of lens vignetting to simulate (1 = maximal) (default: 0.0).')
parser.add_argument('--grayscale', default=False, type=str2bool,
                    help='If True, ignore saturation and convert all images to grayscale by copying the green channel.')
parser.add_argument('--color_saturation', default=1.0, type=float,
                    help='Color saturation factor (> 1 is exaggerated, < 1 is reduced) (default: 1.0).')
parser.add_argument('--radial_distortion', default=0.0, type=float,
                    help='Radial lens distortion k coefficient to simulate (barrel or pincushion) (default: 0.0).')

# Novel options.
parser.add_argument('--load_dir_fraction', default=1.0, type=float,
                    help='If < 1.0, load a subset of the dataset only for faster evaluation.')
parser.add_argument('--ignore_if_exist', default=False, type=str2bool,
                    help='If True, do not repeat a test whose results already exist.')


def _get_test_tag(args, epoch):
    test_tag = f'test_epoch{epoch}'
    if len(args.custom_tag) != 0:
        test_tag += '_' + args.custom_tag
    if args.random_patch_positions:
        test_tag += '_rpp'
    if args.disable_patch_jitter:
        test_tag += '_dpj'
    if args.size_factor_range != _DEFAULT_SIZE_FACTOR_RANGE:
        test_tag += f'_sf{args.size_factor_range[0]}-{args.size_factor_range[1]}'
    if args.patch_localization_only:
        test_tag += '_plo'
    if args.post_crop_random_resize:
        test_tag += '_pcrr'
    if args.load_dir_fraction < 1.0:
        test_tag += f'_ldf{args.load_dir_fraction:.3f}'
    if args.red_chroma_aber != 0.0:
        test_tag += f'_rca{args.red_chroma_aber:.3f}'
    if args.green_chroma_aber != 0.0:
        test_tag += f'_gca{args.green_chroma_aber:.3f}'
    if args.blue_chroma_aber != 0.0:
        test_tag += f'_bca{args.blue_chroma_aber:.3f}'
    if args.vignetting_strength != 0.0:
        test_tag += f'_vs{args.vignetting_strength:.3f}'
    if args.grayscale:
        test_tag += '_gray'
    if args.color_saturation != 1.0:
        test_tag += f'_cs{args.color_saturation:.3f}'
    if args.radial_distortion != 0.0:
        test_tag += f'_rd{args.radial_distortion:.3f}'
    return test_tag


def _verify_flags(args, model_tag, test_tag):
    if _find_property(model_tag, 'bglo', _DEFAULT_BACKBONE_GLOBAL)[-2:] != args.backbone_global[-2:]:
        raise ValueError(
            'backbone_global argument does not match with parsed model path, this is almost certainly a mistake!')
    if _find_property(model_tag, 'bpat', _DEFAULT_BACKBONE_PATCH)[-2:] != args.backbone_patch[-2:]:
        raise ValueError(
            'backbone_patch argument does not match with parsed model path, this is almost certainly a mistake!')
    if int(_find_property(model_tag, 'pd', _DEFAULT_PATCH_DIM)) != args.patch_dim:
        raise ValueError(
            'patch_dim argument does not match with parsed model path, this is almost certainly a mistake!')
    if int(_find_property(model_tag, 'gd', _DEFAULT_GLOBAL_DIM)) != args.global_dim:
        raise ValueError(
            'global_dim argument does not match with parsed model path, this is almost certainly a mistake!')
    if float(_find_property(model_tag, 'ar', _DEFAULT_ASPECT_RATIO)) != args.aspect_ratio:
        raise ValueError(
            'aspect_ratio argument does not match with parsed model path, this is almost certainly a mistake!')
    default_grid_size_str = f'{_DEFAULT_GRID_SIZE[0]}-{_DEFAULT_GRID_SIZE[1]}'
    grid_size_str = _find_property(model_tag, 'gs', default_grid_size_str).split('-')
    if abs(float(grid_size_str[0]) - args.grid_size[0]) > 1e-4 or \
            abs(float(grid_size_str[1]) - args.grid_size[1]) > 1e-4:
        raise ValueError(
            'grid_size argument does not match with parsed model path, this is almost certainly a mistake!')
    if ('_acg' in model_tag) != args.add_coordinates_global:
        raise ValueError(
            'add_coordinates_global argument does not match with parsed model path, this is almost certainly a mistake!')
    if abs(float(_find_property(model_tag, 'lp', _DEFAULT_LAMBDA_PATCH)) - args.lambda_patch) > 1e-4:
        raise ValueError(
            'lamba_patch argument does not match with parsed model path, this is almost certainly a mistake!')
    if abs(float(_find_property(model_tag, 'lr', _DEFAULT_LAMBDA_RECT)) - args.lambda_rect) > 1e-4:
        raise ValueError(
            'lambda_rect argument does not match with parsed model path, this is almost certainly a mistake!')
    if ('_plo' in model_tag) and not('_plo' in test_tag):
        raise ValueError(
            'patch_localization_only was enabled during training, which means it must also be enabled during testing! '
            '(the reverse is allowed however)')
    if ('_mpb' in model_tag) != args.mix_patch_batch:
        raise ValueError(
            'mix_patch_batch argument does not match with parsed model path, this is almost certainly a mistake!')


def _find_property(tag, name, default):
    '''
    Finds a property value in the full given tag, or default if not found.
    Example: tag='..._bs32_...' and name='bs' will return '32'.
    '''
    if '_' + name not in tag:
        return default

    start = tag.index('_' + name)
    sub_tag = tag[start+len(name)+1:]

    if '_' in sub_tag:
        end = sub_tag.index('_')
        value = sub_tag[:end]
    else:
        value = sub_tag

    return value


def _copy_partial_batch(mbatch):
    '''
    Omits information that can be inferred from other parts.
    Excludes: global, patches.
    '''
    result = dict()
    result['orig_res'] = copy.deepcopy(mbatch['orig_res'])
    result['cur_res'] = copy.deepcopy(mbatch['cur_res'])
    result['path'] = copy.deepcopy(mbatch['path'])
    result['angle'] = copy.deepcopy(mbatch['angle'])
    result['size_factor'] = copy.deepcopy(mbatch['size_factor'])
    result['cropped'] = copy.deepcopy(mbatch['cropped'])
    result['labels'] = copy.deepcopy(mbatch['labels'])
    result['x1'] = copy.deepcopy(mbatch['x1'])
    result['x2'] = copy.deepcopy(mbatch['x2'])
    result['y1'] = copy.deepcopy(mbatch['y1'])
    result['y2'] = copy.deepcopy(mbatch['y2'])
    result['rectangle'] = copy.deepcopy(mbatch['rectangle'])
    return result


def _tensor_or_collection_to_numpy(tensor_or_list):
    '''
    Recursively converts a CUDA tensor or list or tuple of CUDA tensors to a numpy array
    (no extra dimension is introduced).
    '''
    if torch.is_tensor(tensor_or_list):
        return tensor_or_list.detach().cpu().numpy()
    elif isinstance(tensor_or_list, list) or isinstance(tensor_or_list, tuple):
        if len(tensor_or_list) != 0:
            numpy_list = [_tensor_or_collection_to_numpy(x) for x in tensor_or_list]
            if numpy_list[0].shape == ():
                numpy_array = np.array(numpy_list)
            else:
                numpy_array = np.concatenate(numpy_list, axis=0)
        else:
            numpy_array = np.array([], dtype=np.float32)
        return numpy_array
    else:
        raise ValueError('Type ' + str(type(tensor_or_list)) + ' is not supported')


def _test_model(
        my_model, loader, device,
        location_losses, rectangle_loss, binary_loss,
        grid_size, location_classes, lambda_patch, lambda_rect,
        patch_localization_only, mix_patch_batch,
        keep_detailed):
    start_time = time.time()
    my_model.eval()

    # Initialize stats & data.
    # Combine all losses in list: locations, rectangle, binary classification, total.
    running_losses = [0.0] * (location_classes + 3)
    running_corrects = [0] * (location_classes + 1)  # Threshold = 0.5.
    partial_batches = []
    model_outputs = []
    processed_outputs = []

    # Start iteration.
    for mbatch in tqdm.tqdm(loader):
        if keep_detailed:
            partial_batches.append(_copy_partial_batch(mbatch))

        patches = mbatch['patches'].to(device)  # (B, 16, 3, 96, 96)
        true_locations = mbatch['labels'].type(torch.LongTensor).to(device)
        global_images = mbatch['global'].to(device)
        true_rectangles = mbatch['rectangle'].to(device)
        true_scores = mbatch['cropped'].to(device)

        # Minimize mutual information among labels within batches.
        # NOTE: This is handled by model.py now to ensure correct embedding ordering.
        # if mix_patch_batch:
        #     batch_size = patches.shape[0]
        #     patches = data.roll_per_batch_index(patches, range(batch_size))
        #     true_locations = data.roll_per_batch_index(true_locations, range(batch_size))

        # NOTE: predicted_rectangles_scores is None if patch_localization_only is enabled.
        if keep_detailed:
            predicted_locations_list, predicted_rectangles_scores, embeddings = \
                my_model(patches, global_images, return_embeddings=True)
            
            output_detail = dict()
            output_detail['predicted_locations_list'] = \
                _tensor_or_collection_to_numpy(predicted_locations_list)
            if not patch_localization_only:
                output_detail['predicted_rectangles_scores'] = \
                    _tensor_or_collection_to_numpy(predicted_rectangles_scores)
            output_detail['embedding_tensor'] = _tensor_or_collection_to_numpy(embeddings[0])
            if not patch_localization_only:
                output_detail['rect_net_penult'] = _tensor_or_collection_to_numpy(embeddings[1])
            model_outputs.append(output_detail)
        
        else:
            predicted_locations_list, predicted_rectangles_scores = \
                my_model(patches, global_images, return_embeddings=False)

        # Get absolute patch position losses.
        current_loss = 0.0
        patch_loss_values = []
        patch_correct = []
        for i in range(location_classes):
            patch_loss_value = location_losses[i](
                predicted_locations_list[i], true_locations[:, i])
            patch_loss_values.append(patch_loss_value)
            current_loss = current_loss + patch_loss_value * lambda_patch
            predicted_location_classes = torch.argmax(
                predicted_locations_list[i], dim=1)
            patch_correct.append(torch.mean((
                predicted_location_classes == true_locations[:, i]).type(torch.float32)).item())

        # Get global crop rectangle and classification losses.
        if not patch_localization_only:
            predicted_rectangles = predicted_rectangles_scores[:, :4]
            predicted_scores = predicted_rectangles_scores[:, 4]
            rectangle_loss_value = rectangle_loss(
                predicted_rectangles, true_rectangles)
            binary_loss_value = binary_loss(predicted_scores, true_scores)
            current_loss = current_loss + rectangle_loss_value * lambda_rect
            current_loss = current_loss + binary_loss_value

        if keep_detailed:
            if not patch_localization_only:
                processed_outputs.append(
                    (_tensor_or_collection_to_numpy(patch_loss_values), np.array(patch_correct),
                     rectangle_loss_value.item(), binary_loss_value.item()))
            else:
                processed_outputs.append(
                    (_tensor_or_collection_to_numpy(patch_loss_values), np.array(patch_correct)))

        # Update stats.
        for i in range(location_classes):
            running_losses[i] += patch_loss_values[i].item()
            running_corrects[i] += patch_correct[i]
        if not patch_localization_only:
            running_losses[location_classes] += rectangle_loss_value.item()
            running_losses[location_classes + 1] += binary_loss_value.item()
            running_losses[location_classes + 2] += current_loss.item()
            binary_predictions = (predicted_scores >= 0.0)
            running_corrects[location_classes] += torch.mean(
                (binary_predictions == true_scores).type(torch.float32)).item()
        else:
            running_losses[-1] += current_loss.item()

        # Clean-up.
        del patches
        del true_locations
        del global_images
        del true_rectangles
        del true_scores
        del predicted_locations_list
        del patch_loss_values
        del patch_correct
        if not patch_localization_only:
            del predicted_rectangles_scores
            del predicted_rectangles
            del predicted_scores
            del rectangle_loss_value
            del binary_loss_value
        del current_loss
        del mbatch

    # Print & write stats.
    mean_losses = np.array(running_losses) / len(loader)
    accuracies = np.array(running_corrects) / len(loader)
    print(f'_test_model() took {int(time.time() - start_time):d}s')
    print(f'Mean loss: {mean_losses[-1]:.3f}')
    print(f'Accuracy: {accuracies[-1]:.3f}')
    print(f'Mean patch location accuracy: {accuracies[:-1].mean():.3f}')

    # Aggregate detailed data for further analysis.
    detailed_data = dict()
    if keep_detailed:
        detailed_data['partial_batches'] = partial_batches
        detailed_data['model_outputs'] = model_outputs
        detailed_data['processed_outputs'] = processed_outputs

    return (mean_losses, accuracies), detailed_data


def _store_metrics_readable(metrics, epoch, model_tag, test_tag, test_dir, save_path):
    mean_losses, accuracies = metrics
    location_classes = len(mean_losses) - 3
    lines = []

    lines.append('Model tag:')
    lines.append(model_tag)
    lines.append('')
    lines.append('Epoch (1-based):')
    lines.append(str(epoch + 1))
    lines.append('')
    lines.append('Test tag:')
    lines.append(test_tag)
    lines.append('')
    lines.append('Dataset:')
    lines.append(test_dir)
    lines.append('')

    for i in range(location_classes):
        lines.append('Patch location loss ' + str(i) + ':')
        lines.append(str(mean_losses[i]))
        lines.append('')
        lines.append('Patch location accuracy ' + str(i) + ':')
        lines.append(str(accuracies[i]))
        lines.append('')

    lines.append('Rectangle regression loss:')
    lines.append(str(mean_losses[location_classes]))
    lines.append('')
    lines.append('Crop classification loss:')
    lines.append(str(mean_losses[location_classes + 1]))
    lines.append('')
    lines.append('Total loss:')
    lines.append(str(mean_losses[location_classes + 2]))
    lines.append('')
    lines.append('Crop classification accuracy:')
    lines.append(str(accuracies[-1]))
    lines.append('')
    lines.append('Mean patch location accuracy:')
    lines.append(str(accuracies[:-1].mean()))
    lines.append('')

    with open(save_path, 'w') as f:
        f.writelines([line + '\n' for line in lines])
    print('Stored useful metrics to ' + save_path)


def _store_detailed_data(detailed_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(detailed_data, f)
    num_batches = len(detailed_data['partial_batches'])
    print(f'Stored detailed data ({num_batches} batches) to {save_path}')


def main(args):

    # Enable reproducibility.
    random.seed(1754)
    np.random.seed(1754)
    torch.manual_seed(1754)

    # Overwrite args for consistency.
    if args.patch_localization_only:
        args.backbone_global = 'none'

    # Obtain model path & tag.
    model_path = logistics.find_checkpoint_to_test(args.model_path)
    model_tag = pathlib.Path(model_path).parent.name
    epoch = logistics.get_epoch_from_path(model_path)
    test_tag = _get_test_tag(args, epoch)
    print('Model tag:', model_tag)
    print('Epoch (1-based):', epoch + 1)
    print('Test tag:', test_tag)

    # Define output directory path.
    result_dir = os.path.join(args.result_dir, model_tag)

    # Verify flags.
    _verify_flags(args, model_tag, test_tag)

    # Cancel if this test has already been done.
    info_save_path = os.path.join(result_dir, test_tag + '_info.txt')
    if args.ignore_if_exist and os.path.isfile(info_save_path):
        print('===> Test already done! Quitting...')
        sys.exit(0)

    # Define extra variables.
    size_factor_range = tuple(args.size_factor_range)
    grid_size = tuple(args.grid_size)
    use_patches = (args.backbone_patch.lower() != 'none')
    use_global = (args.backbone_global.lower() != 'none')
    location_classes = np.product(grid_size) if use_patches else 0
    patch_dim = args.patch_dim
    global_size = (args.global_dim, int(args.global_dim / args.aspect_ratio))
    random_patch_positions = args.random_patch_positions
    patch_jitter = not(args.disable_patch_jitter)
    add_coordinates_global = args.add_coordinates_global
    post_crop_random_resize = args.post_crop_random_resize
    load_dir_fraction = args.load_dir_fraction
    lambda_patch = args.lambda_patch
    lambda_rect = args.lambda_rect
    patch_localization_only = args.patch_localization_only
    mix_patch_batch = args.mix_patch_batch
    aberration_config = aberrations.AberrationConfig(
        args.red_chroma_aber, args.green_chroma_aber, args.blue_chroma_aber,
        args.vignetting_strength, args.color_saturation, args.grayscale,
        args.radial_distortion)
    crop_probability = 0.0 if patch_localization_only else 0.5
    resize_max_width = 2048

    # Print flags.
    print('Batch size:', args.batch_size)
    print('Size factor range:', size_factor_range)
    print('Grid size:', grid_size)
    print('Use patches:', use_patches)
    print('Use globals:', use_global)
    print('Patch dimension:', patch_dim)
    print('Global size:', global_size)
    print('Random patch positions within grid cells:', random_patch_positions)
    print('Random patch jitter:', patch_jitter)
    print('Add coordinates to globals:', add_coordinates_global)
    print('Post-crop random resize:', post_crop_random_resize)
    print('Load dataset fraction:', load_dir_fraction)
    print('Patch loss term weight:', lambda_patch)
    print('Rectangle loss term weight:', lambda_rect)
    print('Patch localization only:', patch_localization_only)
    print('Mix patch batch:', mix_patch_batch)
    print('Crop probability:', crop_probability)
    print('Random resize max width:', resize_max_width)

    # Print image aberration flags.
    print('Red chromatic aberration:', args.red_chroma_aber, '%')
    print('Green chromatic aberration:', args.green_chroma_aber, '%')
    print('Blue chromatic aberration:', args.blue_chroma_aber, '%')
    print('Vignetting strength:', args.vignetting_strength)
    print('Color saturation:', args.color_saturation)
    print('Grayscale:', args.grayscale)
    print('Radial distortion:', args.radial_distortion)

    # Initialize dataset and data loader.
    test_dataset = data.ImageCropPatchScaleDataset(
        args.test_dir, size_factor_range, use_patches, use_global, grid_size, patch_dim,
        global_size, aberration_config, random_patch_positions, patch_jitter, crop_probability,
        False, resize_max_width,
        preresize_width='random' if post_crop_random_resize else None,
        load_dir_fraction=load_dir_fraction)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        drop_last=True)

    # Initialize model and device.
    my_model = model.JointLocationCropRectangleNet(
        backbone_patch=args.backbone_patch, backbone_global=args.backbone_global,
        location_classes=location_classes, patch_embedding_size=64, global_embedding_size=64,
        patch_localization_only=patch_localization_only,
        add_coordinates_global=add_coordinates_global, mix_patch_batch=mix_patch_batch)
    if len(args.gpus):
        # Specify first GPU ID.
        device = torch.device('cuda:' + str(args.gpus[0]))
        my_model = torch.nn.DataParallel(my_model, device_ids=args.gpus)
    else:
        device = torch.device('cuda')  # Use all GPUs.
        my_model = torch.nn.DataParallel(my_model)
    my_model = my_model.to(device)

    # Load weights; this typically enables DataParallel as well.
    my_model.load_state_dict(torch.load(model_path))
    my_model.eval()
    my_model = my_model.to(device)

    # Initialize loss functions.
    location_losses = []
    for _ in range(location_classes):
        location_losses.append(torch.nn.CrossEntropyLoss())
    rectangle_loss = torch.nn.MSELoss()
    binary_loss = torch.nn.BCEWithLogitsLoss()

    # Start test.
    with torch.no_grad():
        metrics, detailed_data = _test_model(
            my_model, test_loader, device,
            location_losses, rectangle_loss, binary_loss,
            grid_size, location_classes, lambda_patch, lambda_rect,
            patch_localization_only, mix_patch_batch,
            args.store_detailed)

    # Save results.
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    _store_metrics_readable(metrics, epoch, model_tag, test_tag, args.test_dir, info_save_path)
    if args.store_detailed:
        save_path = os.path.join(result_dir, test_tag + '_detailed_data.p')
        _store_detailed_data(detailed_data, save_path)


if __name__ == '__main__':

    args = parser.parse_args()

    main(args)
