'''
Image crop detection by absolute patch localization.
Training + validation loop and configuration logic.
Basile Van Hoorick, Fall 2020.
'''

# Library imports.
import argparse
import copy
import cv2
import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
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

_DEFAULT_NUM_EPOCHS = 25
_DEFAULT_INITIAL_LEARN_RATE = 0.005
_DEFAULT_LEARN_RATE_DECAY = 0.942
_DEFAULT_BACKBONE_GLOBAL = 'resnet34'
_DEFAULT_BACKBONE_PATCH = 'resnet18'
_DEFAULT_GLOBAL_DIM = 224
_DEFAULT_PATCH_DIM = 96
_DEFAULT_ASPECT_RATIO = 1.5
_DEFAULT_SIZE_FACTOR_RANGE = [0.5, 0.9]
_DEFAULT_GRID_SIZE = [4, 4]
_DEFAULT_LAMBDA_PATCH = 0.15
_DEFAULT_LAMBDA_RECT = 3.0

_DEFAULT_TRAIN_DIR = r'data/train/'
_DEFAULT_VAL_DIR = r'data/val/'
_DEFAULT_CHECKPOINT_DIR = r'checkpoints/'
_DEFAULT_LOG_DIR = r'logs/'
_DEFAULT_IMAGE_DIR = r'images/'  # Unused.
_DEFAULT_BATCH_SIZE = 8


# Process arguments.
parser = argparse.ArgumentParser()

parser.add_argument('--custom_tag', default='', type=str,
                    help='Name of current training run to prepend to model tag (default: empty).')

# Paths.
parser.add_argument('--train_dir', default=_DEFAULT_TRAIN_DIR,
                    type=str, help='Path to training data directory.')
parser.add_argument('--val_dir', default=_DEFAULT_VAL_DIR,
                    type=str, help='Path to validation data directory.')
parser.add_argument('--checkpoint_dir', default=_DEFAULT_CHECKPOINT_DIR,
                    type=str, help='Path to directory where model checkpoints should be stored.')
parser.add_argument('--log_dir', default=_DEFAULT_LOG_DIR,
                    type=str, help='Path to directory where TensorBoard logs should be stored.')
parser.add_argument('--image_dir', default=_DEFAULT_IMAGE_DIR,
                    type=str, help='Path to directory where example images should be stored.')
parser.add_argument('--resume', default='', type=str,
                    help='If specified, path to existing PyTorch model file to continue training on. '
                    'If "latest", find and select most recent model and epoch automatically.')

# Training options.
parser.add_argument('--batch_size', default=_DEFAULT_BATCH_SIZE, type=int,
                    help='Mini-batch size (default: 8)')
parser.add_argument('--gpus', nargs='+', default=[],
                    type=int, help='GPU IDs to use (default: all)')
parser.add_argument('--num_epochs', default=_DEFAULT_NUM_EPOCHS, type=int,
                    help='Number of epochs to train up to (default: 50)')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of PyTorch data loader multiprocessing workers (default: 0)')
parser.add_argument('--initial_learn_rate', default=_DEFAULT_INITIAL_LEARN_RATE, type=float,
                    help='Initial learning rate (default: 0.005)')
parser.add_argument('--learn_rate_decay', default=_DEFAULT_LEARN_RATE_DECAY, type=float,
                    help='Multiplicative learning rate factor per epoch for exponential decay (default: 0.942)')

# Stubborn memory leak workaround.
parser.add_argument('--stop_after_epochs', default=-1, type=int,
                    help='If >= 1, halt the process after training for this number of epochs '
                    'to avoid the Python / PyTorch data loading memory leak (default: -1).')

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
parser.add_argument('--random_patch_positions', default=False, type=bool,
                    help='If True, select patches from uniformly random locations within sub-grid cells instead of picking the exact center.')
parser.add_argument('--disable_patch_jitter', default=False, type=bool,
                    help='If True, do not randomly jitter extracted patch positions by +/-8 pixels.')
parser.add_argument('--size_factor_range', nargs='+', default=_DEFAULT_SIZE_FACTOR_RANGE, type=int,
                    help='Image crop factor range (default: 0.5 to 0.9).')
parser.add_argument('--grid_size', nargs='+', default=_DEFAULT_GRID_SIZE, type=int,
                    help='Uniformly sized grid cell counts for patch location classification '
                    '(width x height) (default: 4 x 4).')
parser.add_argument('--add_coordinates_global', default=True, type=bool,
                    help='If True, add fractional (x, y) coordinates to input global images to '
                    'encourage semantic features to be linked to their position.')
parser.add_argument('--post_crop_random_resize', default=True, type=bool,
                    help='If True, resize all images to random but sufficiently large dimensions '
                    'just BEFORE extracting patches as well, to simulate lack of size knowledge '
                    'at test time.')

# Loss options.
parser.add_argument('--lambda_patch', default=_DEFAULT_LAMBDA_PATCH, type=float,
                    help='Weight for patch localization loss term (default: 0.15).')
parser.add_argument('--lambda_rect', default=_DEFAULT_LAMBDA_RECT, type=float,
                    help='Weight for crop rectangle regression loss term (default: 3.0).')
parser.add_argument('--patch_localization_only', default=False, type=bool,
                    help='If True, train absolute patch localization network only, '
                    'ignoring globals and the rectangle and classification heads.')
parser.add_argument('--mix_patch_batch', default=True, type=bool,
                    help='If True, mix patch location labels within a batch by using cyclic shifts '
                    'to counter the peculiar batch norm / memory effect during train mode.')


def _get_model_tag(args):
    model_prefix = datetime.datetime.now().strftime('%Y-%m-%d')
    model_tag = model_prefix
    if len(args.custom_tag) != 0:
        model_tag += '_' + args.custom_tag
    if args.batch_size != _DEFAULT_BATCH_SIZE:
        model_tag += '_bs' + str(args.batch_size)
    if args.num_epochs != _DEFAULT_NUM_EPOCHS:
        model_tag += '_ne' + str(args.num_epochs)
    if args.backbone_global != _DEFAULT_BACKBONE_GLOBAL:
        model_tag += '_bglo' + args.backbone_global[:2] + args.backbone_global[-2:]
    if args.backbone_patch != _DEFAULT_BACKBONE_PATCH:
        model_tag += '_bpat' + args.backbone_patch[:2] + args.backbone_patch[-2:]
    if args.patch_dim != _DEFAULT_PATCH_DIM:
        model_tag += '_pd' + str(args.patch_dim)
    if args.global_dim != _DEFAULT_GLOBAL_DIM:
        model_tag += '_gd' + str(args.global_dim)
    if args.aspect_ratio != _DEFAULT_ASPECT_RATIO:
        model_tag += f'_ar{args.aspect_ratio:.2f}'
    if args.random_patch_positions:
        model_tag += '_rpp'
    if args.disable_patch_jitter:
        model_tag += '_dpj'
    if args.size_factor_range != _DEFAULT_SIZE_FACTOR_RANGE:
        model_tag += f'_sf{args.size_factor_range[0]}-{args.size_factor_range[1]}'
    if args.grid_size != _DEFAULT_GRID_SIZE:
        model_tag += f'_gs{args.grid_size[0]}-{args.grid_size[1]}'
    if args.add_coordinates_global:
        model_tag += '_acg'
    if args.post_crop_random_resize:
        model_tag += '_pcrr'
    if args.lambda_patch != _DEFAULT_LAMBDA_PATCH:
        model_tag += f'_lp{args.lambda_patch:.3f}'
    if args.lambda_rect != _DEFAULT_LAMBDA_RECT:
        model_tag += f'_lr{args.lambda_rect:.3f}'
    if args.patch_localization_only:
        model_tag += '_plo'
    if args.mix_patch_batch:
        model_tag += '_mpb'
    return model_tag


def _verify_flags(args, model_tag):
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
    if ('_rpp' in model_tag) != args.random_patch_positions:
        raise ValueError(
            'random_patch_positions argument does not match with parsed model path, this is almost certainly a mistake!')
    if ('_dpj' in model_tag) != args.disable_patch_jitter:
        raise ValueError(
            'disable_patch_jitter argument does not match with parsed model path, this is almost certainly a mistake!')
    default_size_factor_str = f'{_DEFAULT_SIZE_FACTOR_RANGE[0]}-{_DEFAULT_SIZE_FACTOR_RANGE[1]}'
    size_factor_str = _find_property(model_tag, 'sf', default_size_factor_str).split('-')
    if abs(float(size_factor_str[0]) - args.size_factor_range[0]) > 1e-4 or \
            abs(float(size_factor_str[1]) - args.size_factor_range[1]) > 1e-4:
        raise ValueError(
            'size_factor_range argument does not match with parsed model path, this is almost certainly a mistake!')
    default_grid_size_str = f'{_DEFAULT_GRID_SIZE[0]}-{_DEFAULT_GRID_SIZE[1]}'
    grid_size_str = _find_property(model_tag, 'gs', default_grid_size_str).split('-')
    if abs(float(grid_size_str[0]) - args.grid_size[0]) > 1e-4 or \
            abs(float(grid_size_str[1]) - args.grid_size[1]) > 1e-4:
        raise ValueError(
            'grid_size argument does not match with parsed model path, this is almost certainly a mistake!')
    if ('_acg' in model_tag) != args.add_coordinates_global:
        raise ValueError(
            'add_coordinates_global argument does not match with parsed model path, this is almost certainly a mistake!')
    if ('_pcrr' in model_tag) != args.post_crop_random_resize:
        raise ValueError(
            'post_crop_random_resize argument does not match with parsed model path, this is almost certainly a mistake!')
    if abs(float(_find_property(model_tag, 'lp', _DEFAULT_LAMBDA_PATCH)) - args.lambda_patch) > 1e-4:
        raise ValueError(
            'lamba_patch argument does not match with parsed model path, this is almost certainly a mistake!')
    if abs(float(_find_property(model_tag, 'lr', _DEFAULT_LAMBDA_RECT)) - args.lambda_rect) > 1e-4:
        raise ValueError(
            'lambda_rect argument does not match with parsed model path, this is almost certainly a mistake!')
    if ('_plo' in model_tag) != args.patch_localization_only:
        raise ValueError(
            'patch_localization_only argument does not match with parsed model path, this is almost certainly a mistake!')
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


def _train_epoch(
        my_model, loader, device, epoch,
        location_losses, rectangle_loss, binary_loss,
        optimizer, writer, grid_size, location_classes,
        lambda_patch, lambda_rect, patch_localization_only,
        mix_patch_batch):
    start_time = time.time()
    my_model.train()

    # Initialize stats.
    # Combine all losses in list: locations, rectangle, binary classification, total.
    running_losses = [0.0] * (location_classes + 3)
    running_corrects = [0] * (location_classes + 1)  # Threshold = 0.5.

    # Start iteration.
    for mbatch in tqdm.tqdm(loader):
        patches = mbatch['patches'].to(device)  # (B, 16, 3, 96, 96) but could be -1.0.
        true_locations = mbatch['labels'].type(torch.LongTensor).to(device)  # (B, 16).
        global_images = mbatch['global'].to(device)  # Could be -1.0 if use_global is disabled.
        true_rectangles = mbatch['rectangle'].to(device)
        true_scores = mbatch['cropped'].to(device)

        # Minimize mutual information among labels within batches.
        # NOTE: This is handled by model.py now to ensure correct embedding ordering.
        # if mix_patch_batch:
        #     batch_size = patches.shape[0]
        #     patches = data.roll_per_batch_index(patches, range(batch_size))
        #     true_locations = data.roll_per_batch_index(true_locations, range(batch_size))

        # NOTE: predicted_rectangles_scores is None if patch_localization_only is enabled.
        predicted_locations_list, predicted_rectangles_scores = \
            my_model(patches, global_images)

        # Address absolute patch positions.
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

        # Address global crop rectangle and classification.
        if not patch_localization_only:
            predicted_rectangles = predicted_rectangles_scores[:, :4]
            predicted_scores = predicted_rectangles_scores[:, 4]
            rectangle_loss_value = rectangle_loss(
                predicted_rectangles, true_rectangles)
            binary_loss_value = binary_loss(predicted_scores, true_scores)
            current_loss = current_loss + rectangle_loss_value * lambda_rect
            current_loss = current_loss + binary_loss_value

        # Optimization step.
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

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
    print(f'_train_epoch() took {int(time.time() - start_time):d}s')
    print(f'Mean loss: {mean_losses[-1]:.3f}')
    print(f'Accuracy: {accuracies[-1]:.3f}')
    print(f'Mean patch location accuracy: {accuracies[:-1].mean():.3f}')

    for i in range(location_classes):
        writer.add_scalar(
            f'train_detail/location_{i}_loss', mean_losses[i], epoch)
        writer.add_scalar(
            f'train_detail/location_{i}_accuracy', accuracies[i], epoch)
    writer.add_scalar('train/rectangle_loss',
                      mean_losses[location_classes], epoch)
    writer.add_scalar('train/binary_loss',
                      mean_losses[location_classes + 1], epoch)
    writer.add_scalar('train/total_loss',
                      mean_losses[location_classes + 2], epoch)
    if location_classes != 0:
        writer.add_scalar('train/mean_location_loss',
                          mean_losses[:location_classes].mean(), epoch)
        writer.add_scalar('train/mean_location_accuracy',
                          accuracies[:location_classes].mean(), epoch)
    writer.add_scalar('train/accuracy',
                      accuracies[-1], epoch)

    return mean_losses, accuracies


def _val_epoch(
        my_model, loader, device, epoch,
        location_losses, rectangle_loss, binary_loss,
        writer, grid_size, location_classes,
        lambda_patch, lambda_rect, patch_localization_only,
        mix_patch_batch):
    start_time = time.time()
    my_model.eval()

    # Initialize stats.
    # Combine all losses in list: locations, rectangle, binary classification, total.
    running_losses = [0.0] * (location_classes + 3)
    running_corrects = [0] * (location_classes + 1)  # Threshold = 0.5.

    # Start iteration.
    for mbatch in tqdm.tqdm(loader):
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
        predicted_locations_list, predicted_rectangles_scores = \
            my_model(patches, global_images)

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
    print(f'_val_epoch() took {int(time.time() - start_time):d}s')
    print(f'Mean loss: {mean_losses[-1]:.3f}')
    print(f'Accuracy: {accuracies[-1]:.3f}')
    print(f'Mean patch location accuracy: {accuracies[:-1].mean():.3f}')

    for i in range(location_classes):
        writer.add_scalar(
            f'val_detail/location_{i}_loss', mean_losses[i], epoch)
        writer.add_scalar(
            f'val_detail/location_{i}_accuracy', accuracies[i], epoch)
    writer.add_scalar('val/rectangle_loss',
                      mean_losses[location_classes], epoch)
    writer.add_scalar('val/binary_loss',
                      mean_losses[location_classes + 1], epoch)
    writer.add_scalar('val/total_loss',
                      mean_losses[location_classes + 2], epoch)
    if location_classes != 0:
        writer.add_scalar('val/mean_location_loss',
                          mean_losses[:location_classes].mean(), epoch)
        writer.add_scalar('val/mean_location_accuracy',
                          accuracies[:location_classes].mean(), epoch)
    writer.add_scalar('val/accuracy',
                      accuracies[-1], epoch)

    return mean_losses, accuracies


def _get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _update_best_epoch_weights(dir_path, my_model, epoch, val_acc):
    '''
    Overwrites the best epoch model file if appropriate,
    storing essential information (epoch and validation accuracy) within the file name.
    '''
    file_names = os.listdir(dir_path)
    file_names = [fn for fn in file_names
                  if 'best' in fn and '.pt' in fn and '_valacc' in fn]

    if len(file_names) == 0:
        print('No best epoch model weights file stored yet.')
        existing_path = None
        existing_val_acc = -1.0
    else:
        if len(file_names) > 1:
            print('WARNING: More than one matching file name found??? =>', file_names)
        existing_name = file_names[0]
        existing_path = os.path.join(dir_path, existing_name)
        existing_val_acc = float(existing_name.split('.pt')[0].split('_valacc')[1])

    if val_acc > existing_val_acc:
        if existing_path is not None:
            os.remove(existing_path)
        new_name = f'best_epoch{epoch}_valacc{val_acc:.7f}.pt'
        new_path = os.path.join(dir_path, new_name)
        torch.save(my_model.state_dict(), new_path)
        print(f'Previous best validation accuracy: {existing_val_acc:.5f}')
        print('Stored best epoch so far to:', new_path)


def _update_stored_metrics(dir_path, metrics_train, metrics_val, epoch):
    # Read or create.
    file_path = os.path.join(dir_path, 'running_metrics.p')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            metrics_dict = pickle.load(f)
    else:
        print('Metrics file does not exist yet, creating ' + file_path + '...')
        metrics_dict = dict()
        metrics_dict['train'] = dict()
        metrics_dict['val'] = dict()

    # Add epoch stats for each phase.
    metrics_dict['train'][epoch] = metrics_train
    metrics_dict['val'][epoch] = metrics_val

    # Write updated stats dictionary.
    with open(file_path, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print('Stored overall metrics to ' + file_path)


def _training_loop(
        my_model, train_loader, val_loader, device,
        location_losses, rectangle_loss, binary_loss,
        optimizer, scheduler, writer, grid_size,
        location_classes, lambda_patch, lambda_rect,
        patch_localization_only, mix_patch_batch,
        use_patches, use_global,
        start_epoch, num_epochs, checkpoint_dir, stop_after_epoch):
    for epoch in range(start_epoch, num_epochs):
        print('')
        print('==============')
        print(f'Epoch {epoch + 1} / {num_epochs}')
        print('Training phase')
        print('==============')

        # Log learning rate.
        learn_rate = _get_learning_rate(optimizer)
        print(f'Learning rate: {learn_rate:.5f}')
        writer.add_scalar(
            'misc/learning_rate', learn_rate, epoch)

        # Show example training images.
        if use_patches or use_global:
            mbatch = next(iter(train_loader))
            if use_patches:
                grid = torchvision.utils.make_grid(mbatch['patches'][0])
                writer.add_image('train/patches_upper_left', grid, epoch)
                del grid
                grid = torchvision.utils.make_grid(mbatch['patches'][-1])
                writer.add_image('train/patches_lower_right', grid, epoch)
                del grid
            if use_global:
                grid = torchvision.utils.make_grid(mbatch['global'])
                writer.add_image('train/global_images', grid, epoch)
                del grid
            del mbatch

        mean_losses_train, accuracies_train = _train_epoch(
            my_model, train_loader, device, epoch,
            location_losses, rectangle_loss, binary_loss,
            optimizer, writer, grid_size, location_classes,
            lambda_patch, lambda_rect, patch_localization_only,
            mix_patch_batch)
        scheduler.step()

        # Store checkpoint.
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, f'epoch{epoch}_train.pt')
        torch.save(my_model.state_dict(), save_path)

        print('')
        print('================')
        print(f'Epoch {epoch + 1} / {num_epochs}')
        print('Validation phase')
        print('================')

        # Show example validation images.
        if use_patches or use_global:
            mbatch = next(iter(val_loader))
            if use_patches:
                grid = torchvision.utils.make_grid(mbatch['patches'][0])
                writer.add_image('val/patches_upper_left', grid, epoch)
                del grid
                grid = torchvision.utils.make_grid(mbatch['patches'][-1])
                writer.add_image('val/patches_lower_right', grid, epoch)
                del grid
            if use_global:
                grid = torchvision.utils.make_grid(mbatch['global'])
                writer.add_image('val/global_images', grid, epoch)
                del grid
            del mbatch

        with torch.no_grad():
            mean_losses_val, accuracies_val = _val_epoch(
                my_model, val_loader, device, epoch,
                location_losses, rectangle_loss, binary_loss,
                writer, grid_size, location_classes,
                lambda_patch, lambda_rect, patch_localization_only,
                mix_patch_batch)

        # Keep track of best weights.
        _update_best_epoch_weights(checkpoint_dir, my_model, epoch, accuracies_val[-1])

        # Store overall metrics for this run.
        _update_stored_metrics(checkpoint_dir,
                               (mean_losses_train, accuracies_train),
                               (mean_losses_val, accuracies_val), epoch)

        # Force run garbage collection.
        gc.collect()

        # Memory leak workaround.
        if epoch == stop_after_epoch:
            print('Finishing program to avoid memory leak...')
            break


def main(args):

    # Enable reproducibility.
    # random.seed(1)
    # np.random.seed(1)
    # torch.manual_seed(1)

    # Overwrite args for consistency.
    if args.patch_localization_only:
        args.backbone_global = 'none'

    # Obtain model tag.
    model_tag = _get_model_tag(args)
    print('Model tag:', model_tag)

    # Define paths.
    checkpoint_dir, log_dir, image_dir, resume_path, model_tag = logistics.get_dirs_from_resume(
        args.checkpoint_dir, args.log_dir, args.image_dir, args.resume, model_tag)

    # Verify flags (especially useful if specifying resume path).
    _verify_flags(args, model_tag)

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
    lambda_patch = args.lambda_patch
    lambda_rect = args.lambda_rect
    patch_localization_only = args.patch_localization_only
    mix_patch_batch = args.mix_patch_batch
    aberration_config = None
    crop_probability = 0.0 if patch_localization_only else 0.5
    resize_max_width = 2048

    # Print flags.
    print('Checkpoint directory:', checkpoint_dir)
    print('Logging directory:', log_dir)
    print('Batch size:', args.batch_size)
    print('Number of epochs:', args.num_epochs)
    print('Stop after epochs:', args.stop_after_epochs)
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
    print('Patch loss term weight:', lambda_patch)
    print('Rectangle loss term weight:', lambda_rect)
    print('Patch localization only:', patch_localization_only)
    print('Mix patch batch:', mix_patch_batch)
    print('Crop probability:', crop_probability)
    print('Random resize max width:', resize_max_width)

    # Initialize dataset and data loader.
    print('Initializing data loaders...')
    train_dataset = data.ImageCropPatchScaleDataset(
        args.train_dir, size_factor_range, use_patches, use_global, grid_size, patch_dim,
        global_size, aberration_config, random_patch_positions, patch_jitter, crop_probability,
        True, resize_max_width,
        preresize_width='random' if post_crop_random_resize else None)
    val_dataset = data.ImageCropPatchScaleDataset(
        args.val_dir, size_factor_range, use_patches, use_global, grid_size, patch_dim,
        global_size, aberration_config, random_patch_positions, patch_jitter, crop_probability,
        True, resize_max_width,
        preresize_width='random' if post_crop_random_resize else None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        drop_last=True, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        drop_last=True, pin_memory=False)

    # Initialize model and device.
    print('Initializing model...')
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

    # Load weights if specified & get true learning rate.
    if os.path.isfile(resume_path):
        print('Loading weights from:', resume_path)
        my_model.load_state_dict(torch.load(resume_path))
        my_model.train()
        my_model = my_model.to(device)
        start_epoch = logistics.get_epoch_from_path(resume_path) + 1
        start_learn_rate = args.initial_learn_rate * (args.learn_rate_decay ** start_epoch)
        print(f'Last stored epoch is {start_epoch}, starting from epoch {start_epoch + 1}')
    else:
        start_epoch = 0
        start_learn_rate = args.initial_learn_rate

    # Detect whether training has already finished.
    if start_epoch >= args.num_epochs:
        print('WARNING: Last epoch has already finished; no further training needed')
        return

    # Memory leak workaround.
    if args.stop_after_epochs >= 1:
        stop_after_epoch = start_epoch + args.stop_after_epochs - 1
        print(f'Will stop the program after completing epoch: {stop_after_epoch + 1}')
    else:
        stop_after_epoch = -1
        print(f'WARNING: Memory leak might occur (no early stopping specified)')

    # Initialize loss functions.
    location_losses = []
    for _ in range(location_classes):
        location_losses.append(torch.nn.CrossEntropyLoss())
    rectangle_loss = torch.nn.MSELoss()
    binary_loss = torch.nn.BCEWithLogitsLoss()

    # Initialize optimizer, scheduler, and writer.
    optimizer = torch.optim.Adam(
        my_model.parameters(), lr=start_learn_rate,
        weight_decay=5e-4 if patch_localization_only else 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.learn_rate_decay)  # 100 epochs => 100x drop
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    # Start training
    _training_loop(
        my_model, train_loader, val_loader, device,
        location_losses, rectangle_loss, binary_loss,
        optimizer, scheduler, writer, grid_size,
        location_classes, lambda_patch, lambda_rect,
        patch_localization_only, mix_patch_batch,
        use_patches, use_global,
        start_epoch, args.num_epochs, checkpoint_dir, stop_after_epoch)


if __name__ == '__main__':

    if os.name != 'nt':
        # https://github.com/pytorch/pytorch/issues/973
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

    args = parser.parse_args()

    main(args)
