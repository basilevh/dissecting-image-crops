'''
Image crop detection by absolute patch localization.
Neural network architecture description in PyTorch.
Basile Van Hoorick, Fall 2020.
'''

# Library imports.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import scipy
import seaborn as sns
import shutil
import time
import torch
import torch.nn
import torch.nn.functional
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import torchvision
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import torchvision.utils
import tqdm

# Repository imports.
import data


def _create_resnet(name, input_channels, output_size):
    name = name.lower()
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(512, output_size)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(512, output_size)
    elif name == 'none':
        model = None
    else:
        raise ValueError('Unknown resnet backbone.')
    return model


def _add_coordinates_tensor(input_tensor):
    '''
    Adds coordinates of every pixel as fractions of image dimensions in [0, 1].

    Args:
        input_tensor: (B, 3, H, W) tensor.

    Returns:
        (B, 5, H, W) tensor with two extra channels describing (y, x).
    '''
    B, C, H, W = input_tensor.shape
    coords = torch.zeros_like(input_tensor[:, :2])  # Same device and data type.
    y = torch.linspace(start=0.0, end=1.0, steps=H)  # (H)
    x = torch.linspace(start=0.0, end=1.0, steps=W)  # (W)
    y = y.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, H, 1)
    x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, W)
    y = y.repeat((B, 1, 1, W))  # (B, 1, H, W)
    x = x.repeat((B, 1, H, 1))  # (B, 1, H, W)
    coords[:, 0:1] = y  # Device and/or data type conversion will happen automatically.
    coords[:, 1:2] = x
    result = torch.cat((input_tensor, coords), dim=1)
    return result


class JointLocationCropRectangleNet(torch.nn.Module):
    '''
    Deep neural network with (up to) two input branches and (up to) three output branches.
    Inputs are image patches and downscaled global images.
    Outputs are position estimates, crop rectangles, and binary crop decisions.
    '''

    def __init__(self, backbone_patch='resnet18', backbone_global='resnet34',
                 location_classes=16, patch_embedding_size=64, global_embedding_size=64,
                 patch_localization_only=False, add_coordinates_global=True,
                 mix_patch_batch=True):
        '''
        Initializes the model, backbones, and embedding dimensionality.

        Args:
            backbone_patch: resnet18 / resnet34 / none.
            backbone_global: resnet18 / resnet34 / none.
            location_classes: Total number of grid cells to classify patches into (= M).
            patch_embedding_size: Number of features extracted per patch.
            global_embedding_size: Number of features extracted from global.
            patch_localization_only: If True, do not produce crop rectangle or classification.
            add_coordinates_global: If True, add (x, y) as two extra channels to incoming globals.
            mix_patch_batch: If True, mix patch location labels within a batch by using cyclic
                shifts to counter the peculiar BatchNorm-related memory effect during train mode.
        '''
        super(JointLocationCropRectangleNet, self).__init__()

        # Define backbones for small patches and global images.
        # Sometimes either patches or globals are not used; keep track of this in boolean flags.
        self._backbone_patch = _create_resnet(
            backbone_patch, 3, patch_embedding_size)
        self._backbone_global = _create_resnet(
            backbone_global, 5 if add_coordinates_global else 3, global_embedding_size)
        self._use_patches = (
            backbone_patch is not 'none' and self._backbone_patch is not None)
        self._use_global = (
            backbone_global is not 'none' and self._backbone_global is not None)

        # Define parameters.
        self._location_classes = location_classes
        self._patch_embedding_size = patch_embedding_size
        self._global_embedding_size = global_embedding_size
        if self._use_patches and self._use_global:
            self._total_embedding_size = location_classes * \
                patch_embedding_size + global_embedding_size
            print('===> Both patches and globals are used.')
        elif self._use_patches and not(self._use_global):
            self._total_embedding_size = location_classes * patch_embedding_size
            print('===> Only patches are used.')
        elif not(self._use_patches) and self._use_global:
            self._total_embedding_size = global_embedding_size
            print('===> Only globals are used.')
        else:
            raise ValueError(
                'Invalid configuration: Both patches and globals are disabled.')
        self._patch_localization_only = patch_localization_only
        self._add_coordinates_global = add_coordinates_global
        self._mix_patch_batch = mix_patch_batch

        # Define heads for patch localization and crop rectangle prediction.
        if self._use_patches:
            self._location_net = torch.nn.Linear(
                patch_embedding_size, location_classes)
        else:
            self._location_net = None

        self._rectangle_net = torch.nn.Sequential(
            torch.nn.Linear(self._total_embedding_size, 512),
            # torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            # torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 5)
        )

    def forward(self, patches_tensor, global_image, clone_input=False, return_embeddings=False):
        '''
        Args:
            patches_tensor: (B, M, 3, P, P) tensor consisting of M=16 small image patches.
            global_image: (B, 3, H, W) tensor representing the thumbnail.
            clone_input: Clone inputs before modifying.
            return_embeddings: If True, add concatenated embedding to the returned tuple.
                For detailed data collection and visualization purposes.

        Returns:
            (locations_list, rectangle_score, embeddings).
            locations_list: list(M) of (B, M) tensors.
            rectangle_score: (B, 5) tensor.
            embeddings: Only if return_embeddings is True. (embedding_tensor, rect_net_penult).
                embeddings_list: list(M + 1) of (B, 64) tensors.
                rect_net_penult: (B, 256) tensor representing the intermediate rectangle_net
                    output before the last ReLU and Linear.
        '''

        embeddings_list = []  # length-17 list of (B, 64)
        locations_list = []  # length-16 list of (B, 16)
        
        if clone_input:
            if patches_tensor is not None:
                patches_tensor = patches_tensor.clone()
            if global_image is not None:
                global_image = global_image.clone()

        if self._use_patches:
            
            batch_size = patches_tensor.shape[0]
            num_patches = patches_tensor.shape[1]  # (B, 16, 3, 96, 96)
            forward_shifts = np.arange(batch_size)
            backward_shifts = -forward_shifts
            # backward_shifts = batch_size - forward_shifts
            # backward_shifts[0] = 0
            # print(forward_shifts, backward_shifts)

            if self._mix_patch_batch:
                # assert(self._patch_embedding_size == self._global_embedding_size)
                patches_tensor = data.roll_per_batch_index(patches_tensor, forward_shifts)
                # (B, 16, 64).
                embeddings_tensor = torch.zeros(
                    (batch_size, num_patches, self._patch_embedding_size)).to(patches_tensor.device)
                # (B, 16, 16).
                locations_tensor = torch.zeros(
                    (batch_size, num_patches, num_patches)).to(patches_tensor.device)
                    
            for i in range(num_patches):
                patch = patches_tensor[:, i, :, :, :]
                embedding = self._backbone_patch(patch)  # (B, 64)
                location = self._location_net(embedding)  # (B, 16)
                
                if self._mix_patch_batch:
                    embeddings_tensor[:, i] = embedding
                    locations_tensor[:, i] = location
                
                else:
                    embeddings_list.append(embedding)  # up to length-16 list of (B, 64)
                    locations_list.append(location)  # up to length-16 list of (B, 16)
            
            if self._mix_patch_batch:
                # The embeddings and location vectors need to be cyclically shifted back.
                # NOTE: Make sure the first two dimensions are all equal to those of patches_tensor.
                embeddings_tensor = data.roll_per_batch_index(embeddings_tensor, backward_shifts)
                locations_tensor = data.roll_per_batch_index(locations_tensor, backward_shifts)
                embeddings_list = [embeddings_tensor[:, i] for i in range(num_patches)]
                locations_list = [locations_tensor[:, i] for i in range(num_patches)]
                
                # Ensure that the input appears unmodified after inference!
                patches_tensor = data.roll_per_batch_index(patches_tensor, backward_shifts)

        # print('embeddings_list:', len(embeddings_list), embeddings_list[0].shape)
        # print(np.array([torch.sum(x).item() for x in embeddings_list]))
        # print('locations_list:', len(locations_list), locations_list[0].shape)
        # print(np.array([torch.sum(x).item() for x in locations_list]))

        # From now on: ignore embeddings_tensor, locations_tensor, and use
        # embeddings_list, locations_list instead.

        if not self._patch_localization_only:

            if self._use_global:

                if self._add_coordinates_global:
                    global_image = _add_coordinates_tensor(global_image)

                embedding = self._backbone_global(global_image)
                embeddings_list.append(embedding)

            # embeddings_list now contains 17 tensors of shape (B, 64).
            # We wish to convert this to a tensor of shape (B, 1088).
            embedding_tensor = torch.cat(embeddings_list, dim=1)  # (B, 1088)

            # Note: this will fail (as desired) if embeddings_tensor does not match total_embedding_size.
            rectangle_score = self._rectangle_net(embedding_tensor)  # (B, 5)

            if return_embeddings:
                rect_net_penult = self._rectangle_net[:-2](embedding_tensor)
                embeddings_return = (embedding_tensor, rect_net_penult)
                return locations_list, rectangle_score, embeddings_return

            else:
                # locations_list is empty if patches are disabled.
                # rectangle_score will always exist regardless of configuration.
                return locations_list, rectangle_score  # tuple(list of (B, 16), (B, 5))

        else:
            # We care about absolute patch localization results only.

            if return_embeddings:
                embedding_tensor = torch.cat(embeddings_list, dim=1)  # (B, 1024)
                embeddings_return = (embedding_tensor, None)
                return locations_list, None, embeddings_return

            else:
                return locations_list, None  # tuple(list of (B, 16), None)
