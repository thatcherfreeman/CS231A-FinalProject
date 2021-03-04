import argparse
import os
from typing import Tuple, List
import random
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import numpy as np # type: ignore
import torch
from torch import nn
from torchvision.transforms import functional as tv_f
from tqdm import tqdm # type: ignore
from torchvision import transforms
import tensorflow as tf
import tensorflow_datasets as tfds

import models

def save_model(model: nn.Module, path: str) -> nn.Module:
    model = model.cpu()
    torch.save(model.state_dict(), path)
    model = model.to(get_device())
    return model


def load_model(model: nn.Module, path: str) -> nn.Module:
    model = model.cpu()
    model.load_state_dict(torch.load(path))
    model = model.to(get_device())
    return model


def get_device() -> torch.device:
    '''
    Guesses the best device for the current machine.
    '''
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_training_data(args: argparse.Namespace) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    '''
    Returns Training and Development datasets
    '''
    train_percent = f'[:{round(100 * (1 - args.dev_frac)):d}%]'
    train_dataset : tf.data.Dataset = tfds.load('nyu_depth_v2', split=f'train{train_percent}', as_supervised=True)
    train_dataset = train_dataset.batch(args.train_batch_size)

    dev_percent = f'[-{round(100 * (args.dev_frac)):d}%:]'
    dev_dataset : tf.data.Dataset = tfds.load('nyu_depth_v2', split=f'train{dev_percent}', as_supervised=True)
    dev_dataset = dev_dataset.batch(args.dev_batch_size)
    return train_dataset, dev_dataset

def load_test_data(args: argparse.Namespace) -> tf.data.Dataset:
    test_dataset : tf.data.Dataset = tfds.load('nyu_depth_v2', split='validation', as_supervised=True)
    test_dataset = test_dataset.batch(args.batch_size)
    return test_dataset


def preprocess_training_example(np_image: np.ndarray, np_depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @Param image ndarray of shape N, H, W, C, values [0...255]
    @Param depth ndarray of shape N, H, W

    @Returns Tuple of
        image torch.Tensor of shape N, C, H', W', values [0...255]
        depth torch.Tensor of shape N, 1, H', W'

    Apply preprocessing to training example
    - Convert to pytorch tensor
    - Random crop
    - Color jitter
    - Random flip
    - Resizing
    '''

    # Consider moving to device before data augmentation.
    device = get_device()
    depth = torch.Tensor(np_depth).to(device)
    depth = torch.unsqueeze(depth, 1)
    image = torch.Tensor(np_image).to(device).permute(0, 3, 1, 2)

    N, C, H, W = image.size()

    # Color Jitter
    image = image / 255 # colorjitter caps at 1.0
    image = transforms.ColorJitter(0.4, 0.4, 0.4, 0)(image)
    image = image * 255

    # Rotation
    angle = transforms.RandomRotation.get_params([-5, 5])
    image = tv_f.rotate(image, angle, resample=2)
    depth = tv_f.rotate(depth, angle, resample=2)

    # Random Crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, (0.8, 1.0), (W/H, W/H))
    image = tv_f.crop(image, i, j, h, w)
    depth = tv_f.crop(depth, i, j, h, w)

    # Resize to output dimensions
    image = tv_f.resize(image, (H, W))
    depth = tv_f.resize(depth, (H, W))

    # Random Flip
    if torch.rand(1) > 0.5:
        image = tv_f.hflip(image)
        depth = tv_f.hflip(depth)

    return image, depth

def preprocess_test_example(np_image: np.ndarray, np_depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @Param image ndarray of shape N, H, W, C
    @Param depth ndarray of shape N, H, W

    @Returns Tuple of
        image torch.Tensor of shape N, C, H', W'
        depth torch.Tensor of shape N, 1, H', W'

    Preprocessing to be done on test examples
    '''
    # Resize for model input if necessary
    # Don't do random crop/jitter/flip
    device = get_device()
    depth = torch.Tensor(np_depth).to(device)
    depth = torch.unsqueeze(depth, 1)
    image = torch.Tensor(np_image).to(device).permute(0, 3, 1, 2)
    return image, depth

def depth_proportional_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) / (target + 0.5))

def l1_log_loss(input: torch.Tensor, pos_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log(torch.abs(input - pos_target)))

def l1_norm_loss(input: torch.Tensor, pos_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(input - pos_target))

def l2_norm_loss(input: torch.Tensor, pos_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.pow(input - pos_target, 2))

def verify_versions() -> None:
    # Version 1.5.0 has a bug where certain type annotations don't pass typecheck
    assert torch.__version__.startswith('1.7.1'), 'Incorrect torch version installed!'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_diagram(image: np.ndarray, input_image: np.ndarray, gt_depth: np.ndarray, pred_depth: np.ndarray, filename: str):
    '''
    image shape N, 3, H, W
    input_image shape N, 3, H, W
    gt_depth shape N, 1, H, W
    pred_depth shape N, 1, H, W
    filename str to save image into
    '''
    image = np.clip(image[0] / 255, 0, 1)
    input_image = np.clip(input_image[0] / 255, 0, 1)
    gt_depth = gt_depth[0, 0]
    pred_depth = pred_depth[0, 0]
    image = np.transpose(image, axes=(1, 2, 0))
    input_image = np.transpose(input_image, axes=(1, 2, 0))

    depth_min = 0
    depth_max = max(np.max(pred_depth), np.max(gt_depth))
    plt.figure(figsize=(6,1.25))
    plt.subplot(1, 4, 1)
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(1, 4, 2)
    plt.axis('off')
    plt.imshow(input_image)
    plt.subplot(1, 4, 3)
    plt.axis('off')
    plt.imshow(gt_depth, cmap='gray', vmin=depth_min, vmax=depth_max)
    plt.subplot(1, 4, 4)
    plt.axis('off')
    im = plt.imshow(pred_depth, cmap='gray', vmin=depth_min, vmax=depth_max)
    # _add_colorbar(im)

    plt.savefig(filename, dpi=640)
    plt.close()


def make_3_col_diagram(image: np.ndarray, gt_depth: np.ndarray, pred_depth: np.ndarray, filename: str):
    '''
    image shape N, 3, H, W
    input_image shape N, 3, H, W
    gt_depth shape N, 1, H, W
    pred_depth shape N, 1, H, W
    filename str to save image into
    '''
    image = np.clip(image[0] / 255, 0, 1)
    gt_depth = gt_depth[0, 0]
    pred_depth = pred_depth[0, 0]
    image = np.transpose(image, axes=(1, 2, 0))

    depth_min = 0
    depth_max = max(np.max(pred_depth), np.max(gt_depth))
    plt.figure(figsize=(6,1.25))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(gt_depth, cmap='gray', vmin=depth_min, vmax=depth_max)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    im = plt.imshow(pred_depth, cmap='gray', vmin=depth_min, vmax=depth_max)
    # _add_colorbar(im)

    plt.savefig(filename, dpi=640)
    plt.close()

def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
