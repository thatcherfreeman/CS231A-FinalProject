import argparse
import os
from typing import Tuple, List
import random

import pickle
import numpy as np # type: ignore
import torch
from torch import nn
from tqdm import tqdm # type: ignore

import models
import tensorflow as tf
import tensorflow_datasets as tfds


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
    train_dataset : tf.data.Dataset = tfds.load('nyu_depth_v2', split=f'train{train_percent}')
    train_dataset = train_dataset.batch(args.train_batch_size).shuffle(1000, reshuffle_each_iteration=True)

    dev_percent = f'[-{round(100 * (args.dev_frac)):d}%:]'
    dev_dataset : tf.data.Dataset = tfds.load('nyu_depth_v2', split=f'train{dev_percent}')
    dev_dataset = dev_dataset.batch(args.dev_batch_size)
    return train_dataset, dev_dataset


def preprocess_training_example(image: np.ndarray, depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @Param image ndarray of shape N, H, W, C
    @Param depth ndarray of shape N, H, W

    @Returns Tuple of
        image torch.Tensor of shape N, H', W', C
        depth torch.Tensor of shape N, H', W'

    Apply preprocessing to training example
    - Convert to pytorch tensor
    TODO:
    - Random crop
    - Color jitter
    - Random flip
    - Resizing
    '''

    # Consider moving to device before data augmentation.
    return torch.Tensor(image), torch.Tensor(depth)

def preprocess_test_example(image: np.ndarray, depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @Param image ndarray of shape N, H, W, C
    @Param depth ndarray of shape N, H, W

    @Returns Tuple of
        image torch.Tensor of shape N, H', W', C
        depth torch.Tensor of shape N, H', W'

    Preprocessing to be done on test examples
    '''
    # Resize for model input if necessary
    # Don't do random crop/jitter/flip
    return torch.Tensor(image), torch.Tensor(depth)


def l1_norm_loss(input: torch.Tensor, pos_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(input - pos_target))

def l2_norm_loss(input: torch.Tensor, pos_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.pow(input - pos_target, 2))

def verify_versions() -> None:
    # Version 1.5.0 has a bug where certain type annotations don't pass typecheck
    assert torch.__version__ == '1.7.1', 'Incorrect torch version installed!'
