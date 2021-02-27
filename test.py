import argparse
import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # type: ignore

from args import add_test_args, add_common_args
import models
import model_utils


def test_model(
    dev_dl: tf.data.Dataset,
    model: nn.Module,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()

    print('\Computing evaluation metrics...')
    total_pixels = 0
    total_examples = 0
    squared_error = 0
    rel_error = 0
    log_error = 0
    threshold1 = 0 # 1.25
    threshold2 = 0 # 1.25^2
    threshold3 = 0 # corresponds to 1.25^3
    eps = 0.5

    print('  Running forward inference...')
    torch.set_grad_enabled(False)
    with tqdm(total=args.batch_size * len(dev_dl)) as progress_bar:
        for i, (x_batch_orig, y_batch) in enumerate(dev_dl.as_numpy_iterator()):
            x_batch, y_batch = model_utils.preprocess_test_example(x_batch_orig, y_batch)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass on model
            y_pred = model(x_batch).detach()

            # TODO: Process y_pred in the optimal way (round it off, etc)
            # Maybe clamp from 0 to infty or something
            nanmask = getNanMask(y_batch)
            total_pixels = torch.sum(~nanmask)
            total_examples += x_batch.shape[0]
            
            # RMS, REL, LOG10, threshold calculation
            squared_error += (torch.sum(torch.pow(y_pred - y_batch, 2)).item() / total_pixels)**0.5
            rel_error += torch.sum(torch.abs(y_pred - y_batch) / y_batch).item() / total_pixels
            log_error += torch.sum(torch.abs(removeNans(torch.log10(y_pred)) - removeNans(torch.log10(y_batch)))).item() / total_pixels
            threshold1 += torch.sum(torch.max(y_pred / y_batch, y_batch / y_pred) < 1.25).item() / total_pixels
            threshold2 += torch.sum(torch.max(y_pred / y_batch, y_batch / y_pred) < 1.25**2).item() / total_pixels
            threshold3 += torch.sum(torch.max(y_pred / y_batch, y_batch / y_pred) < 1.25**3).item() / total_pixels
            # total_pixels += np.prod(y_batch.shape)

            progress_bar.update(len(x_batch))

            del x_batch
            del y_pred
            del y_batch

    print('\n  Calculating overall metrics...')
    print()
    print('*' * 30)
    print(f'RMS:   {squared_error / total_examples}')
    print(f'REL:   {rel_error / total_examples}')
    print(f'LOG10: {log_error / total_examples}')
    print(f'delta1:{threshold1 / total_examples}')
    print(f'delta2:{threshold2 / total_examples}')
    print(f'delta3:{threshold3 / total_examples}')
    print('*' * 30)

    return model

def removeNans(mat: torch.Tensor) -> torch.Tensor:
    nanmask = torch.logical_or(torch.isnan(mat), torch.isinf(mat))
    _mat = mat.clone()
    _mat[nanmask] = 0
    return _mat

def getNanMask(mat: torch.Tensor) -> torch.Tensor:
    nanmask = torch.logical_or(torch.isnan(mat), torch.isinf(mat))
    return nanmask

def main():
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    device = model_utils.get_device()

    # Load dataset from disk
    dev_dl = model_utils.load_test_data(args)

    # Initialize a model
    model = models.get_model(args.model)()

    # load from checkpoint if path specified
    assert args.load_path is not None
    model = model_utils.load_model(model, args.load_path)
    model.eval()

    # Move model to GPU if necessary
    model.to(device)

    # test!
    test_model(
        dev_dl,
        model,
        args,
    )


if __name__ == '__main__':
    model_utils.verify_versions()
    main()


