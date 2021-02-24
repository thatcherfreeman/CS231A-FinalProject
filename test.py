import argparse
import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # type: ignore

from args import add_test_args, add_common_args
import models
import model_utils


def test_model(
    dev_dl: data.DataLoader,
    model: nn.Module,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()

    print('\Computing evaluation metrics...')
    total_pixels = 0
    squared_error = 0
    rel_error = 0
    log_error = 0
    threshold1 = 0 # 1.25
    threshold2 = 0 # 1.25^2
    threshold3 = 0 # corresponds to 1.25^3

    print('  Running forward inference...')
    torch.set_grad_enabled(False)
    with tqdm(total=args.batch_size * len(dev_dl)) as progress_bar:
        for i, (x_batch, y_batch) in enumerate(dev_dl):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass on model
            y_pred = model(x_batch).detach()

            # TODO: Process y_pred in the optimal way (round it off, etc)
            # Maybe clamp from 0 to infty or something

            # RMS, REL, LOG10, threshold calculation
            squared_error += torch.sum(torch.pow(y_pred - y_batch, 2)).item()
            rel_error += torch.sum(torch.abs(y_pred - y_batch) / y_batch).item()
            log_error += torch.sum(torch.abs(torch.log10(y_pred) - torch.log10(y_batch))).item()
            threshold1 += torch.sum(torch.max(y_pred / y_batch, y_batch / y_pred) < 1.25).item()
            threshold2 += torch.sum(torch.max(y_pred / y_batch, y_batch / y_pred) < 1.25**2).item()
            threshold3 += torch.sum(torch.max(y_pred / y_batch, y_batch / y_pred) < 1.25**3).item()
            total_pixels += np.prod(y_batch.shape)

            progress_bar.update(len(x_batch))

            del x_batch
            del y_pred
            del y_batch

    print('\n  Calculating overall metrics...')
    print()
    print('*' * 30)
    print(f'RMS:   {(squared_error / total_pixels)**0.5}')
    print(f'REL:   {rel_error / total_pixels}')
    print(f'LOG10: {log_error / total_pixels}')
    print(f'delta1:{threshold1 / total_pixels}')
    print(f'delta2:{threshold2 / total_pixels}')
    print(f'delta3:{threshold3 / total_pixels}')
    print('*' * 30)

    return model


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


