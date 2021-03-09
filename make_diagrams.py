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


def make_images(
    dev_dl: tf.data.Dataset,
    model: nn.Module,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()

    print('  Running forward inference...')
    torch.set_grad_enabled(False)
    with tqdm(total=args.batch_size * len(dev_dl)) as progress_bar:
        for i, (x_batch_orig, y_batch) in enumerate(dev_dl.as_numpy_iterator()):
            x_batch, y_batch = model_utils.preprocess_test_example(x_batch_orig, y_batch)
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)

            # Forward pass on model
            y_pred = model(x_batch).detach()

            model_utils.make_3_col_diagram(x_batch.cpu().numpy(), y_batch.cpu().numpy(), y_pred.cpu().numpy(), f'{args.save_dir}/{args.name}/{args.name}_{i}.png')

            progress_bar.update(len(x_batch))

            del x_batch
            del y_pred

    return model


def main():
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    device = model_utils.get_device()

    assert args.name is not None
    os.makedirs(f'{args.save_dir}/{args.name}')

    # Load dataset from disk
    dev_dl = model_utils.load_test_data(args)
    dev_dl = dev_dl.take(args.num_images)

    # Initialize a model
    model = models.get_model(args.model)(args.size)

    # load from checkpoint if path specified
    assert args.load_path is not None
    model = model_utils.load_model(model, args.load_path)
    model.eval()

    # Move model to GPU if necessary
    model.to(device)

    # test!
    make_images(
        dev_dl,
        model,
        args,
    )


if __name__ == '__main__':
    model_utils.verify_versions()
    main()


