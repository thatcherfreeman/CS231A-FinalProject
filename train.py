import argparse
import os

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # type: ignore
import tensorflow as tf

from args import add_train_args, add_experiment, add_common_args, save_arguments
import models
import model_utils


def train_model(
    train_ds: tf.data.Dataset,
    dev_ds: tf.data.Dataset,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()
    loss_fn = model_utils.l1_norm_loss # TODO: Set this to the correct loss fn
    val_loss_fn = model_utils.l1_norm_loss # TODO: Set this to the correct loss fn
    best_val_loss = torch.tensor(float('inf'))
    saved_checkpoints = []
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{args.experiment}')

    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    for e in range(1, args.train_epochs + 1):
        print(f'Training epoch {e}...')

        # Training portion
        torch.cuda.empty_cache()
        with tqdm(total=args.train_batch_size * len(train_ds)) as progress_bar:
            model.train()
            for i, (x_batch, y_batch) in enumerate(train_ds.as_numpy_iterator()):
                x_batch, y_batch = model_utils.preprocess_training_example(x_batch, y_batch)
                # x_batch = x_batch.to(device)
                # y_batch = y_batch.to(device)
                y_batch = y_batch.cuda(non_blocking=True)
                x_batch = x_batch.cuda()
                x_batch = torch.autograd.Variable(x_batch)
                y_batch = torch.autograd.Variable(y_batch)

                ones = torch.ones(y_batch.size(0), 1, y_batch.size(2),y_batch.size(3)).float().cuda()
                ones = torch.autograd.Variable(ones)

                # Forward pass on model
                optimizer.zero_grad()
                y_pred = model(x_batch)

                depth_grad = get_gradient(y_batch)
                output_grad = get_gradient(y_pred)
                depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(y_batch)
                depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(y_batch)
                output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(y_batch)
                output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(y_batch)

                depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
                output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

                # depth_normal = F.normalize(depth_normal, p=2, dim=1)
                # output_normal = F.normalize(output_normal, p=2, dim=1)

                loss_depth = torch.log(torch.abs(y_pred - y_batch) + 0.5).mean()
                loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
                loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
                loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

                loss = loss_depth + loss_normal + (loss_dx + loss_dy)
                # loss = loss_fn(y_pred, y_batch)
                losses.update(loss.data[0], image.size(0))

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                if args.use_scheduler:
                    lr_scheduler.step(loss)

                progress_bar.update(len(x_batch))
                progress_bar.set_postfix(loss=loss.item())
                writer.add_scalar("train/Loss", loss, ((e - 1) * len(train_ds) + i) * args.train_batch_size)

                # Periodically save a diagram
                if (i + 1) % args.picture_frequency == 0:
                    model_utils.make_diagram(
                        x_batch.cpu().numpy(),
                        y_batch.cpu().numpy(),
                        y_pred.cpu().detach().numpy(),
                        f'{args.save_path}/{args.experiment}/diagram_{e}_{i+1}.png',
                    )

                del x_batch
                del y_batch
                del y_pred
                del loss

        # Validation portion
        torch.cuda.empty_cache()
        with tqdm(total=args.dev_batch_size * len(dev_ds)) as progress_bar:
            model.eval()
            val_loss = 0.0
            num_batches_processed = 0
            for i, (x_batch, y_batch) in enumerate(dev_ds.as_numpy_iterator()):
                x_batch, y_batch = model_utils.preprocess_test_example(x_batch, y_batch)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass on model in validation environment
                y_pred = model(x_batch)

                # TODO: Process y_pred in whatever way inference requires.

                loss = val_loss_fn(y_pred, y_batch)

                val_loss += loss.item()
                num_batches_processed += 1

                progress_bar.update(len(x_batch))
                progress_bar.set_postfix(val_loss=val_loss / num_batches_processed)
                writer.add_scalar("Val/Loss", loss, ((e - 1) * len(dev_ds) + i) * args.dev_batch_size)

                del x_batch
                del y_batch
                del y_pred
                del loss

            # Save model if it's the best one yet.
            if val_loss / num_batches_processed < best_val_loss:
                best_val_loss = val_loss / num_batches_processed
                filename = f'{args.save_path}/{args.experiment}/{model.__class__.__name__}_best_val.checkpoint'
                model_utils.save_model(model, filename)
                print(f'Model saved!')
                print(f'Best validation loss yet: {best_val_loss}')
            # Save model on checkpoints.
            if e % args.checkpoint_freq == 0:
                filename = f'{args.save_path}/{args.experiment}/{model.__class__.__name__}_epoch_{e}.checkpoint'
                model_utils.save_model(model, filename)
                print(f'Model checkpoint reached!')
                saved_checkpoints.append(filename)
                # Delete checkpoints if there are too many
                while len(saved_checkpoints) > args.num_checkpoints:
                    os.remove(saved_checkpoints.pop(0))

    return model


def main():
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    add_experiment(args)
    device = model_utils.get_device()

    # Load dataset from disk
    train_ds, dev_ds = model_utils.load_training_data(args)

    # Initialize a model
    model = models.get_model(args.model)()

    # load from checkpoint if path specified
    if args.load_path is not None:
        model = model_utils.load_model(model, args.load_path)

    # Move model to GPU if necessary
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=30,
        verbose=True,
    )

    os.makedirs(f'{args.save_path}/{args.experiment}')
    print(f'Created new experiment: {args.experiment}')
    save_arguments(args, f'{args.save_path}/{args.experiment}/args.txt')

    # Train!
    trained_model = train_model(
        train_ds,
        dev_ds,
        model,
        optimizer,
        scheduler,
        args,
    )

    # Save trained model
    filename = f'{args.save_path}/{args.experiment}/{model.__class__.__name__}_trained.checkpoint'
    model_utils.save_model(trained_model, filename)


if __name__ == '__main__':
    model_utils.verify_versions()
    main()
