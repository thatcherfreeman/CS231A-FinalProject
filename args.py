import argparse
import os
import json


def add_experiment(args: argparse.Namespace) -> None:
    if args.save_path not in os.listdir('.'):
        os.makedirs(args.save_path)
    num_folders = len(os.listdir(args.save_path))
    args.experiment = f'{args.model}_exp{num_folders}'
    if args.name is not None:
        args.experiment += f'_{args.name}'


def save_arguments(args: argparse.Namespace, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for the optimizer',
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=1,
        help='training mini-batch size',
    )
    parser.add_argument(
        '--dev_batch_size',
        type=int,
        default=1,
        help='development mini-batch size',
    )
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=30,
        help='Number of epochs to train for',
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=5,
        help='Number of epochs between each checkpoint',
    )
    parser.add_argument(
        '--num_checkpoints',
        type=int,
        default=6,
        help='Number of checkpoints to keep'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='regularization strength',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='checkpoints',
        help='specify path to save the trained model'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Directory to store tensorboard logs',
    )
    parser.add_argument(
        '--use_scheduler',
        action='store_true',
        help='Use this flag to avoid learning rate scheduling.',
    )
    parser.add_argument(
        '--dev_frac',
        type=float,
        default=0.03,
        help='Indicates fraction of data to be partitioned into dev set.',
    )
    parser.add_argument(
        '--picture_frequency',
        type=int,
        default=10000,
        help='Every x batches, save image of predicted depth map.',
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--model',
        type=str,
        default='Baseline',
        help='choose the model to train',
    )
    parser.add_argument(
        '--load_path',
        type=str,
        default=None,
        help='specify path to load the model at the given path before training.'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Give the model a name that will be a part of the experiment path.',
    )


def add_test_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='mini-batch size',
    )
