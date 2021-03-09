import argparse
import os
import json
from typing import Any, Tuple, Dict


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

def load_arguments(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as f:
        obj = json.load(f)
        return obj


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
        default=3,
        help='Number of epochs to train for',
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=1,
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
        default='Baseline34',
        help='choose the model to train',
    )
    parser.add_argument(
        '--load_path',
        type=str,
        default=None,
        help='specify path to .checkpoint file to load the model at the given path before training.'
    )
    parser.add_argument(
        '--load_dir',
        type=str,
        default=None,
        help='specify path to folder containing .checkpoint files.'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Give the model a name that will be a part of the experiment path.',
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs='+',
        default=(288,384),
        help='specify the internal processing size of the model'
    )

def add_test_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--num_images',
        type=int,
        default=5,
        help='Selects the number of images to make.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Number of images to process at once. Don\'t change this one.',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='diagrams',
        help='Folder to save diagrams into'
    )
