"""
Some boilerplate code for the main function, containing common command line
arguments. Does not do anything meaningful as is.
"""

import argparse
import logging
import datetime
from pathlib import Path

from lib.trainer import TrainerConfig, Trainer
from lib.metrics import AccuracyCallback
from lib.callbacks import EarlyStoppingCallback, MLFlowCallback
from lib.torch_utils import set_seed
from lib.util import conditional_email_sender

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset


@conditional_email_sender
def main(args):
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # set up logging
    TIMESTAMP = datetime.datetime.today().strftime("%y%m%d_%H:%M")
    log_file = checkpoint_path / f"train_{TIMESTAMP}.log"
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Writing log output to {log_file}")

    set_seed(args.seed)

    model = torchvision.models.resnet18()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # For the default scheduler, use a cooldown of 10000 epochs to ensure that the
    # learning rate is only decayed once. Reducing the LR only once prevents problems
    # with decaying too much/fast, as proposed by Andrej Karpathy in a 27-08-21 tweet.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, cooldown=10000
    )

    config = TrainerConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        use_swa=args.use_swa,
    )
    # trainer = Trainer(
    #    config,
    #    model,
    #    optimizer,
    #    ds_train,
    #    ds_eval,
    #    loss_fn=criterion,
    #    callbacks=[
    #        AccuracyCallback(monitor=True),
    #        CheckpointCallback(checkpoint_path, monitor="accuracy"),
    #        # EarlyStoppingCallback(max_epochs_no_change=5),
    #        MLFlowCallback(
    #           experiment_name=args.experiment_name,
    #           artifacts_to_log=[log_file],
    #           parameters_to_log={"learning_rate": args.learning_rate})
    #    ]
    # )


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--experiment_name", type=str, default="default",
                        help="Name of the experiment that will be logged in mlflow.")
    parser.add_argument("--load_model", type=str,
                        help=("Load a model from a checkpoint specified by a "
                              "path with a .tar extension."))
    parser.add_argument("--data_path", type=str,
                        help="Path where the data is stored.")
    parser.add_argument("--checkpoint_path", type=str,
                        default=str(Path().cwd() / "checkpoints"),
                        help="Path to directory where to store model checkpoints.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--use_swa", action="store_true", default=False,
                        help="If set, use Stochastic Weight averaging (SWA).")
    parser.add_argument("--max_epochs_no_change", type=int, default=5,
                        help="Number of epochs until early stopping occurs.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="How many subprocesses to use for data loading.")
    parser.add_argument("--debug_mode", action="store_true", default=False,
                        help="Use a small subset of the data to enable faster debugging")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for initializing random number generators.")
    parser.add_argument("--email_to_notify", type=str, default=None,
                        help=("Specify an email which will be notified when training "
                              "starts/end/crashes."))
    args = parser.parse_args()
    # fmt: on

    main(args)
