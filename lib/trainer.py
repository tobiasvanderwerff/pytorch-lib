"""
Basic training loop. This code is meant to be generic and can be used to train different types of neural networks.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Callable, Sequence
from pathlib import Path

from .callbacks import CallbackHandler, TrainerCallback, CheckpointCallback

import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


logger = logging.getLogger(__name__)

DEFAULT_CALLBACKS = [CheckpointCallback("./checkpoints/")]


@dataclass
class TrainerConfig:
    """
    Args:
        batch_size (int): batch size
        epochs (int): max number of epochs to train
        grad_norm_clip (float): norm at which to clip the gradients
        max_epochs_no_change (int): number of epochs until early stopping occurs
        num_workers: how many subprocesses to use for data loading. `0` means
            that the data will be loaded in the main process.
        checkpoint_path (Path or str): directory to save checkpoints to
        use_mixed_precision (bool): use mixed precision. This means that
            float16 is used whenever possible, rather than float32, which is the
            default behavior. This can significantly speed up training.
        model_name (str): model name used when saving checkpoints. The model
            name will include a model_name key carrying this name.
    """

    batch_size: int = 128
    epochs: int = 10
    grad_norm_clip: float = 5.0
    max_epochs_no_change: int = 10
    num_workers: int = 0
    use_mixed_precision: bool = True  # if true, use float16 where possible
    model_name: str = None

    def to_csv(self):
        # TODO
        raise NotImplementedError("Not implemented yet")


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_ds: Dataset,
        eval_ds: Dataset = None,
        test_ds: Dataset = None,
        loss_fn: nn.Module = None,
        train_batch_sampler: Sampler = None,
        train_collate_fn: Callable = None,
        eval_collate_fn: Callable = None,
        evaluation_callback_fn: Callable = None,
        callbacks: Sequence[TrainerCallback] = None,
        metric_callbacks: Sequence[TrainerCallback] = None,
    ):
        """
        Args:
            config (TrainerConfig)
            model (nn.Module): model
            optimizer (optim.Optimizer) optimizer
            train_ds (Dataset): dataset for training the model
            eval_ds (Dataset): dataset for evaluating the model (optional)
            test_ds (Dataset): dataset for testing the model after
                training (optional)
            loss_fn (nn.Module): loss function to use during training
            train_batch_sampler (torch.utils.data.Sampler): batch_sampler
                argument to use for the training dataloader
            train_collate_fn (Callable): collate_fn argument for the
                training dataloader
            eval_collate_fn (Callable): collate_fn argument for the
                eval dataloader
            callbacks (Sequence[TrainerCallback]): list of callbacks to call during
                training/evaluation
        """

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.test_ds = test_ds
        self.loss_fn = loss_fn
        self.train_batch_sampler = train_batch_sampler
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn

        self.best_scores = {}
        self.best_state_dict = None
        self.epoch = 0
        self.early_stopping_active = False
        self.losses = {"train": [], "eval": [], "test": []}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        callbacks = (
            DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        )
        self.callback_handler = CallbackHandler(callbacks)

    def get_train_dataloader(self):
        if self.train_batch_sampler:
            trainloader = DataLoader(
                self.train_ds,
                batch_sampler=self.train_batch_sampler,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=self.train_collate_fn,
            )
        else:
            trainloader = DataLoader(
                self.train_ds,
                self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=self.train_collate_fn,
            )
        return trainloader

    def get_eval_dataloader(self):
        if self.eval_ds is not None:
            return DataLoader(
                self.eval_ds,
                self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=self.eval_collate_fn,
            )

    def get_test_dataloader(self):
        if self.test_ds is not None:
            return DataLoader(
                self.test_ds,
                self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=self.eval_collate_fn,
            )

    def train(self):
        model, optimizer, config = self.model, self.optimizer, self.config

        trainloader = self.get_train_dataloader()
        evalloader = self.get_eval_dataloader()
        testloader = self.get_test_dataloader()

        scaler = torch.cuda.amp.GradScaler()  # used for mixed precision training

        def run_epoch(split):
            if split == "train":
                self.callback_handler.on_train_epoch_start(self)
            elif split == "eval":
                self.callback_handler.on_validation_epoch_start(self)

            is_train = True if split == "train" else False
            losses = []
            self.epoch_metrics = {}

            if split == "train":
                dataloader = trainloader
            elif split == "eval":
                dataloader = evalloader
            else:
                dataloader = testloader

            criterion = self.loss_fn if self.loss_fn else nn.CrossEntropyLoss()

            pbar = tqdm(dataloader, total=len(dataloader)) if is_train else dataloader
            for data in pbar:
                model.train(is_train)  # put model in training or evaluation mode

                # put data on the appropriate device (cpu or gpu)
                imgs, *targets = [el.to(self.device) for el in data]
                if len(targets) == 1:
                    targets = targets[0]

                with torch.cuda.amp.autocast(
                    config.use_mixed_precision
                ):  # mixed precision
                    logits = model(imgs)
                    loss = criterion(logits, targets)

                losses.append(loss.item())
                self.losses[split].append(loss.item())

                # calculate metrics
                self.epoch_metrics.update(
                    self.callback_handler.on_evaluate(self, logits, targets)
                )

                if is_train:
                    if config.use_mixed_precision:
                        # scale loss, to avoid underflow when using mixed precision
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    self.callback_handler.on_after_backward(self)
                    # clip gradients to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    if config.use_mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()  # updates the scale for next iteration
                    else:
                        optimizer.step()  # update weights
                    optimizer.zero_grad()  # set the gradients back to zero
            epoch_loss = np.mean(losses)
            info_str = f"epoch {ep} - {split}_loss: {epoch_loss:.4f}. "
            if split == "eval":
                for metric_name in self.epoch_metrics.keys():
                    info_str += (
                        f"{metric_name}: {self.epoch_metrics[metric_name]:.4f}. "
                    )
            logger.info(info_str)

            if split == "train":
                self.callback_handler.on_train_epoch_end(self)
            elif split == "eval":
                self.callback_handler.on_validation_epoch_end(self)

        for ep in range(config.epochs):
            self.epoch = ep
            run_epoch("train")
            #             plot_grad_flow(model.named_parameters())
            if self.eval_ds is not None:
                with torch.no_grad():
                    run_epoch("eval")
            if self.early_stopping_active:
                logger.info(
                    f"Stopped early at epoch {ep}. Best scores: {self.best_scores}"
                )
                if self.test_ds is not None:
                    logger.info("Calculating results on test set...")
                    model.load_state_dict(self.best_state_dict)
                    with torch.no_grad():
                        run_epoch("test")
                break
        self.callback_handler.on_fit_end(self)
