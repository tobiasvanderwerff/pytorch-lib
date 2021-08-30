"""
Basic training loop. This code is meant to be generic and can be used to train different types of neural networks.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Callable, Sequence, Dict, Any
from pathlib import Path

from .callbacks import (
    CallbackHandler,
    TrainerCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricCallback,
)

import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from mlflow import log_metric, log_param, log_artifacts


logger = logging.getLogger(__name__)

DEFAULT_CALLBACKS = []


@dataclass
class TrainerConfig:
    """
    Args:
        batch_size (int): batch size
        max_epochs (int): max number of epochs to train
        grad_norm_clip (float): norm at which to clip the gradients
        max_epochs_no_change (int): number of epochs until early stopping occurs
        num_workers: how many subprocesses to use for data loading. `0` means
            that the data will be loaded in the main process.
        checkpoint_path (Path or str): directory to save checkpoints to
        use_mixed_precision (bool): use mixed precision. This means that
            float16 is used whenever possible, rather than float32, which is the
            default behavior. This can significantly speed up training and
            lower memory footprint.
        use_swa (bool): use Stochastic Weight Averaging (SWA). See
            https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
            for more details.
        swa_start (float): only used if use_swa=True. This parameter indicates when to
            start averaging weights. Averaging then starts when `current_epoch >=
            swa_start * max_epochs`. Note that early stopping can interfere with this,
            so it might be useful to train using one or the other.
        model_name (str): model name used when saving checkpoints. The model
            name will include a model_name key carrying this name.
    """

    batch_size: int = 128
    max_epochs: int = 10
    grad_norm_clip: float = 5.0
    max_epochs_no_change: int = 10
    num_workers: int = 0
    use_mixed_precision: bool = True
    use_swa: bool = False
    swa_start: float = 0.75
    model_name: str = None

    def dump(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dictionary containing all hyperparameters specified above
        """
        return self.__dict__

    def to_csv(self):
        # TODO
        raise NotImplementedError()


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
        callbacks: Sequence[TrainerCallback] = None,
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

        if config.use_swa:
            # TODO: maybe it makes sense to create a SWA callback, which has hooks for
            # trainer init, and on_train_epoch_end.

            logger.info("Using Stochastic Weight Averaging (SWA).")
            # AveragedModel class keeps track of running averages of model
            # weights, used for averaging after training.
            self.swa_model = optim.swa_utils.AveragedModel(model)
            # SWALR is a learning rate scheduler that anneals the learning rate to a
            # fixed value, and then keeps it constant.
            # TODO: how to set these parameters properly? Right now they are based on
            # what I say from other implementations.
            self.swa_scheduler = optim.swa_utils.SWALR(
                optimizer,
                swa_lr=optimizer.param_groups[0]["lr"] / 10,
                anneal_strategy="cos",
                anneal_epochs=10,
            )
            self.swa_started_ = False
        if config.use_mixed_precision:
            logger.info("Using mixed precision training.")
            self.scaler = torch.cuda.amp.GradScaler()

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
        if self.eval_ds is None:
            return None
        return DataLoader(
            self.eval_ds,
            self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self.eval_collate_fn,
        )

    def get_test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self.eval_collate_fn,
        )

    def train(self):
        self.callback_handler.on_fit_start(self)
        for ep in range(self.config.max_epochs):
            self.epoch = ep
            self._run_epoch("train")
            if self.config.use_swa and self.swa_started_:
                # update BN statistics for the SWA model after training
                dataloader = self.get_train_dataloader()
                torch.optim.swa_utils.update_bn(dataloader, self.swa_model)
            if self.eval_ds is not None:
                self.validate()
            if self.early_stopping_active:
                if self.test_ds is not None:
                    logger.info("Calculating results on test set...")
                    self.test()
                break
        self.callback_handler.on_fit_end(self)

    @torch.no_grad()
    def validate(self):
        self._run_epoch("eval")

    @torch.no_grad()
    def test(self):
        self._run_epoch("test")

    def _run_epoch(self, split: str):
        if split == "train":
            self.callback_handler.on_train_epoch_start(self)
        elif split == "eval":
            self.callback_handler.on_validation_epoch_start(self)

        config, model, optimizer = self.config, self.model, self.optimizer
        self.split_ = split
        self.epoch_losses = []
        self.epoch_metrics = {}

        is_train = True if split == "train" else False

        if split == "train":
            dataloader = self.get_train_dataloader()
        elif split == "eval":
            dataloader = self.get_eval_dataloader()
        else:
            dataloader = self.get_test_dataloader()

        assert dataloader is not None, f"{split} dataloader not specified."

        criterion = self.loss_fn if self.loss_fn else nn.CrossEntropyLoss()

        pbar = tqdm(dataloader, total=len(dataloader)) if is_train else dataloader
        for data in pbar:
            model.train(is_train)  # put model in training or evaluation mode

            # put data on the appropriate device (cpu or gpu)
            imgs, *targets = [el.to(self.device) for el in data]
            if len(targets) == 1:
                targets = targets[0]

            with torch.cuda.amp.autocast(config.use_mixed_precision):  # mixed precision
                logits = model(imgs)
                loss = criterion(logits, targets)

            self.loss_, self.logits_, self.targets_ = loss, logits, targets  # for hooks

            self.epoch_losses.append(loss.item())
            self.losses[split].append(loss.item())

            self.callback_handler.on_evaluate(self)
            self.callback_handler.on_after_evaluate(self)

            if is_train:
                if config.use_mixed_precision:
                    # scale loss, to avoid underflow when using mixed precision
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                else:
                    loss.backward()
                self.callback_handler.on_after_backward(self)
                # clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_norm_clip
                )
                if config.use_mixed_precision:
                    self.scaler.step(optimizer)
                    self.scaler.update()  # updates the scale for next iteration
                else:
                    optimizer.step()  # update weights
                if (
                    config.use_swa
                    and self.epoch >= config.swa_start * config.max_epochs
                ):
                    if not self.swa_started_:
                        logger.info("Starting Stochastic Weight Averaging.")
                        self.swa_started_ = True
                    self.swa_model.update_parameters(model)
                    self.swa_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_loss = np.mean(self.epoch_losses)

        if split == "train":
            self.callback_handler.on_train_epoch_end(self)
        elif split == "eval":
            self.callback_handler.on_validation_epoch_end(self)

        info_str = f"epoch {self.epoch} - {split}_loss: {epoch_loss:.4f}. "
        if split == "eval":
            for metric_name in self.epoch_metrics.keys():
                info_str += f"{metric_name}: {self.epoch_metrics[metric_name]:.4f}. "
        logger.info(info_str)
