import random
import math
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Iterable, Sequence

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, Sampler


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes: int, confidence: float = 1.0, dim: int = -1):
        """
        Adds noise to one-hot encoded label vectors.

        Label smoothing prevents the pursuit of hard probabilities without
        discouraging correct classification. Degree of label smoothing is
        indicated by confidence. If confidence == 1, return regular
        cross-entropy loss. Otherwise, apply label smoothing before calculating
        cross-entropy loss.

        Args:
            n_classes (int): number of classes
            confidence (float): new value for the target class. Remaining `1 -
                confidence` is uniformly distributed over the remaining classes.
            dim (int): dimension to reduce when applying softmax
        """
        super().__init__()
        self.n_classes = n_classes
        self.confidence = confidence
        self.dim = dim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): logits to calculate the loss for
            target (torch.Tensor): target values
        Returns:
            Cross entropy loss with label smoothing applied
        """
        assert 0 < self.confidence <= 1
        if self.confidence == 1.0:
            return F.cross_entropy(input, target)
        log_probs = F.log_softmax(input, self.dim)
        target_hot = F.one_hot(target, self.n_classes)
        target_smooth = self.label_smoothing(
            target_hot, self.n_classes, 1 - self.confidence
        )
        return (target_smooth * -log_probs).sum(self.dim).mean()

    @staticmethod
    @torch.no_grad()
    def label_smoothing(
        labels: torch.Tensor, n_classes: int, epsilon: float = 0.1
    ) -> torch.Tensor:
        noise = epsilon / (n_classes - 1)
        return torch.where(labels == 1, 1 - epsilon, noise)


class MultiTaskLoss(nn.Module):
    """
    Calculates the loss for several tasks and creates a weighted sum out of
    them.
    """

    def __init__(
        self,
        loss_fns: Sequence[nn.Module],
        weights: Sequence[float] = None,
        ignore_index: int = None,
    ):
        """
        Args:
            loss_fns (sequence): loss functions for all tasks
            weights (sequence): weights per task for the weighted final loss
            ignore_index (int): target value to ignore (optional)
        """
        super().__init__()
        self.loss_fns = loss_fns
        self.weights = weights
        self.ignore_index = ignore_index

    def forward(
        self, inputs: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            inputs: sequence containing the outputs for each task
            targets: sequence containing the targets for each task
        Returns:
            Weighted loss combining losses for all tasks
        """
        assert len(inputs) == len(targets), f"{len(inputs)} vs {len(targets)}"
        losses = []
        for i, loss_fn in enumerate(self.loss_fns):
            input, target = inputs[i], targets[i]

            if self.ignore_index:
                mask = target != self.ignore_index
                input, target = input[mask], target[mask]

            if target.numel() != 0:
                losses.append(self.loss_fns[i](input, target))
        if self.weights is not None:
            assert len(self.weights) == len(self.loss_fns)
            losses = [w * l for w, l in zip(self.weights, losses)]
        return sum(losses)


class DoubleTaskSampler(Sampler):
    """
    Data sampler for multi-task learning.

    Samples indices for two different tasks, maintaining an even distribution of
    class examples for both tasks across batches. Set this sampler as the
    `batch_sampler` argument for the Pytorch dataloader.

    NOTE: this sampler is dataset-specific and should be adapted to the dataset
    at hand. It does not work out-of-the-box.
    """

    def __init__(self, dataset: Dataset, batch_size: int):
        """
        Args:
            dataset (Dataset): torch dataset
            batch_size (int): size of the batches
        """
        self.dataset = dataset
        self.batch_size = batch_size

        if isinstance(dataset, Subset):
            self.style_indices = [
                ix
                for ix in range(len(dataset))
                if dataset.dataset.img_ids[dataset.indices[ix]].parent.parent.name
                in STYLE_CLASSES
            ]
        else:
            self.style_indices = [
                ix
                for ix in range(len(dataset))
                if dataset.img_ids[ix].parent.parent.name in STYLE_CLASSES
            ]
        self.char_indices = [
            ix for ix in range(len(dataset)) if ix not in self.style_indices
        ]

        self.style_to_take = int(batch_size * len(self.style_indices) / len(dataset))
        self.char_to_take = batch_size - self.style_to_take

    def __iter__(self):
        random.shuffle(self.style_indices)
        random.shuffle(self.char_indices)

        for i in range(len(self)):  # no drop_last
            idx_list = self.style_indices[
                i * self.style_to_take : (i + 1) * self.style_to_take
            ]
            idx_list += self.char_indices[
                i * self.char_to_take : (i + 1) * self.char_to_take
            ]
            yield idx_list

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
