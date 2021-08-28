import torch
import logging

from .callbacks import TrainerCallback, MetricCallback
from .trainer import Trainer

from sklearn.metrics import confusion_matrix, classification_report, f1_score


logger = logging.getLogger(__name__)

# TODO: use a data structure that keeps track of predictions and targets, which
# can then be used for other callbacks (accuracy, F1, etc.)


class F1ScoreCallback(MetricCallback):
    """Calculates macro F1 score at every epoch end."""

    def __init__(self, monitor=False, ignore_index=-100):
        super().__init__("F1_macro", higher_is_better=True)
        self.ignore_index = -100
        self.monitor = monitor
        self.predictions = []
        self.targets = []
        self.best = float("-inf")

    def on_evaluate(
        self, trainer: Trainer, logits: torch.Tensor, targets: torch.Tensor
    ):
        _, preds = logits.max(-1)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        indices = [ix for ix, t in enumerate(targets) if t != ignore_index]
        self.preds += [pr for ix, pr in enumerate(preds) if ix in indices]
        self.targets += [t for ix, t in enumerate(targets) if ix in indices]

    def on_validation_epoch_start(self, trainer: Trainer):
        self.predictions = []
        self.targets = []

    def on_validation_epoch_end(self, trainer: Trainer):
        score = f1_score(self.targets, self.predictions, average="macro")
        self.scores.append(score)
        self.check_for_new_best()
        trainer.epoch_metrics.update({self.name: score})


class ClassificationReportCallback(TrainerCallback):
    """
    Shows the Sklearn classification report at the end of each epoch.
    """

    def __init__(self):
        self.preds = []
        self.targets = []

    def on_evaluate(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index=-100
    ):
        """
        Args:
            logits (torch.Tensor): 2-dimensional tensor
            targets (torch.Tensor): 1-dimensional tensor
            ignore_index (int): target value to ignore
        """
        _, preds = logits.max(-1)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        indices = [ix for ix, t in enumerate(targets) if t != ignore_index]
        self.preds += [pr for ix, pr in enumerate(preds) if ix in indices]
        self.targets += [t for ix, t in enumerate(targets) if ix in indices]

        return {}  # empty dictionary because this is not a proper metric

    def on_train_epoch_end(self, trainer: Trainer):
        # display the classification report
        report = classification_report(
            self.targets, self.preds, target_names=STYLE_CLASSES
        )
        logger.info("Classification report:\n" + str(report))
        self.preds = []
        self.targets = []


class ConfusionMatrixCallback(TrainerCallback):
    """
    Calculates confusion matrix over running statistics.
    """

    def __init__(self):
        self.predictions = []
        self.targets = []

    def on_evaluate(
        self,
        trainer: Trainer,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index=-100,
    ):
        """
        Args:
            logits (torch.Tensor): 2-dimensional tensor
            targets (torch.Tensor): 1-dimensional tensor
            ignore_index (int): target value to ignore
        """
        _, preds = logits.max(-1)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        indices = [ix for ix, target in enumerate(targets) if target != ignore_index]
        self.predictions += [pred for ix, pred in enumerate(preds) if ix in indices]
        self.targets += [target for ix, target in enumerate(targets) if ix in indices]

        return {}  # empty dictionary because this is not a proper metric

    def on_train_epoch_end(self, trainer: Trainer):
        # display the confusion matrix at the end of the epoch
        mat = confusion_matrix(self.targets, self.predictions)
        logger.info("Confusion matrix (Archaic, Hasmonean, Herodian):\n" + str(mat))
        self.predictions = []
        self.targets = []


class AccuracyCallback(MetricCallback):
    def __init__(self, monitor=False, ignore_index=-100):
        super().__init__("accuracy", higher_is_better=True)
        self.ignore_index = ignore_index
        self.n_correct = 0
        self.n_samples = 0
        self.monitor = monitor
        self.best = float("-inf")

    def on_evaluate(self, trainer: Trainer, logits, targets):
        _, preds = logits.max(-1)
        self.n_correct += (
            torch.logical_and(targets != self.ignore_index, preds == targets)
            .sum()
            .item()
        )
        self.n_samples += (targets != self.ignore_index).sum().item()

    def on_validation_epoch_start(self, trainer: Trainer):
        self.n_correct = 0
        self.n_samples = 0

    def on_validation_epoch_end(self, trainer: Trainer):
        if self.n_samples == 0:
            return {self.name: 0}

        score = self.n_correct / self.n_samples
        self.scores.append(score)
        self.check_for_new_best()
        if self.epoch_new_best:
            trainer.best_scores[self.name] = score

        trainer.epoch_metrics.update({self.name: score})
