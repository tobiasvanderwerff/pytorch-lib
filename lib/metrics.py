import torch
import logging

from .callbacks import TrainerCallback, MetricCallback
from .trainer import Trainer

from sklearn.metrics import confusion_matrix, classification_report, f1_score


logger = logging.getLogger(__name__)

# TODO: use a data structure that keeps track of predictions and targets, which
# can then be used for other callbacks (accuracy, F1, etc.)


class F1ScoreCallback(MetricCallback):
    """Calculates macro F1 score metric."""

    def __init__(self, monitor=False, ignore_index=-100):
        super().__init__("F1_macro", higher_is_better=True)
        self.ignore_index = ignore_index
        self.monitor = monitor
        self.predictions = []
        self.targets = []
        self.best = float("-inf")

    def _f1_score(self):
        if self.predictions == [] or self.targets == []:
            return 0
        return f1_score(self.targets, self.predictions, average="macro")

    def on_evaluate(self, trainer: Trainer):
        logits, targets = trainer.logits_, trainer.targets_
        _, preds = logits.max(-1)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        indices = [ix for ix, t in enumerate(targets) if t != self.ignore_index]
        self.preds += [pr for ix, pr in enumerate(preds) if ix in indices]
        self.targets += [t for ix, t in enumerate(targets) if ix in indices]
        trainer.epoch_metrics.update({self.name: self._f1_score()})

    def on_validation_epoch_start(self, trainer: Trainer):
        self.predictions = []
        self.targets = []

    def on_validation_epoch_end(self, trainer: Trainer):
        score = self._f1_score()
        self.scores.append(score)
        self.check_for_new_best()


class ClassificationReportCallback(TrainerCallback):
    """
    Shows the Sklearn classification report at the end of each epoch.
    """

    def __init__(self, ignore_index=-100):
        self.preds = []
        self.targets = []
        self.ignore_index = ignore_index

    def on_evaluate(self, trainer: Trainer):
        logits, targets = trainer.logits_, trainer.targets_
        _, preds = logits.max(-1)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        indices = [ix for ix, t in enumerate(targets) if t != self.ignore_index]
        self.preds += [pr for ix, pr in enumerate(preds) if ix in indices]
        self.targets += [t for ix, t in enumerate(targets) if ix in indices]

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

    def __init__(self, ignore_index=-100):
        self.predictions = []
        self.targets = []
        self.ignore_index = ignore_index

    def on_evaluate(self, trainer: Trainer):
        logits, targets = trainer.logits_, trainer.targets_
        _, preds = logits.max(-1)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        indices = [
            ix for ix, target in enumerate(targets) if target != self.ignore_index
        ]
        self.predictions += [pred for ix, pred in enumerate(preds) if ix in indices]
        self.targets += [target for ix, target in enumerate(targets) if ix in indices]

    def on_train_epoch_end(self, trainer: Trainer):
        # display the confusion matrix at the end of the epoch
        mat = confusion_matrix(self.targets, self.predictions)
        logger.info("Confusion matrix (Archaic, Hasmonean, Herodian):\n" + str(mat))
        self.predictions = []
        self.targets = []


class AccuracyCallback(MetricCallback):
    def __init__(self, monitor=False, ignore_index=-100):
        super().__init__("accuracy", higher_is_better=True)
        self.n_correct = 0
        self.n_samples = 0
        self.monitor = monitor
        self.best = float("-inf")
        self.ignore_index = ignore_index

    def _accuracy(self):
        if self.n_samples == 0:
            return 0
        return self.n_correct / self.n_samples

    def on_evaluate(self, trainer: Trainer):
        logits, targets = trainer.logits_, trainer.targets_
        _, preds = logits.max(-1)
        self.n_correct += (
            torch.logical_and(targets != self.ignore_index, preds == targets)
            .sum()
            .item()
        )
        self.n_samples += (targets != self.ignore_index).sum().item()
        trainer.epoch_metrics.update({self.name: self._accuracy()})

    def on_validation_epoch_start(self, trainer: Trainer):
        self.n_correct = 0
        self.n_samples = 0

    def on_validation_epoch_end(self, trainer: Trainer):
        score = self._accuracy()
        self.scores.append(score)
        self.check_for_new_best()
        if self.epoch_new_best:
            trainer.best_scores[self.name] = score
