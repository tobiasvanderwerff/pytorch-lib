import logging
from typing import List, Mapping, Union
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import numpy as np
import numpy.linalg as LA

logger = logging.getLogger(__name__)


class TrainerCallback(ABC):
    def on_evaluate(self, trainer: ".trainer.Trainer", logits: torch.Tensor,
                    targets: torch.Tensor) -> Mapping[str, float]:
        """ 
        This hook should return a dictionary {metric: value}, containing a
        metric name and calculated value.
        """
        pass

    def on_train_epoch_start(self, trainer: ".trainer.Trainer"):
        """ Hook invoked at the beginning of a training epoch. """
        pass

    def on_train_epoch_end(self, trainer: ".trainer.Trainer"):
        """ Hook invoked at the end of a training epoch. """
        pass

    def on_validation_epoch_start(self, trainer: ".trainer.Trainer"):
        """ 
        Hook invoked at the beginning of a val epoch. 
        """
        pass

    def on_validation_epoch_end(self, trainer: ".trainer.Trainer"):
        """ 
        Hook invoked at the end of a val epoch. For example, can be used to
        show intermediate predictions of the model.
        """
        pass

    def on_after_backward(self, trainer: ".trainer.Trainer"):
        """ Hook invoked after loss.backward() and before optimizers are stepped. """
        pass

    def on_fit_end(self, trainer: ".trainer.Trainer"):
        """ Hook invoked after the Trainer.train() function has completed. """
        pass


class MetricCallback(TrainerCallback):
    def __init__(self, name, higher_is_better):
        self.name = name
        self.higher_is_better = higher_is_better
        self.scores = []
        self.epoch_new_best = False

    def _better_than(self, val1: Union[float, int], val2: Union[float, int]):
        if self.higher_is_better:
            return val1 > val2
        else:  # lower is better
            return val1 < val2

    def check_for_new_best(self):
        score = self.scores[-1]
        if self._better_than(score, self.best):
            self.best = score
            self.epoch_new_best = True
            logger.info(f"New best score - {self.name}: {score:.4f}")
        else:
            self.epoch_new_best = False


class CallbackHandler:
    def __init__(self, callbacks: List[TrainerCallback]):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        self.callbacks.append(cb)

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return '\n'.join(cb.__class__.__name__ for cb in self.callbacks)

    @property
    def metric_callbacks(self):
        res = []
        for cb in isinstance(self.callbacks, MetricCallback):
            res.append(cb)
        return res

    def on_evaluate(self, predictions, targets):
        metrics = self.call_hook('on_evaluate', predictions, targets)
        return metrics

    def on_train_epoch_start(self):
        self.call_hook('on_train_epoch_start') 

    def on_train_epoch_end(self):
        self.call_hook('on_train_epoch_end') 

    def on_validation_epoch_end(self):
        self.call_hook('on_validation_epoch_end') 

    def on_after_backward(self):
        self.call_hook('on_after_backward') 

    def on_validation_epoch_start(self):
        self.call_hook('on_validation_epoch_start') 

    def on_fit_end(self):
        self.call_hook('on_fit_end') 

    def call_hook(self, hook_name, *args):
        output = {}
        for cb in self.callbacks:
            hook = getattr(cb, hook_name)
            res = hook(*args)
            if res is not None:
                output.update(res)
        return output


class GradientNormTrackingCallback(TrainerCallback):
    """ 
    Calculates the 2-norm of the gradients in the model and displays the
    average for each epoch. 
    """

    def __init__(self):
        grad_norms = []

    def on_after_backward(self, trainer: ".trainer.Trainer"):
        grad_norm = 0
        for p in trainer.model.parameters():
            if p.requires_grad:
                grad_norm += LA.norm(p.grad.detach().cpu().numpy()) ** 2
        self.grad_norms.append(np.sqrt(grad_norm))

    def on_train_epoch_end(self, trainer: ".trainer.Trainer"):
        avg_grad_norm = np.mean(grad_norms)
        logger.info(f"Average gradient L2 norm this epoch: {avg_grad_norm}")
        self.grad_norms = []


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, max_epochs_no_change=10,
                 higher_is_better=True):
        """
        Monitor one or several metrics for improvement over time and stop early
        after no improvement has occured for a given number of epochs.

        Args:
            max_epochs_no_change (int): number of epochs that the model
                does not improve when early stopping occurs.
        """
        self.max_epochs_no_change = max_epochs_no_change
        self.epochs_no_change = 0

    def on_validation_epoch_end(self, trainer: ".trainer.Trainer"):
        for cb in trainer.metric_callbacks:
            if cb.monitor:
                if cb.epoch_new_best: self.epochs_no_change = 0
                else: self.epochs_no_change += 1
        if self.epochs_no_change >= self.max_epochs_no_change:
            trainer.early_stopping_active = True


class CheckpointCallback(TrainerCallback):
    def __init__(self, dir_path: Union[Path, str],
                 monitor: Union[str, List[str]] = None):
        """
        Monitors given metrics and saves checkpoints each epoch when one of the
        metrics improves.

        If no metrics are specified to monitor, one checkpoint will be saved at
        the end of training.

        Args:
            dir_path (Union[Path, str]): directory to save the checkpoints
            monitor (Union[str, List[str]]): names of the metrics to monitor
                (optional)
        """
        if not isinstance(monitor, list):
            monitor = [monitor]
        self.monitor = monitor
        self.dir_path = Path(dir_path)

        self.dir_path.mkdir(exist_ok=True, parents=True)

    def _save_checkpoint(self, trainer: ".trainer.Trainer"):
        if trainer.config.model_name is not None:
            model_name = trainer.config.model_name
        else:
            model_name = type(self.model).__name__
        file_name = (
            f'{model_name}-epoch{trainer.epoch}-' +
            '-'.join(f'{metric}={value:.4f}' for metric, value in trainer.best_scores.items()) +
            '.tar'
            )
        
        cpt = {'model_state_dict': trainer.model.state_dict(), 
               'optimizer_state_dict': trainer.optimizer.state_dict()}
        cpt.update(**kwargs)
        torch.save(cpt, Path(self.dir_path) / file_name)
        
    def on_validation_epoch_end(self, trainer: ".trainer.Trainer"):
        if self.monitor is None: return
        for cb in trainer.metric_callbacks:
            if cb.name in self.monitor:
                if cb.epoch_new_best:
                    logger.info("Saving checkpoint.")
                    trainer.best_state_dict = trainer.model.state_dict()
                    self._save_checkpoint(trainer)
                    break

    def on_fit_end(self, trainer: ".trainer.Trainer"):
        if self.monitor is not None: return
        logger.info("Saving checkpoint.")
        trainer.best_state_dict = trainer.model.state_dict()
        self._save_checkpoint(trainer)
