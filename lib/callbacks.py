import logging
from typing import List, Mapping, Union, Any, Sequence, Optional, Dict
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import numpy as np
import numpy.linalg as LA
import mlflow

logger = logging.getLogger(__name__)


class TrainerCallback(ABC):
    def on_evaluate(self, trainer: ".trainer.Trainer"):
        """Hook invoked after calculating the loss, before backpropagating."""
        pass

    def on_after_evaluate(self, trainer: ".trainer.Trainer"):
        """Hook invoked after on_evaluate (e.g. for logging the eval results)."""
        pass

    def on_train_epoch_start(self, trainer: ".trainer.Trainer"):
        """Hook invoked at the beginning of a training epoch."""
        pass

    def on_train_epoch_end(self, trainer: ".trainer.Trainer"):
        """Hook invoked at the end of a training epoch."""
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
        """Hook invoked after loss.backward() and before optimizers are stepped."""
        pass

    def on_fit_start(self, trainer: ".trainer.Trainer"):
        """Hook invoked as the first operation when the Trainer.train() function starts."""
        pass

    def on_fit_end(self, trainer: ".trainer.Trainer"):
        """Hook invoked after the Trainer.train() function has completed."""
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
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    @property
    def metric_callbacks(self):
        res = []
        for cb in self.callbacks:
            if isinstance(cb, MetricCallback):
                res.append(cb)
        return res

    def on_evaluate(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_evaluate", trainer)

    def on_after_evaluate(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_after_evaluate", trainer)

    def on_train_epoch_start(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_train_epoch_start", trainer)

    def on_train_epoch_end(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_train_epoch_end", trainer)

    def on_validation_epoch_end(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_validation_epoch_end", trainer)

    def on_after_backward(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_after_backward", trainer)

    def on_validation_epoch_start(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_validation_epoch_start", trainer)

    def on_fit_start(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_fit_start", trainer)

    def on_fit_end(self, trainer: ".trainer.Trainer"):
        self.call_hook("on_fit_end", trainer)

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
    def __init__(self, max_epochs_no_change=10, higher_is_better=True):
        """
        Monitor one or several metrics for improvement over time and stop early
        after no improvement has occured for a given number of epochs.

        Args:
            max_epochs_no_change (int): number of epochs that the model
                does not improve when early stopping occurs.
        """
        self.max_epochs_no_change = max_epochs_no_change
        self.epochs_no_change = 0

        # TODO: early stopping based on loss as default

    def on_validation_epoch_end(self, trainer: ".trainer.Trainer"):
        if trainer.callback_handler.metric_callbacks == []:
            logger.warn(
                "List of metric callbacks is empty. Early stopping cannot occur."
            )
        for cb in trainer.callback_handler.metric_callbacks:
            if cb.monitor:
                if cb.epoch_new_best:
                    self.epochs_no_change = 0
                else:
                    self.epochs_no_change += 1
        if self.epochs_no_change >= self.max_epochs_no_change:
            trainer.early_stopping_active = True
            logger.info(
                f"Stopped early at epoch {trainer.epoch}. Best scores: {trainer.best_scores}"
            )


class CheckpointCallback(TrainerCallback):
    def __init__(
        self, dir_path: Union[Path, str], monitor: Union[str, List[str]] = None
    ):
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
        if monitor is not None and not isinstance(monitor, list):
            monitor = [monitor]
        self.monitor = monitor
        self.dir_path = Path(dir_path)

        self.dir_path.mkdir(exist_ok=True, parents=True)
        self.cpt_files = []

    def _save_checkpoint(self, trainer: ".trainer.Trainer"):
        # TODO: add some sort of label encoder to the saved model (e.g. the one # by sklearn)
        logger.info("Saving checkpoint.")
        if trainer.config.model_name is not None:
            model_name = trainer.config.model_name
        else:
            model_name = type(trainer.model).__name__
        file_name = (
            f"{model_name}-epoch{trainer.epoch}-"
            + "-".join(
                f"{metric}={value:.4f}" for metric, value in trainer.best_scores.items()
            )
            + ".tar"
        )
        model = (
            trainer.swa_model
            if (trainer.config.use_swa and trainer.swa_started_)
            else trainer.model
        )
        cpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }

        cpt_file = Path(self.dir_path) / file_name
        torch.save(cpt, cpt_file)
        self.cpt_files.append(cpt_file)

    def on_validation_epoch_end(self, trainer: ".trainer.Trainer"):
        if self.monitor is None:
            return
        for cb in trainer.callback_handler.metric_callbacks:
            if cb.name in self.monitor:
                if cb.epoch_new_best:
                    model = (
                        trainer.swa_model
                        if (trainer.config.use_swa and trainer.swa_started_)
                        else trainer.model
                    )
                    trainer.best_state_dict = model.state_dict()
                    self._save_checkpoint(trainer)
                    return

    def on_fit_end(self, trainer: ".trainer.Trainer"):
        """
        If no metrics to monitor were specified, save a checkpoint at the end of
        training.
        """
        if self.monitor is not None:
            return
        model = (
            trainer.swa_model
            if (trainer.config.use_swa and trainer.swa_started_)
            else trainer.model
        )
        trainer.best_state_dict = model.state_dict()
        self._save_checkpoint(trainer)


class MLFlowCallback(TrainerCallback):
    """
    Logs all relevant parameters/metrics to a local MLFlow server.

    All logs are stored in the `mlflow/` directory.
    """

    def __init__(
        self,
        experiment_name: str = "default",
        port: int = 5000,
        parameters_to_log: Optional[Dict[str, Any]] = None,
        artifacts_to_log: Optional[Sequence[Union[str, Path]]] = None,
    ):
        """
        Args:
            experiment_name (str): experiment name that will be used in the MLFlow UI to
                group this run under.
            port (int): port that the tracking server URI will use.
            parameters_to_log: Optional[Dict[str, Any]]: dictionary of parameter:value
                pairs, which will be logged at the start of training (optional).
            artifacts_to_log (Optional[List[Union[str, Path]]]): list of files that
                should be saved as artifacts at the end of every training epoch (optional).
        """
        self.experiment_name = experiment_name
        self.port = port
        self.parameters_to_log = parameters_to_log
        self.artifacts_to_log = artifacts_to_log

        mlflow.set_experiment(experiment_name)

        logger.info("Logging to MLFlow.")

    def on_fit_start(self, trainer: ".trainer.Trainer"):
        mlflow.log_params(trainer.config.dump())
        mlflow.log_params(self.parameters_to_log)

    def on_after_evaluate(self, trainer: ".trainer.Trainer"):
        mlflow.log_metrics(trainer.epoch_metrics)
        for l in trainer.epoch_losses:
            mlflow.log_metric(f"{trainer.split_}_loss", l)

    def on_train_epoch_start(self, trainer: ".trainer.Trainer"):
        mlflow.log_metric("_epoch", trainer.epoch)

    def on_train_epoch_end(self, trainer: ".trainer.Trainer"):
        # save specified artifacts to mlflow
        if self.artifacts_to_log is None:
            return
        assert isinstance(self.artifacts_to_log, (list, tuple))
        for path in self.artifacts_to_log:
            mlflow.log_artifact(path)

    def on_fit_end(self, trainer: ".trainer.Trainer"):
        # save model checkpoints to mlflow
        for cb in trainer.callback_handler.callbacks:
            if isinstance(cb, CheckpointCallback):
                for cpt_path in cb.cpt_files:
                    mlflow.log_artifact(cpt_path, "checkpoints")
                break
