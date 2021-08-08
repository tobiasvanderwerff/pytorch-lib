import random
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(cp_path: Union[Path, str], model: nn.Module, 
               device: str, optimizer: optim.Optimizer = None) -> None:
    """ Load a model from a checkpoint. 

    Note: checkpoint should be saved as a mapping from keys to values.
    
    Args:
        cp_path (Path or str): path where the checkpoint is saved
        model (nn.Module): model to load the checkpoint into
        device (str): device to load the parameters into
        optimizer (optim.Optimizer): optimizer to initalize from the 
            checkpoint (optional)
    """
    cp = torch.load(cp_path, map_location=torch.device(device))
    model.load_state_dict(cp['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(cp['optimizer_state_dict'])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
