"""
Early stopping callback for CISSN training loops.

Monitors validation loss and saves the best checkpoint whenever it improves.
"""
import os
from typing import Optional

import numpy as np
import torch


class EarlyStopping:
    """Stop training when validation loss has not improved for `patience` epochs.

    Saves encoder + head checkpoints whenever validation loss reaches a new minimum.
    """

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait after the last improvement.
            verbose: If True, print a message each time a checkpoint is saved.
            delta: Minimum change in monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: torch.nn.Module, head: torch.nn.Module, path: str) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, head, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, head, path)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: torch.nn.Module, head: torch.nn.Module, path: str) -> None:
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        torch.save(head.state_dict(), os.path.join(path, "checkpoint_head.pth"))
        self.val_loss_min = val_loss
