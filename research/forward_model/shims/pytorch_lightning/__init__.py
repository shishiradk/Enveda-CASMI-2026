"""Inference-only stand-in for pytorch_lightning (pure torch, MIT, ours).

ms-pred model classes derive from ``pl.LightningModule`` but inference only
needs an ``nn.Module`` that can be built from the hyper-parameters stored in a
checkpoint.  Nothing related to training is provided.
"""
import pathlib

import torch
import torch.nn as nn

from . import loggers, utilities  # noqa: F401

__version__ = "0.0-inference-shim"


def _load_any(path, map_location="cpu"):
    """Load either a Lightning .ckpt or our stripped {'state_dict','hyper_parameters'} file."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        pass
    # Lightning checkpoints written on Linux pickle PosixPath objects, which
    # cannot be instantiated on Windows; map them to the pure flavour while loading.
    saved = pathlib.PosixPath
    try:
        if not isinstance(pathlib.Path(), pathlib.PosixPath):
            pathlib.PosixPath = pathlib.PurePosixPath
        return torch.load(path, map_location=map_location, weights_only=False)
    finally:
        pathlib.PosixPath = saved


class LightningModule(nn.Module):
    def save_hyperparameters(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def log_dict(self, *args, **kwargs):
        return None

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location="cpu", strict=True, **overrides):
        ckpt = _load_any(checkpoint_path, map_location)
        hparams = dict(ckpt.get("hyper_parameters", {}))
        hparams.update(overrides)
        model = cls(**hparams)
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        return model


class Trainer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("pytorch_lightning shim: training is not supported")


class Callback:
    pass


def seed_everything(seed=0, workers=False):
    torch.manual_seed(seed)
    return seed
