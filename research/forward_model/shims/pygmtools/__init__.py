"""Placeholder: ms-pred imports pygmtools for training-time matching losses only."""


def _unavailable(*args, **kwargs):
    raise NotImplementedError("pygmtools shim: matching solvers are training-only and not provided")


hungarian = sinkhorn = _unavailable
