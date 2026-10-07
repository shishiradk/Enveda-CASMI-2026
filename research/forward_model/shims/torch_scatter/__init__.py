"""Pure-PyTorch versions of the torch_scatter functions ms-pred calls.

Written for the Enveda CASMI 2026 project (MIT).  ``index`` must have the same
shape as ``src`` (that is how ms-pred calls them).  Groups that receive no
element are 0, as in torch_scatter.  The arg-index outputs of scatter_min /
scatter_max are not computed (ms-pred discards them) and are returned as None.
"""
import torch

__version__ = "0.0-puretorch-shim"


def _size(index, dim, dim_size):
    if dim_size is not None:
        return int(dim_size)
    return int(index.max().item()) + 1 if index.numel() else 0


def _zeros(src, index, dim, dim_size):
    shape = list(src.shape)
    shape[dim] = _size(index, dim, dim_size)
    return src.new_zeros(shape)


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    if out is None:
        out = _zeros(src, index, dim, dim_size)
    return out.scatter_add_(dim, index, src)


scatter_sum = scatter_add


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    tot = scatter_add(src, index, dim, None, dim_size)
    cnt = scatter_add(torch.ones_like(src), index, dim, None, dim_size).clamp(min=1)
    return tot / cnt


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    res = _zeros(src, index, dim, dim_size)
    res = res.scatter_reduce(dim, index, src, reduce="amin", include_self=False)
    return res, None


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    res = _zeros(src, index, dim, dim_size)
    res = res.scatter_reduce(dim, index, src, reduce="amax", include_self=False)
    return res, None


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    if reduce == "min":
        return scatter_min(src, index, dim, out, dim_size)[0]
    if reduce == "max":
        return scatter_max(src, index, dim, out, dim_size)[0]
    raise NotImplementedError(reduce)


def scatter_softmax(src, index, dim=-1, eps=1e-12, dim_size=None):
    if not torch.is_floating_point(src):
        raise ValueError("scatter_softmax needs floating point input")
    gmax = scatter_max(src, index, dim, None, dim_size)[0]
    rec = (src - gmax.gather(dim, index)).exp()
    norm = scatter_add(rec, index, dim, None, dim_size).gather(dim, index)
    return rec / (norm + eps)
