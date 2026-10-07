"""The few dgl.nn names ms-pred references (pure torch)."""
import torch
import torch.nn as _nn


def expand_as_pair(input_, g=None):
    if isinstance(input_, tuple):
        return input_
    return input_, input_


class AvgPooling(_nn.Module):
    """Mean of node features per member graph of a batch (0 for empty graphs)."""

    def forward(self, graph, feat):
        bnn = graph.batch_num_nodes()
        seg = torch.repeat_interleave(torch.arange(bnn.shape[0], device=feat.device), bnn)
        out = feat.new_zeros((bnn.shape[0],) + tuple(feat.shape[1:]))
        out.index_add_(0, seg, feat)
        shape = (-1,) + (1,) * (feat.dim() - 1)
        return out / bnn.clamp(min=1).to(feat.dtype).view(shape)


class SumPooling(_nn.Module):
    def forward(self, graph, feat):
        bnn = graph.batch_num_nodes()
        seg = torch.repeat_interleave(torch.arange(bnn.shape[0], device=feat.device), bnn)
        out = feat.new_zeros((bnn.shape[0],) + tuple(feat.shape[1:]))
        out.index_add_(0, seg, feat)
        return out


class _Unsupported(_nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            f"dgl shim: {type(self).__name__} is not implemented (not needed by the "
            "MassSpecGym ICEBERG / GLACIER checkpoints)"
        )


class GlobalAttentionPooling(_Unsupported):
    pass


class GraphConv(_Unsupported):
    pass
