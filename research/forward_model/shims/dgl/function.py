"""Built-in message / reduce descriptors used by ms-pred (pure torch)."""
import torch


class _Message:
    out = "m"

    def __call__(self, g):
        raise NotImplementedError


class _Reduce:
    pass


def _match(a, b):
    """DGL broadcasting: pad the lower-rank operand with trailing axes."""
    while a.dim() < b.dim():
        a = a.unsqueeze(-1)
    while b.dim() < a.dim():
        b = b.unsqueeze(-1)
    return a, b


class copy_u(_Message):
    def __init__(self, u, out):
        self.u, self.out = u, out

    def __call__(self, g):
        return g.ndata[self.u][g._src]


copy_src = copy_u


class copy_e(_Message):
    def __init__(self, e, out):
        self.e, self.out = e, out

    def __call__(self, g):
        return g.edata[self.e]


class u_mul_e(_Message):
    def __init__(self, u, e, out):
        self.u, self.e, self.out = u, e, out

    def __call__(self, g):
        a, b = _match(g.ndata[self.u][g._src], g.edata[self.e])
        return a * b


class sum(_Reduce):  # noqa: A001 - mirrors dgl.function.sum
    def __init__(self, msg, out):
        self.msg, self.out = msg, out

    def __call__(self, g, m):
        res = m.new_zeros((g.num_nodes(),) + tuple(m.shape[1:]))
        res.index_add_(0, g._dst, m)
        return res
