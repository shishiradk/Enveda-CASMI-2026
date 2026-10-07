"""Minimal pure-PyTorch stand-in for the subset of the DGL API that ms-pred
(ICEBERG 2.1 / GLACIER) inference touches.

Written from scratch for the Enveda CASMI 2026 project (MIT licence, same as
the rest of this repository).  It is NOT a general DGL replacement: only
homogeneous graphs, only the calls listed below, no sparse formats, no C++.

Semantics that the upstream code relies on and that are reproduced here:
  * a batched graph is one big graph plus per-member node / edge counts;
  * ``node_subgraph`` keeps the selected nodes in the given order, keeps the
    induced edges in their original edge-id order and drops batch information;
  * ``reorder_graph(..., 'custom', nodes_perm=p)`` makes new node ``i`` the old
    node ``p[i]`` and then sorts the edges by (new) source node;
  * ``update_all`` with the built-in ``copy_u / copy_e / u_mul_e`` messages and
    a ``sum`` reducer;
  * ``apply_edges`` with an optional edge-id subset (missing rows are zero).
"""
from contextlib import contextmanager

import torch

from . import function  # noqa: F401
from . import backend  # noqa: F401

__version__ = "0.0-puretorch-shim"

NID = "_ID"
EID = "_ID"


def _as_long(x, device=None):
    if torch.is_tensor(x):
        return x.long() if device is None else x.long().to(device)
    return torch.as_tensor(x, dtype=torch.long, device=device)


class _EdgeBatch:
    """What a user-defined edge function receives in ``apply_edges``."""

    class _Lazy:
        def __init__(self, store, idx):
            self._store, self._idx = store, idx

        def __getitem__(self, key):
            val = self._store[key]
            return val if self._idx is None else val[self._idx]

        def __contains__(self, key):
            return key in self._store

    def __init__(self, g, eids):
        src = g._src if eids is None else g._src[eids]
        dst = g._dst if eids is None else g._dst[eids]
        self.src = self._Lazy(g.ndata, src)
        self.dst = self._Lazy(g.ndata, dst)
        self.data = self._Lazy(g.edata, eids)
        self._n = int(src.shape[0])

    def __len__(self):
        return self._n


class DGLGraph:
    is_homogeneous = True
    idtype = torch.int64

    def __init__(self, src, dst, num_nodes):
        self._src = src
        self._dst = dst
        self._n = int(num_nodes)
        self.ndata = {}
        self.edata = {}
        self._bnn = None
        self._bne = None

    # ---- sizes -----------------------------------------------------------
    @property
    def device(self):
        return self._src.device

    def num_nodes(self):
        return self._n

    number_of_nodes = num_nodes

    def num_edges(self):
        return int(self._src.shape[0])

    number_of_edges = num_edges

    def edges(self, order="eid", form="uv"):
        return self._src, self._dst

    @property
    def batch_size(self):
        return 1 if self._bnn is None else int(self._bnn.shape[0])

    def batch_num_nodes(self):
        if self._bnn is None:
            return torch.tensor([self._n], dtype=torch.long, device=self.device)
        return self._bnn

    def batch_num_edges(self):
        if self._bne is None:
            return torch.tensor([self.num_edges()], dtype=torch.long, device=self.device)
        return self._bne

    def set_batch_num_nodes(self, val):
        self._bnn = _as_long(val, self.device)

    def set_batch_num_edges(self, val):
        self._bne = _as_long(val, self.device)

    # ---- data frames -----------------------------------------------------
    @property
    def srcdata(self):
        return self.ndata

    @property
    def dstdata(self):
        return self.ndata

    @contextmanager
    def local_scope(self):
        old_n, old_e = self.ndata, self.edata
        self.ndata, self.edata = dict(old_n), dict(old_e)
        try:
            yield
        finally:
            self.ndata, self.edata = old_n, old_e

    def local_var(self):
        g = DGLGraph(self._src, self._dst, self._n)
        g.ndata, g.edata = dict(self.ndata), dict(self.edata)
        g._bnn, g._bne = self._bnn, self._bne
        return g

    def to(self, device):
        g = DGLGraph(self._src.to(device), self._dst.to(device), self._n)
        g.ndata = {k: v.to(device) for k, v in self.ndata.items()}
        g.edata = {k: v.to(device) for k, v in self.edata.items()}
        g._bnn = None if self._bnn is None else self._bnn.to(device)
        g._bne = None if self._bne is None else self._bne.to(device)
        return g

    def in_degrees(self):
        return torch.bincount(self._dst, minlength=self._n)

    def out_degrees(self):
        return torch.bincount(self._src, minlength=self._n)

    # ---- message passing -------------------------------------------------
    def apply_edges(self, func, edges=None):
        eids = None if edges is None else _as_long(edges, self.device)
        out = func(_EdgeBatch(self, eids))
        for key, val in out.items():
            if eids is None:
                self.edata[key] = val
            else:
                cur = self.edata.get(key)
                if cur is None or cur.shape[1:] != val.shape[1:] or cur.dtype != val.dtype:
                    cur = val.new_zeros((self.num_edges(),) + tuple(val.shape[1:]))
                else:
                    cur = cur.clone()
                cur[eids] = val
                self.edata[key] = cur

    def update_all(self, message_func, reduce_func, apply_node_func=None):
        if not isinstance(message_func, function._Message) or not isinstance(
            reduce_func, function._Reduce
        ):
            raise NotImplementedError(
                "dgl shim: update_all only supports dgl.function built-ins"
            )
        msg = message_func(self)
        self.ndata[reduce_func.out] = reduce_func(self, msg)
        if apply_node_func is not None:
            raise NotImplementedError("dgl shim: apply_node_func")

    # ---- structure -------------------------------------------------------
    def subgraph(self, nodes, **kwargs):
        return node_subgraph(self, nodes)


def graph(data, num_nodes=None, idtype=None, device=None, **kwargs):
    src, dst = data
    src = _as_long(src, device)
    dst = _as_long(dst, device)
    if num_nodes is None:
        num_nodes = int(max(src.max().item(), dst.max().item()) + 1) if src.numel() else 0
    return DGLGraph(src, dst, num_nodes)


def batch(graphs, ndata=None, edata=None):
    graphs = list(graphs)
    if len(graphs) == 0:
        raise ValueError("dgl shim: cannot batch an empty list")
    device = graphs[0].device
    sizes = torch.tensor([g.num_nodes() for g in graphs], dtype=torch.long, device=device)
    offsets = torch.cumsum(sizes, 0) - sizes
    src = torch.cat([g._src + o for g, o in zip(graphs, offsets)])
    dst = torch.cat([g._dst + o for g, o in zip(graphs, offsets)])
    out = DGLGraph(src, dst, int(sizes.sum().item()))
    for key in graphs[0].ndata:
        out.ndata[key] = torch.cat([g.ndata[key] for g in graphs], 0)
    for key in graphs[0].edata:
        out.edata[key] = torch.cat([g.edata[key] for g in graphs], 0)
    out._bnn = torch.cat([g.batch_num_nodes() for g in graphs])
    out._bne = torch.cat([g.batch_num_edges() for g in graphs])
    return out


def unbatch(g):
    bnn, bne = g.batch_num_nodes().tolist(), g.batch_num_edges().tolist()
    out, n0, e0 = [], 0, 0
    for n, e in zip(bnn, bne):
        sub = DGLGraph(g._src[e0:e0 + e] - n0, g._dst[e0:e0 + e] - n0, n)
        sub.ndata = {k: v[n0:n0 + n] for k, v in g.ndata.items()}
        sub.edata = {k: v[e0:e0 + e] for k, v in g.edata.items()}
        out.append(sub)
        n0, e0 = n0 + n, e0 + e
    return out


def node_subgraph(g, nodes, store_ids=True, **kwargs):
    nodes = torch.as_tensor(nodes, device=g.device) if not torch.is_tensor(nodes) else nodes
    if nodes.dtype == torch.bool:
        idx = torch.nonzero(nodes, as_tuple=True)[0]
    else:
        idx = nodes.long()
    new_id = torch.full((g.num_nodes(),), -1, dtype=torch.long, device=g.device)
    new_id[idx] = torch.arange(idx.numel(), device=g.device)
    s, d = new_id[g._src], new_id[g._dst]
    keep = torch.nonzero((s >= 0) & (d >= 0), as_tuple=True)[0]
    out = DGLGraph(s[keep], d[keep], idx.numel())
    out.ndata = {k: v[idx] for k, v in g.ndata.items() if k != NID}
    out.edata = {k: v[keep] for k, v in g.edata.items() if k != EID}
    if store_ids:
        out.ndata[NID] = idx
        out.edata[EID] = keep
    return out


def reorder_graph(g, node_permute_algo=None, edge_permute_algo="src",
                  store_ids=True, permute_config=None):
    if node_permute_algo != "custom":
        raise NotImplementedError("dgl shim: only node_permute_algo='custom'")
    perm = _as_long(permute_config["nodes_perm"], g.device)
    out = node_subgraph(g, perm, store_ids=False)
    key = out._src if edge_permute_algo == "src" else out._dst
    order = torch.sort(key, stable=True)[1]
    out._src, out._dst = out._src[order], out._dst[order]
    out.edata = {k: v[order] for k, v in out.edata.items()}
    if store_ids:
        out.ndata[NID] = perm
        out.edata[EID] = order
    return out


def _segment_ids(g):
    bnn = g.batch_num_nodes()
    return torch.repeat_interleave(torch.arange(bnn.shape[0], device=g.device), bnn), bnn


def _segment_sum(g, feat):
    seg, bnn = _segment_ids(g)
    out = feat.new_zeros((bnn.shape[0],) + tuple(feat.shape[1:]))
    out.index_add_(0, seg, feat)
    return out, bnn


def sum_nodes(g, feat, weight=None):
    return _segment_sum(g, g.ndata[feat])[0]


def mean_nodes(g, feat, weight=None):
    tot, bnn = _segment_sum(g, g.ndata[feat])
    shape = (-1,) + (1,) * (tot.dim() - 1)
    return tot / bnn.clamp(min=1).to(tot.dtype).view(shape)


from . import nn  # noqa: E402,F401
