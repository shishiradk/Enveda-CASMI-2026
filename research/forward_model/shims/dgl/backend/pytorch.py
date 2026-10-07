"""pad / pack helpers with the dgl.backend.pytorch signatures (pure torch)."""
import torch


def _lengths(lengths, device):
    if torch.is_tensor(lengths):
        return lengths.long().to(device)
    return torch.as_tensor(lengths, dtype=torch.long, device=device)


def pad_packed_tensor(input, lengths, value, l_min=None):
    lengths = _lengths(lengths, input.device)
    max_len = int(lengths.max().item())
    if l_min is not None:
        max_len = max(max_len, l_min)
    mask = torch.arange(max_len, device=input.device)[None, :] < lengths[:, None]
    out = input.new_full((lengths.shape[0], max_len) + tuple(input.shape[1:]), value)
    out[mask] = input
    return out


def pack_padded_tensor(input, lengths):
    lengths = _lengths(lengths, input.device)
    mask = torch.arange(input.shape[1], device=input.device)[None, :] < lengths[:, None]
    return input[mask]
