"""Retrieval evaluation metrics: Recall@k and MRR@25, molecule-level (InChIKey14)."""

from typing import Sequence


def rank_of(ranked_ids: Sequence[str], true_id: str) -> int | None:
    """1-indexed rank of true_id in ranked_ids, or None if absent."""
    try:
        return ranked_ids.index(true_id) + 1
    except ValueError:
        return None


def recall_at_k(ranked_ids: Sequence[str], true_id: str, k: int) -> bool:
    rank = rank_of(ranked_ids[:k], true_id)
    return rank is not None


def reciprocal_rank(ranked_ids: Sequence[str], true_id: str, cutoff: int = 25) -> float:
    rank = rank_of(ranked_ids[:cutoff], true_id)
    return 0.0 if rank is None else 1.0 / rank


def summarize(per_query_ranked_ids, per_query_true_id, ks=(1, 5, 10, 25)):
    """per_query_ranked_ids: list of ranked-id lists (already truncated/sorted best-first).
    per_query_true_id: list of true ids, same length.
    Returns a dict of Recall@k for each k in ks, plus MRR@25 and n.
    """
    n = len(per_query_true_id)
    recalls = {k: 0 for k in ks}
    rr_sum = 0.0
    for ranked_ids, true_id in zip(per_query_ranked_ids, per_query_true_id):
        for k in ks:
            if recall_at_k(ranked_ids, true_id, k):
                recalls[k] += 1
        rr_sum += reciprocal_rank(ranked_ids, true_id, cutoff=max(ks))
    return {
        "n": n,
        **{f"recall@{k}": recalls[k] / n if n else 0.0 for k in ks},
        "mrr@25": rr_sum / n if n else 0.0,
    }
