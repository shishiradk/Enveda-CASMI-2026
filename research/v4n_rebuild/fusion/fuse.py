"""RRF fusion of v4n and E7 rank lists.

API:
  fuse(v4n_lists, e7_lists, key_fn=None, keep_top=40)
    v4n_lists: {mol_id: [(smiles, key), ...]}  (rank-ordered)
    e7_lists:  {mol_id: [(smiles, key), ...]}  (rank-ordered)
    key_fn: optional callable (smiles) -> canonical_key for deduplication
    keep_top: return this many top candidates (default 40 for forward-model re-rank)

    Returns: {mol_id: [(smiles, score), ...]} (rank-ordered by RRF score)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


def fuse(
    v4n_lists: Dict[str, List[Tuple[str, str]]],
    e7_lists: Dict[str, List[Tuple[str, str]]],
    key_fn: Optional[Callable[[str], str]] = None,
    keep_top: int = 40,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Reciprocal-rank fusion of two ranked lists.

    RRF formula (from fork notebook):
      score = 1/(3 + r_v4n) + 0.6/(3 + r_e7)

    Missing ranks contribute 0 to their term.

    Args:
      v4n_lists: {mol_id: [(smiles, score_key), ...]}
      e7_lists:  {mol_id: [(smiles, score_key), ...]}
      key_fn: optional function to canonicalize keys for deduplication
      keep_top: number of candidates to return per molecule

    Returns:
      {mol_id: [(smiles, rrf_score), ...]} sorted by score (descending)
    """

    results = {}

    for mol_id in v4n_lists.keys():
        v4n_candidates = v4n_lists.get(mol_id, [])
        e7_candidates = e7_lists.get(mol_id, [])

        # Build rank indices by score key
        v4n_ranks = {(key_fn(key) if key_fn else key): rank
                     for rank, (_, key) in enumerate(v4n_candidates)}
        e7_ranks = {(key_fn(key) if key_fn else key): rank
                    for rank, (_, key) in enumerate(e7_candidates)}

        # Collect all unique keys across both lists
        all_keys = set(v4n_ranks.keys()) | set(e7_ranks.keys())

        # Compute RRF scores
        rrf_scores = {}
        for key in all_keys:
            score = 0.0
            if key in v4n_ranks:
                score += 1.0 / (3 + v4n_ranks[key])
            if key in e7_ranks:
                score += 0.6 / (3 + e7_ranks[key])
            rrf_scores[key] = score

        # Build output: keep SMILES with highest RRF score per key
        key_to_smiles = {}
        for smiles, key in v4n_candidates:
            canonical_key = key_fn(key) if key_fn else key
            if canonical_key not in key_to_smiles:
                key_to_smiles[canonical_key] = smiles
        for smiles, key in e7_candidates:
            canonical_key = key_fn(key) if key_fn else key
            if canonical_key not in key_to_smiles:
                key_to_smiles[canonical_key] = smiles

        # Sort by RRF score (descending) and keep top N
        sorted_by_score = sorted(
            rrf_scores.items(),
            key=lambda x: (-x[1], x[0])  # descending score, then key for tie-break
        )

        top_candidates = [
            (key_to_smiles[key], score)
            for key, score in sorted_by_score[:keep_top]
        ]

        results[mol_id] = top_candidates

    return results


def rerank_with_forward_models(
    fused_lists: Dict[str, List[Tuple[str, float]]],
    fm_rerank_fn: Optional[Callable[[str, List[str]], List[Tuple[str, float]]]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Optional: apply forward-model re-ranking to fused lists.

    The forward-model re-rank (ICEBERG + GLACIER) is slot-preserving:
    candidates only swap within same-formula groups.

    Args:
      fused_lists: output from fuse()
      fm_rerank_fn: optional callable (mol_id, smiles_list) -> [(smiles, fm_score), ...]

    Returns:
      re-ranked lists if fm_rerank_fn is provided, else unchanged fused_lists
    """

    if fm_rerank_fn is None:
        return fused_lists

    results = {}
    for mol_id, candidates in fused_lists.items():
        smiles_list = [s for s, _ in candidates]
        reranked = fm_rerank_fn(mol_id, smiles_list)
        results[mol_id] = reranked

    return results
