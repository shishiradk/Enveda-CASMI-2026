"""Same-formula re-ordering of a ranked candidate list with forward-model scores (ours, MIT).

Definition (per molecule; ``L`` = the engine's de-duplicated list, best first):

1. Only the first ``topn`` entries take part.  They are split into groups of
   identical molecular formula.  A candidate whose formula is unknown (None)
   is a group of its own.
2. A forward model "covers" a group when it scored at least two of its members
   and those scores are not all equal.  A group covered by no model is left
   exactly as it is.
3. In a covered group every member gets
       fused = z(ranker) + sum_m lam_m * z_m
   * z(ranker): z-score of the ranker score over ALL members of the group
     (sample standard deviation, ddof = 1; 0 for every member when the spread
     is 0).  If any ranker score of the group is missing or not finite, the
     group's ranker scores are replaced by minus the list position.
   * z_m: z-score of model m over the members it scored (ddof = 1).  A member
     the model did not score gets z_m = 0, i.e. it is placed at the mean of the
     scored members for that model; a model that does not cover the group
     contributes 0 for every member.  A candidate without any forward score is
     therefore ordered by its ranker z-score alone.
4. The group's members are sorted by fused value (descending, rounded to 1e-9;
   exact ties keep their current relative order) and written back into the
   positions the group already occupied.  Positions of other formulas, and
   everything after ``topn``, never change.

The function returns a permutation: ``perm[new_position] = old_position``.
"""
import math

GRID = 1e-9


def _ok(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def zscores(vals):
    """z-score over the non-missing values (ddof = 1).  Returns (z list with 0.0 for missing, covered flag)."""
    xs = [float(v) for v in vals if _ok(v)]
    n = len(xs)
    if n < 2:
        return [0.0] * len(vals), False
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    if not sd > 1e-12:
        return [0.0] * len(vals), False
    return [((float(v) - mean) / sd) if _ok(v) else 0.0 for v in vals], True


def rerank_molecule(formulas, ranker, fwd, lams, topn=60):
    """formulas, ranker: one entry per candidate (list order = current ranking).
    fwd: {model: [score | None, ...]}; lams: {model: weight}.  Lists shorter than the
    candidate list are padded with None.  Returns (perm, info)."""
    n = len(formulas)
    top = min(n, int(topn))
    ranker = list(ranker) + [None] * (n - len(ranker))
    fwd = {m: list(v) + [None] * (n - len(v)) for m, v in fwd.items()}
    perm = list(range(n))
    groups = {}
    for i in range(top):
        if formulas[i] is not None:
            groups.setdefault(formulas[i], []).append(i)
    info = {"groups": 0, "groups_covered": 0, "groups_changed": 0, "members_covered": 0}
    for members in groups.values():
        if len(members) < 2:
            continue
        info["groups"] += 1
        zs, covered = {}, False
        for m, lam in lams.items():
            if m not in fwd or not lam:
                continue
            z, cov = zscores([fwd[m][i] for i in members])
            if cov:
                zs[m], covered = z, True
        if not covered:
            continue
        info["groups_covered"] += 1
        info["members_covered"] += len(members)
        rs = [ranker[i] for i in members]
        if not all(_ok(r) for r in rs):
            rs = [-float(i) for i in members]
        zr, _ = zscores(rs)
        fused = []
        for j in range(len(members)):
            val = zr[j] + sum(lams[m] * zs[m][j] for m in zs)
            fused.append(round(val / GRID) * GRID)
        order = sorted(range(len(members)), key=lambda j: (-fused[j], j))
        if order != list(range(len(members))):
            info["groups_changed"] += 1
        for slot, j in zip(members, order):
            perm[slot] = members[j]
    return perm, info


def list_changes(old, new, k=25):
    """Compare two orderings of keys: top-1 changed, top-k order changed, top-k set changed."""
    a, b = list(old[:k]), list(new[:k])
    return {"top1": bool(a[:1] != b[:1]), "order": bool(a != b), "set": bool(set(a) != set(b))}
