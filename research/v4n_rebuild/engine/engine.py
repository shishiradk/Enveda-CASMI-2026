"""v4r evidence engine (our code, MIT): one molecule's spectra -> candidate structures + 86 ranking features.

Channels (v1-equivalent semantics, REBUILD_SPEC 3 / 6):
  L  direct library search: entropy (+ cosine) similarity to visible library spectra of each pool candidate's train
     structure, per query spectrum, same polarity, structure-mass window +-10 ppm
  A  analog propagation: mass-shifted (hybrid) similarity of the merged query (one per polarity) to one representative
     library spectrum per (structure, polarity, library) within +-200 Da; top-100 analogs; Tanimoto propagation
  M  spectrum -> fingerprint model (pluggable bank, engine.model): f.z on single and merged views, log-likelihood,
     cosine, Tanimoto and element-count agreement
  F  bond-cleavage fragment explanation of the query peaks (top `frag_max_cand` candidates by f.z)
  S  subformula explanation (per unique candidate formula)
  G  class-3 generation: derivatives of the best analogs whose mass matches the target (engine.derive)
Simulation masks (REBUILD_SPEC 4): `exclude` (bool over library spectra; direct search + analog representatives),
`exclude_sid` / `exclude_lib` (hide the analog representative of a structure, optionally only in one library),
`drop_pid` (remove a pool row from the candidate window but keep it generatable).

    E = Engine(Library(d), Pool(d), CFTBank([ckpt]), EngineCfg())
    C, X, info = E.run(spectra, target)        # X: float32 [n_cand, 86] in FEATURES order
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import chem, derive, fingerprint, fragments, spectra as sp
from .tables import Library, Pool

NP_LIBS = np.array([2, 3, 4, 5, 6, 7, 10])        # riken gnps massbank mona spectraverse msdial masaryk


@dataclass
class SearchCfg:
    floor: float = 0.002
    topk: int = 256
    power: float = 1.0
    ent_weight: bool = True
    tol: float = 0.01
    ppm_tol: float = 20.0
    ppm_win: float = 10.0
    analog_win: float = 200.0


@dataclass
class EngineCfg:
    ppm_win: float = 10.0
    ppm_fallback: float = 30.0
    search: SearchCfg = field(default_factory=SearchCfg)
    sim_power: float = 4.0
    n_analog: int = 100
    max_query_spectra: int = 16
    use_frag: bool = True
    use_subformula: bool = True
    frag_max_cand: int = 600
    generate: bool = True
    gen_n_analog: int = 6
    gen_min_sim: float = 0.3
    gen_max_per_parent: int = 60
    gen_max_total: int = 150
    gen_tol_da: float = 0.004
    merged_nm_cap: int = 16          # nm of a merged view = number of spectra merged, capped (CFT trained up to 16)
    gen_true_steps: bool = True      # gen_steps = combo length (v1 always wrote 1)


FEATURES: List[str] = [
    "lib_max", "lib_mean", "lib_top3", "lib_nspec", "lib_cos", "lib_max_rank", "lib_gap", "lib_grp_max", "lib_hit",
    "lib_same_adduct",
    "ap", "a1", "best_tan", "top_tan", "mean_tan", "ap_rank", "ap_gap", "ap_grp_max", "top_sim",
    "ap_np", "best_tan_np", "top_sim_np", "ap_np_gap", "ap_sum", "n_analog_hi", "tan_zero_shift",
    "fz", "fz_z", "fz_rank", "fz_gap", "fz_norm", "fz_top", "fz_softmax", "fz_grp_ent", "fz_grp_gap",
    "fz_single", "fz_single_gap", "fz_merged", "fz_merged_gap", "fp_cos", "fp_tan", "fp_ll", "fp_ll_gap",
    "el_c", "el_n", "el_o", "el_sum", "el_rank",
    "fr_int", "fr_int_mean", "fr_cnt", "fr_top10", "fr_strict", "fr_rank", "fr_gap", "fr_z", "fr_n",
    "sf_int", "sf_cnt", "sf_top10", "sf_rank", "sf_gap", "formula_share", "n_formula", "formula_is_top",
    "mass_ppm", "mass_ppm_rank", "n_heavy", "fp_bits", "n_cand", "n_query", "n_pos", "n_neg", "target_mass",
    "q_npeaks", "lib_x_fz", "ap_x_fz", "agree_lib", "agree_ap", "fr_x_fz",
    "is_gen", "gen_parent_sim", "gen_steps", "gen_parent_tan", "n_gen", "gen_rank_fz",
]
COL = {n: i for i, n in enumerate(FEATURES)}
N_FEAT = len(FEATURES)
assert N_FEAT == 86


def rank01(x):
    """Descending rank / (n-1); ties broken by candidate order (stable)."""
    o = np.argsort(-x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def tanimoto(A, B):
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


@dataclass
class Cands:
    pid: np.ndarray
    key: List[str]
    smiles: List[str]
    formula: List[str]
    mass: np.ndarray
    n_heavy: np.ndarray
    train_sid: np.ndarray
    fp: np.ndarray
    frags: list
    is_gen: np.ndarray
    gen_sim: np.ndarray
    gen_steps: np.ndarray
    gen_tan: np.ndarray
    gen_parent: np.ndarray
    gen_combo: list


def merged_adduct(adducts):
    """Most frequent adduct; ties -> earlier in chem.ADDUCT_LIST, then name (deterministic)."""
    cnt: Dict[str, int] = {}
    for a in adducts:
        cnt[a] = cnt.get(a, 0) + 1
    best = max(cnt.values())
    return min((a for a in cnt if cnt[a] == best), key=lambda a: (chem.adduct_index(a), a))


class Engine:
    def __init__(self, L: Library, P: Pool, bank, cfg: Optional[EngineCfg] = None, verbose=True):
        self.L, self.P, self.bank = L, P, bank
        self.cfg = cfg or EngineCfg()
        bits = bank.fp_bits if bank is not None else np.arange(fingerprint.RAW_BITS)
        P.set_bits(bits)
        L.set_bits(bits)
        self.nbits = len(bits)
        self.bits = np.asarray(bits, np.int64)
        self.rep = self._representatives()
        if verbose:
            print(f"[Engine] {len(self.rep['idx']):,} analog representatives, nbits {self.nbits}", flush=True)

    # ------------------------------------------------------------------------------------------------ analogs
    def _representatives(self):
        L = self.L
        g = (L.sid * 2 + (L.mode > 0)) * 16 + L.lib.astype(np.int64)
        o = np.lexsort((-L.n_clean, g))
        gs = g[o]
        first = np.r_[True, gs[1:] != gs[:-1]]
        r = o[first]
        m = L.smass[r]
        s = np.argsort(m, kind="mergesort")
        r = r[s]
        return dict(idx=r, sid=L.sid[r], mass=m[s], mode=L.mode[r], lib=L.lib[r], is_np=np.isin(L.lib[r], NP_LIBS))

    def _weigh(self, mz, it):
        c = self.cfg.search
        return sp.weigh(np.asarray(mz, np.float64), np.asarray(it, np.float64), c.floor, c.topk, c.power, c.ent_weight)

    def analogs(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1):
        """[(sid, sim, shift, is_np)] best first (top n_analog)."""
        c = self.cfg; L = self.L; R = self.rep; sc_ = c.search
        views = []
        for md in (1, -1):
            g = [(s["mz"], s["it"]) for s in spectra if s["mode"] == md]
            if g:
                mm, ii = sp.merge(g)
                views.append((mm, ii, md))
        lo = np.searchsorted(R["mass"], target - sc_.analog_win, "left")
        hi = np.searchsorted(R["mass"], target + sc_.analog_win, "right")
        ridx, rmass, rmode = R["idx"][lo:hi], R["mass"][lo:hi], R["mode"][lo:hi]
        rsid, rlib, rnp = R["sid"][lo:hi], R["lib"][lo:hi], R["is_np"][lo:hi]
        keep = np.ones(len(ridx), bool)
        if exclude_sid >= 0:
            keep &= ~((rsid == exclude_sid) & (rlib == exclude_lib)) if exclude_lib >= 0 else (rsid != exclude_sid)
        if exclude is not None:
            keep &= ~exclude[ridx]
        ridx, rmass, rmode, rsid, rnp = ridx[keep], rmass[keep], rmode[keep], rsid[keep], rnp[keep]
        best: Dict[int, tuple] = {}
        for mz, it, md in views:
            qm, qp = self._weigh(mz, it)
            if len(qm) == 0:
                continue
            sel = rmode == md
            if not sel.any():
                continue
            shift = (target - rmass[sel]).astype(np.float64)
            sc = sp.score_library_shift(qm, qp, ridx[sel], shift, L.off, L.mz, L.it, sc_.tol, sc_.ppm_tol,
                                        sc_.floor, sc_.topk, sc_.power, sc_.ent_weight)
            csid, cnp = rsid[sel], rnp[sel]
            k = min(4 * c.n_analog, len(sc))
            top = np.argpartition(-sc, k - 1)[:k] if k < len(sc) else np.arange(len(sc))
            for j in top:
                s, v = int(csid[j]), float(sc[j])
                if v > best.get(s, (-1.0,))[0]:
                    best[s] = (v, float(shift[j]), bool(cnp[j]))
        items = sorted(best.items(), key=lambda x: -x[1][0])[:c.n_analog]
        return [(s, v[0], v[1], v[2]) for s, v in items]

    # ------------------------------------------------------------------------------------------------ generation
    def generate(self, analogs, target, exclude_keys: set) -> List[dict]:
        c = self.cfg; L = self.L
        out, seen = [], set(exclude_keys)
        parents = [a for a in analogs if a[1] >= c.gen_min_sim][:c.gen_n_analog]
        tol = max(c.gen_tol_da, target * c.ppm_win * 1e-6)
        fpr = fingerprint.raw_fingerprinter()
        for sid, sim, shift, _ in parents:
            pfp = L.train_fp.get([sid])[0].astype(np.float32)
            delta = target - float(L.struct_mass[sid])
            try:
                prods = derive.derive(L.struct_smiles[sid], delta, tol=tol, max_steps=2, max_out=c.gen_max_per_parent)
            except Exception:
                prods = []
            for smi, combo in prods:
                std = chem.standardize(smi)
                if std is None:
                    continue
                cc = chem.canonical(std)
                if cc is None or cc[2] is None or cc[2] in seen:
                    continue
                m, tm, k = cc
                formula, mass, nh = chem.mol_props(m)
                if abs(mass - target) > tol:
                    continue
                fp = fpr.raw(tm)[self.bits].astype(np.float32)
                inter = float((fp * pfp).sum())
                tan = inter / (fp.sum() + pfp.sum() - inter + 1e-9)
                seen.add(k)
                out.append(dict(smiles=std, key=k, formula=formula, mass=mass, n_heavy=nh, fp=fp,
                                frags=fragments.fragments_of_mol(tm), sim=sim,
                                steps=len(combo) if c.gen_true_steps else 1, tan=tan, parent=sid, combo=combo))
                if len(out) >= c.gen_max_total:
                    return out
        return out

    def candidates(self, target, drop_pid=-1, gen=None) -> Cands:
        P = self.P
        w = P.window(target, self.cfg.ppm_win)
        if len(w) == 0:
            w = P.window(target, self.cfg.ppm_fallback)
        if drop_pid >= 0:
            w = w[w != drop_pid]
        g = gen or []
        n = len(w)
        fp = [P.fps(w).astype(np.float32).reshape(n, self.nbits)] + [x["fp"][None] for x in g]
        fp = np.concatenate(fp) if (n + len(g)) else np.zeros((0, self.nbits), np.float32)
        z = np.zeros(n)
        return Cands(
            pid=np.r_[w, np.full(len(g), -1)].astype(np.int64),
            key=[P.key[i] for i in w] + [x["key"] for x in g],
            smiles=[P.smiles[i] for i in w] + [x["smiles"] for x in g],
            formula=[P.formula[i] for i in w] + [x["formula"] for x in g],
            mass=np.r_[P.mass[w], [x["mass"] for x in g]].astype(np.float64),
            n_heavy=np.r_[P.n_heavy[w], [x["n_heavy"] for x in g]].astype(np.int32),
            train_sid=np.r_[P.train_sid[w], np.full(len(g), -1)].astype(np.int64),
            fp=fp, frags=[None] * n + [x["frags"] for x in g],
            is_gen=np.r_[z, np.ones(len(g))].astype(np.float32),
            gen_sim=np.r_[z, [x["sim"] for x in g]].astype(np.float32),
            gen_steps=np.r_[z, [x["steps"] for x in g]].astype(np.float32),
            gen_tan=np.r_[z, [x["tan"] for x in g]].astype(np.float32),
            gen_parent=np.r_[np.full(n, -1), [x["parent"] for x in g]].astype(np.int64),
            gen_combo=[None] * n + [x["combo"] for x in g])

    # ------------------------------------------------------------------------------------------------ model views
    def model_items(self, spectra):
        bank = self.bank
        single = []
        for s in spectra:
            pm, pi = bank.prepare(s["mz"], s["it"], s["prec"])
            single.append(dict(mz=pm, it=pi, prec=float(s["prec"]), adduct=s["adduct"], ce=s.get("ce", np.nan),
                               nm=1, mode=s["mode"], instrument=s.get("instrument", "timsTOF")))
        merged = []
        for md in (1, -1):
            g = [s for s in spectra if s["mode"] == md]
            if not g:
                continue
            mm, ii = sp.merge([(s["mz"], s["it"]) for s in g])
            prec = float(np.median([s["prec"] for s in g]))
            pm, pi = bank.prepare(mm, ii, prec)
            ces = [s["ce"] for s in g if s.get("ce") is not None and np.isfinite(s["ce"])]
            merged.append(dict(mz=pm, it=pi, prec=prec, adduct=merged_adduct([s["adduct"] for s in g]),
                               ce=float(np.mean(ces)) if ces else np.nan, nm=min(len(g), self.cfg.merged_nm_cap),
                               mode=md, instrument=g[0].get("instrument", "timsTOF")))
        return single, merged

    # ------------------------------------------------------------------------------------------------ run
    def run(self, spectra: List[dict], target: float, exclude: Optional[np.ndarray] = None, exclude_sid: int = -1,
            exclude_lib: int = -1, drop_pid: int = -1):
        """spectra: [dict(mz, it, mode (+1/-1), adduct, prec, ce (NaN = unknown), ce_n, instrument)].
        Returns (Cands, X [n, 86] float32, info)."""
        c = self.cfg; L = self.L; P = self.P; S = c.search
        spectra = spectra[:c.max_query_spectra]
        an = self.analogs(spectra, target, exclude, exclude_sid, exclude_lib)
        gen = None
        if c.generate and an:
            w = P.window(target, c.ppm_win)
            if drop_pid >= 0:
                w = w[w != drop_pid]
            gen = self.generate(an, target, {P.key[i] for i in w})
        C = self.candidates(target, drop_pid, gen)
        nc = len(C.pid)
        if nc == 0:
            return C, np.zeros((0, N_FEAT), np.float32), dict(n_cand=0, lib_max=0.0, top_sim=0.0, fzmax=0.0,
                                                               n_gen=0, analogs=an)
        X = np.zeros((nc, N_FEAT), np.float32)
        cfp = C.fp
        cbits = cfp.sum(1)
        n_pos = sum(1 for s in spectra if s["mode"] > 0)
        n_neg = len(spectra) - n_pos

        def put(name, v):
            X[:, COL[name]] = v

        # ---------------- L
        lmax = np.zeros(nc); lmean = np.zeros(nc); ltop3 = np.zeros(nc); lnum = np.zeros(nc)
        lcos = np.zeros(nc); lsame = np.zeros(nc)
        sid2c = {int(s): i for i, s in enumerate(C.train_sid) if s >= 0}
        per_q: Dict[int, List[float]] = {}
        for s in spectra:
            qm, qp = self._weigh(s["mz"], s["it"])
            if len(qm) == 0:
                continue
            lc = L.window(target, S.ppm_win, exclude, s["mode"])
            if len(lc) == 0:
                continue
            ent = sp.score_library(qm, qp, lc, L.off, L.mz, L.it, S.tol, S.ppm_tol, S.floor, S.topk, S.power,
                                   S.ent_weight, 0)
            cos = sp.score_library(qm, qp, lc, L.off, L.mz, L.it, S.tol, S.ppm_tol, S.floor, S.topk, S.power,
                                   S.ent_weight, 1)
            qad = chem.adduct_index(s["adduct"])
            here: Dict[int, float] = {}
            for j, li in enumerate(lc):
                ci = sid2c.get(int(L.sid[li]))
                if ci is None:
                    continue
                v = float(ent[j])
                if v > here.get(ci, 0.0):
                    here[ci] = v
                if v > lmax[ci]:
                    lmax[ci] = v
                if float(cos[j]) > lcos[ci]:
                    lcos[ci] = float(cos[j])
                if L.adduct_ix[li] == qad and v > lsame[ci]:
                    lsame[ci] = v
                lnum[ci] += 1
            for ci, v in here.items():
                per_q.setdefault(ci, []).append(v)
        nq = max(1, len(spectra))
        for ci, vs in per_q.items():
            lmean[ci] = sum(vs) / nq
            ltop3[ci] = float(np.mean(sorted(vs)[-3:]))
        lib_max_all = float(lmax.max())
        put("lib_max", lmax); put("lib_mean", lmean); put("lib_top3", ltop3); put("lib_nspec", np.log1p(lnum / nq))
        put("lib_cos", lcos); put("lib_max_rank", rank01(lmax)); put("lib_gap", lmax - lib_max_all)
        put("lib_grp_max", lib_max_all); put("lib_hit", (lmax > 0).astype(np.float32)); put("lib_same_adduct", lsame)

        # ---------------- A
        zero = np.zeros(nc, np.float32)
        ap = a1 = best_tan = top_tan = mean_tan = ap_sum = tan0 = ap_np = best_tan_np = zero
        top_sim = top_sim_np = 0.0
        n_hi = 0
        if an:
            aid = np.array([a[0] for a in an])
            asim = np.array([a[1] for a in an], np.float32)
            ashift = np.array([a[2] for a in an])
            anp = np.array([a[3] for a in an])
            T = tanimoto(cfp, L.train_fp.get(aid))
            w4 = asim ** c.sim_power
            ap = (T * w4[None, :]).max(1); a1 = (T * asim[None, :]).max(1)
            best_tan = T.max(1); top_tan = T[:, 0]; top_sim = float(asim[0])
            mean_tan = (T * w4[None, :]).sum(1) / (w4.sum() + 1e-9)
            ap_sum = np.log1p((T * w4[None, :]).sum(1))
            n_hi = int((asim > 0.5).sum())
            z0 = np.abs(ashift) < 0.01
            tan0 = (T[:, z0] * asim[None, z0]).max(1) if z0.any() else zero
            if anp.any():
                ap_np = (T[:, anp] * w4[None, anp]).max(1); best_tan_np = T[:, anp].max(1)
                top_sim_np = float(asim[anp][0])
        apmax = float(ap.max())
        put("ap", ap); put("a1", a1); put("best_tan", best_tan); put("top_tan", top_tan); put("mean_tan", mean_tan)
        put("ap_rank", rank01(ap)); put("ap_gap", ap - apmax); put("ap_grp_max", apmax); put("top_sim", top_sim)
        put("ap_np", ap_np); put("best_tan_np", best_tan_np); put("top_sim_np", top_sim_np)
        put("ap_np_gap", ap_np - float(ap_np.max())); put("ap_sum", ap_sum); put("n_analog_hi", n_hi)
        put("tan_zero_shift", tan0)

        # ---------------- M
        fz = np.zeros(nc, np.float32); fz_s = np.zeros(nc, np.float32); fz_m = np.zeros(nc, np.float32)
        el_pred = None
        if self.bank is not None:
            items_s, items_m = self.model_items(spectra)
            zs, es = self.bank.logits_el(items_s)
            zm, em = self.bank.logits_el(items_m)
            z1 = zs.mean(0); z2 = zm.mean(0)
            el_pred = 0.5 * (es.mean(0) + em.mean(0))
            fz_s = cfp @ z1; fz_m = cfp @ z2
            za = 0.5 * (z1 + z2)
            fz = cfp @ za
            pz = np.clip(1.0 / (1.0 + np.exp(-np.clip(za, -30, 30))), 1e-6, 1 - 1e-6)
            ll = cfp @ np.log(pz) + (1 - cfp) @ np.log(1 - pz)
            put("fp_ll", ll / max(1.0, self.nbits) * 100); put("fp_ll_gap", (ll - ll.max()) / 100.0)
            put("fp_cos", (cfp @ pz) / (np.sqrt(cbits) * np.linalg.norm(pz) + 1e-9))
            pb = (pz > 0.5).astype(np.float32)
            inter = cfp @ pb
            put("fp_tan", inter / (cbits + pb.sum() - inter + 1e-9))
        fzmax = float(fz.max())
        smx = np.exp((fz - fzmax) / math.sqrt(self.nbits) * 4.0)
        smx = smx / smx.sum()
        put("fz", fz / 100.0); put("fz_z", zscore(fz)); put("fz_rank", rank01(fz)); put("fz_gap", (fz - fzmax) / 100.0)
        put("fz_norm", fz / np.sqrt(np.maximum(cbits, 1)) / 10.0); put("fz_top", (fz == fzmax).astype(np.float32))
        put("fz_softmax", smx); put("fz_grp_ent", -(smx * np.log(smx + 1e-12)).sum())
        srt = np.sort(fz)[::-1]
        put("fz_grp_gap", (srt[0] - srt[1]) / 100.0 if nc > 1 else 0.0)
        put("fz_single", fz_s / 100.0); put("fz_single_gap", (fz_s - fz_s.max()) / 100.0)
        put("fz_merged", fz_m / 100.0); put("fz_merged_gap", (fz_m - fz_m.max()) / 100.0)
        if el_pred is not None:
            E = np.zeros((nc, len(chem.ELEMS10)), np.float32)
            for i in range(nc):
                v = chem.formula_vec(C.formula[i])
                if v is not None:
                    E[i] = np.log1p(v)
            d = np.abs(E - el_pred[None, :])
            put("el_c", d[:, 0]); put("el_n", d[:, 2]); put("el_o", d[:, 3]); put("el_sum", d.sum(1))
            put("el_rank", rank01(-d.sum(1)))

        # ---------------- F
        peaks = []
        for s in spectra:
            m, v = sp.weigh(np.asarray(s["mz"], np.float64), np.asarray(s["it"], np.float64), 0.002, 128, 1.0, False)
            if len(m):
                peaks.append((m, v / v.max(), s["adduct"]))
        if c.use_frag and peaks:
            sel = np.arange(nc) if nc <= c.frag_max_cand else np.argsort(-fz)[:c.frag_max_cand]
            fr = np.zeros((nc, 5), np.float32); frn = np.zeros(nc, np.float32)
            for i in sel:
                fg = C.frags[i] if C.frags[i] is not None else P.frags(int(C.pid[i]))
                fr[i] = fragments.frag_features(fg, peaks)
                frn[i] = len(fg)
            put("fr_int", fr[:, 0]); put("fr_int_mean", fr[:, 1]); put("fr_cnt", fr[:, 2]); put("fr_top10", fr[:, 3])
            put("fr_strict", fr[:, 4]); put("fr_rank", rank01(fr[:, 0])); put("fr_gap", fr[:, 0] - fr[:, 0].max())
            put("fr_z", zscore(fr[:, 0])); put("fr_n", np.log1p(frn))

        # ---------------- S
        uniq: Dict[str, int] = {}
        for f in C.formula:
            uniq[f] = uniq.get(f, 0) + 1
        put("n_formula", len(uniq))
        put("formula_share", np.array([uniq[f] / nc for f in C.formula], np.float32))
        if c.use_subformula and peaks:
            cache = {f: fragments.subformula_features(f, peaks) for f in uniq}
            sf = np.stack([cache[f] for f in C.formula])
            put("sf_int", sf[:, 0]); put("sf_cnt", sf[:, 2]); put("sf_top10", sf[:, 3]); put("sf_rank", rank01(sf[:, 0]))
            put("sf_gap", sf[:, 0] - sf[:, 0].max())
            put("formula_is_top", (sf[:, 0] >= sf[:, 0].max() - 1e-6).astype(np.float32))

        # ---------------- priors / interactions
        ppm = (C.mass - target) / target * 1e6
        put("mass_ppm", ppm); put("mass_ppm_rank", rank01(-np.abs(ppm)))
        put("n_heavy", C.n_heavy); put("fp_bits", cbits / 100.0)
        put("n_cand", math.log(nc)); put("n_query", len(spectra)); put("n_pos", n_pos); put("n_neg", n_neg)
        put("target_mass", target / 1000.0)
        put("q_npeaks", math.log1p(float(np.mean([len(s["mz"]) for s in spectra]))) if spectra else 0.0)
        mr = rank01(fz)
        put("lib_x_fz", lmax * (1.0 - mr)); put("ap_x_fz", ap * (1.0 - mr))
        put("agree_lib", (1.0 - mr[int(np.argmax(lmax))]) if lib_max_all > 0 else 0.0)
        put("agree_ap", (1.0 - mr[int(np.argmax(ap))]) if apmax > 0 else 0.0)
        put("fr_x_fz", X[:, COL["fr_int"]] * (1.0 - mr))

        # ---------------- G
        put("is_gen", C.is_gen); put("gen_parent_sim", C.gen_sim); put("gen_steps", C.gen_steps)
        put("gen_parent_tan", C.gen_tan); put("n_gen", float(C.is_gen.sum()))
        if C.is_gen.any():
            gi = np.flatnonzero(C.is_gen > 0)
            X[gi, COL["gen_rank_fz"]] = rank01(fz[gi])
        info = dict(n_cand=nc, lib_max=lib_max_all, top_sim=top_sim, fzmax=fzmax, n_gen=int(C.is_gen.sum()),
                    analogs=an)
        return C, X, info


def spectra_from_rows(rows) -> List[dict]:
    """Rows of a test/bench parquet (one molecule) -> engine query spectra (test.parquet column names)."""
    out = []
    for r in rows.itertuples():
        ce = r.collision_energy_ev
        ce = np.atleast_1d(np.asarray(ce, float)) if ce is not None else np.zeros(0)
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                        mode=1 if r.ionization_mode == "positive" else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                        ce=float(ce.mean()) if len(ce) else np.nan, ce_n=int(len(ce)) if len(ce) else 1,
                        instrument=getattr(r, "instrument_type", "timsTOF")))
    return out


def target_of(rows) -> float:
    nm = chem.neutral_mass(rows.precursor_mz.values.astype(np.float64), rows.adduct.values)
    nm = nm[np.isfinite(nm)]
    return float(np.median(nm)) if len(nm) else float("nan")
