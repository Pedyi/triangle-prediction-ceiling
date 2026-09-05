#!/usr/bin/env python3
"""
triangle_prediction_ceiling.py
==============================

Complete verification code for

    "Localization, Free Statistics, and Prediction:
     A Tight Ceiling for Weighted Closing-Vertex Triangle Estimators"

Every theorem, lemma and empirical claim in the paper is checked here. The file
is self-contained: no project imports, one dependency set (numpy, scipy,
networkx, matplotlib optional).

    https://github.com/Pedyi/triangle-prediction-ceiling

--------------------------------------------------------------------------------
NOTATION
--------------------------------------------------------------------------------
For an edge e = (u,v):

    T(e) = N(u) & N(v)                     closing vertices,   t(e) = |T(e)|
    S(e) = (N(u) | N(v)) \\ {u,v}            candidate domain,   |S(e)| = d(u)+d(v)-t(e)-2

Estimator (Definition 3.2): draw e ~ Unif(E), then w ~ q_e supported on S(e),

    X = (m/3) * 1[w in T(e)] / q_e(w)

Lemma 3.3:   E[X] = T,   E[X^2] = (m/9) * sum_e C_q(e),  C_q(e) = sum_{w in T(e)} 1/q_e(w)

Key implementation fact (Remark 3.4): C_q(e) and p_q(e) depend on q_e ONLY through
its restriction to T(e). S(e) is never enumerated anywhere in this file; only
|S(e)| and, for binned scores, the per-bin counts n_j(e) obtained by
inclusion-exclusion.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python triangle_prediction_ceiling.py all
    python triangle_prediction_ceiling.py identities      # Sec 6.2
    python triangle_prediction_ceiling.py channels        # Sec 6.3
    python triangle_prediction_ceiling.py features        # Sec 6.4
    python triangle_prediction_ceiling.py suffix          # Sec 6.5, 6.6
    python triangle_prediction_ceiling.py cost            # Sec 6.7
    python triangle_prediction_ceiling.py metric          # Sec 6.8
    python triangle_prediction_ceiling.py controls        # Sec 6.9
    python triangle_prediction_ceiling.py tables          # LaTeX tables

    --graphs real|synthetic|all|"<name>;<name>" (default: all; separator is ";",
                                                since graph names contain commas)
    --json out.json                             dump raw results
    --quick                                     smaller sweeps, for a smoke test

Real graphs are downloaded from SNAP on first use and cached in ./snap_cache/.
If the download fails the run continues with whatever loaded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from typing import Callable

import numpy as np
import networkx as nx
from scipy.optimize import minimize

np.seterr(over="ignore", invalid="ignore")

CACHE = "snap_cache"
SNAP_URLS = {
    "fb-ego":         "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "ca-GrQc":        "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ca-HepTh":       "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "p2p-Gnutella08": "https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz",
    "oregon1":        "https://snap.stanford.edu/data/oregon1_010331.txt.gz",
    "email-Enron":    "https://snap.stanford.edu/data/email-Enron.txt.gz",
}
SYNTHETIC = {
    "ER(3000,0.004)":     lambda: nx.gnp_random_graph(3000, 0.004, seed=1),
    "BA(3000,5)":         lambda: nx.barabasi_albert_graph(3000, 5, seed=2),
    "WS(3000,10,p=0.01)": lambda: nx.watts_strogatz_graph(3000, 10, 0.01, seed=3),
    "WS(3000,10,p=0.10)": lambda: nx.watts_strogatz_graph(3000, 10, 0.10, seed=4),
    "WS(3000,10,p=0.50)": lambda: nx.watts_strogatz_graph(3000, 10, 0.50, seed=5),
    "RGG(3000,r=0.045)":  lambda: nx.random_geometric_graph(3000, 0.045, seed=6),
    "PowerlawCluster":    lambda: nx.powerlaw_cluster_graph(3000, 5, 0.6, seed=7),
}


# =============================================================================
#  Loading
# =============================================================================

def _load_snap(name: str, url: str, timeout: int = 120) -> nx.Graph:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".txt.gz")
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        txt = gzip.decompress(f.read()).decode("utf-8", "ignore")
    G = nx.Graph()
    for line in txt.splitlines():
        if line and line[0] != "#":
            p = line.split()
            if len(p) >= 2 and p[0] != p[1]:
                G.add_edge(int(p[0]), int(p[1]))
    return G


def load_graphs(which: str = "all", verbose: bool = True) -> dict[str, nx.Graph]:
    want = None
    if which not in ("all", "real", "synthetic"):
        want = {s.strip() for s in which.split(";")}
    out: dict[str, nx.Graph] = {}
    if which in ("all", "real") or want:
        for name, url in SNAP_URLS.items():
            if want and name not in want:
                continue
            try:
                out[name] = _load_snap(name, url)
                if verbose:
                    print(f"  OK   {name:16s} n={out[name].number_of_nodes():7d} "
                          f"m={out[name].number_of_edges():8d}")
            except Exception as ex:
                if verbose:
                    print(f"  FAIL {name:16s} -> {type(ex).__name__}: {ex}")
    if which in ("all", "synthetic") or want:
        for name, fn in SYNTHETIC.items():
            if want and name not in want:
                continue
            G = fn()
            G.remove_edges_from(nx.selfloop_edges(G))
            out[name] = G
            if verbose:
                print(f"  gen  {name:16s} n={G.number_of_nodes():7d} m={G.number_of_edges():8d}")
    return out


# =============================================================================
#  Core structures
# =============================================================================

def build(G: nx.Graph, with_edge_ids: bool = False, with_core: bool = False) -> dict:
    """Per-edge t(e), |S(e)|, and the flattened list of closing vertices.

    Returns arrays of length m (edges) and 3T (closing-vertex incidences).
    with_edge_ids adds TU/TV, the edge indices of (u,w) and (v,w), needed only
    by the suffix simulation.
    """
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    E = [(idx[u], idx[v]) if idx[u] < idx[v] else (idx[v], idx[u]) for u, v in G.edges()]
    m = len(E)
    eid = {e: i for i, e in enumerate(E)} if with_edge_ids else None
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)
    deg = np.array([len(a) for a in adj], np.int64)

    t = np.zeros(m, np.int64)
    Ssz = np.zeros(m, np.int64)
    TE: list[int] = []
    TW: list[int] = []
    TU: list[int] = []
    TV: list[int] = []
    for i, (u, v) in enumerate(E):
        common = adj[u] & adj[v]
        t[i] = len(common)
        Ssz[i] = deg[u] + deg[v] - t[i] - 2
        for w in common:
            TE.append(i)
            TW.append(w)
            if with_edge_ids:
                TU.append(eid[(min(u, w), max(u, w))])
                TV.append(eid[(min(v, w), max(v, w))])

    E = np.array(E, np.int64)
    TE = np.array(TE, np.int64)
    TW = np.array(TW, np.int64)
    pos = t > 0
    # T(e) subset of S(e) must hold for unbiasedness (Remark 3.1)
    assert np.all(Ssz[pos] >= t[pos]), "T(e) not contained in S(e)"

    c = dict(n=n, m=m, E=E, adj=adj, deg=deg, t=t, Ssz=Ssz, pos=pos,
             TE=TE, TW=TW, t_pos=t[pos], S_pos=Ssz[pos],
             T=int(t.sum()) // 3,
             node_tri=np.bincount(TW, minlength=n).astype(float))
    if with_edge_ids:
        c["TU"] = np.array(TU, np.int64)
        c["TV"] = np.array(TV, np.int64)
    if with_core:
        cn = nx.core_number(G)
        c["core"] = np.array([cn[v] for v in nodes], float)
    return c


def kappa_tilde(c: dict) -> float:
    """kappa~ = sum t^2 / sum t|S|   (Eq. 3.7)."""
    t = c["t_pos"].astype(float)
    S = c["S_pos"].astype(float)
    return float(np.sum(t * t) / np.sum(t * S))


def M1_numerator(c: dict) -> float:
    """sum_e t(e)|S(e)|, i.e. 9/m times the second moment of Arm 1."""
    return float(np.sum(c["t_pos"].astype(float) * c["S_pos"].astype(float)))


def G_loc(c: dict) -> float:
    return float((c["n"] - 2) * c["t_pos"].sum() / M1_numerator(c))


def Dbar(c: dict) -> float:
    Eu, Ev = c["E"][:, 0], c["E"][:, 1]
    return float(np.mean(c["deg"][Eu] + c["deg"][Ev] - 1))


# =============================================================================
#  Arms and predictor families  (Definition 3.5)
# =============================================================================

def arm0(c):                       # global uniform
    t = c["t_pos"].astype(float)
    return t * (c["n"] - 2), t / (c["n"] - 2)


def arm1(c):                       # localized, shape-free
    t = c["t_pos"].astype(float)
    S = c["S_pos"].astype(float)
    return t * S, t / S


def pred_oracle(c):                # q_e = Unif(T(e)); attains the ceiling
    t = c["t_pos"].astype(float)
    return t * t, np.ones_like(t)


def pred_mixture(c, eta):          # (1-eta) oracle + eta shape-free
    t = c["t_pos"].astype(float)
    S = c["S_pos"].astype(float)
    q = (1 - eta) / t + eta / S
    return t / q, (1 - eta) + eta * t / S


def pred_random(c, rng, conc=1.0):
    """Random predictor. Only the restriction to T(e) matters, so we sample
    total mass alpha_e on T(e) and a Dirichlet split; S(e) is never touched."""
    t = c["t_pos"]
    Ce = np.zeros(len(t))
    pe = np.zeros(len(t))
    for i, k in enumerate(t):
        w = rng.dirichlet(np.full(int(k), conc))
        w = np.maximum(w, 1e-12)
        w /= w.sum()
        alpha = rng.uniform(0.05, 1.0)
        Ce[i] = np.sum(1.0 / (alpha * w))
        pe[i] = alpha
    return Ce, pe


def pred_spiked(c, rng, spike=0.9):
    """Adversarially concentrated: large concentration penalty by construction."""
    t = c["t_pos"]
    Ce = np.zeros(len(t))
    pe = np.zeros(len(t))
    for i, k in enumerate(t):
        k = int(k)
        q = np.full(k, (1 - spike) / k)
        q[rng.integers(k)] += spike
        q = q / q.sum() * spike
        Ce[i] = np.sum(1.0 / q)
        pe[i] = q.sum()
    return Ce, pe


def gain(c, Ce) -> float:
    """G = M1 / E[X^2]_q, the gain of a predictor over Arm 1."""
    return float(M1_numerator(c) / np.sum(Ce))


# =============================================================================
#  Node scores (continuous)  --  Definition 5.1
# =============================================================================

def G_from_score(c: dict, g: np.ndarray) -> float:
    """C(e) = G_S(e) * sum_{w in T(e)} 1/g(w), with G_S(e) by inclusion-exclusion."""
    g = np.maximum(np.asarray(g, float), 1e-12)
    Eu, Ev = c["E"][:, 0], c["E"][:, 1]
    Sig = (np.bincount(Eu, weights=g[Ev], minlength=c["n"]) +
           np.bincount(Ev, weights=g[Eu], minlength=c["n"]))
    sT_g = np.bincount(c["TE"], weights=g[c["TW"]], minlength=c["m"])
    sT_i = np.bincount(c["TE"], weights=1.0 / g[c["TW"]], minlength=c["m"])
    GS = Sig[Eu] + Sig[Ev] - sT_g - g[Eu] - g[Ev]
    p = c["pos"] & (GS > 0)
    if p.sum() < c["pos"].sum():
        return float("nan")
    return float(M1_numerator(c) / np.sum(GS[p] * sT_i[p]))


def optimise_score(c, feats, restarts=5, rng=None):
    """Maximise G over g = exp(theta . log-features)."""
    rng = rng or np.random.default_rng(0)
    F = np.column_stack([np.log(np.maximum(f, 1.0)) for f in feats])

    def neg(th):
        v = G_from_score(c, np.exp(np.clip(F @ th, -30, 30)))
        return 1e6 if v != v else -v

    best, bth = -neg(np.zeros(F.shape[1])), np.zeros(F.shape[1])
    for _ in range(restarts):
        r = minimize(neg, rng.normal(0, 1.0, F.shape[1]), method="Nelder-Mead",
                     options=dict(maxiter=800, xatol=1e-3, fatol=1e-5))
        if -r.fun > best:
            best, bth = -r.fun, r.x
    return best, bth


def score_null(c, feats, theta, nperm=10):
    """Permutation null: relabel scores across vertices, graph untouched."""
    F = np.column_stack([np.log(np.maximum(f, 1.0)) for f in feats])
    g = np.exp(np.clip(F @ theta, -30, 30))
    vals = []
    for r in range(nperm):
        rng = np.random.default_rng(770 + r)
        v = G_from_score(c, g[rng.permutation(c["n"])])
        if v == v:
            vals.append(v)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std())


def feature_sets(c):
    """F1 admissible (one pass, O(n)); F2 needs a declared multi-pass budget;
    F3 is INADMISSIBLE -- computing node triangle counts is the problem."""
    return {
        "F1 deg": [c["deg"].astype(float)],
        "F2 deg+core": [c["deg"].astype(float), c["core"]],
        "F3 +nodetri (INADMISSIBLE)": [c["deg"].astype(float), c["core"], c["node_tri"]],
    }


# =============================================================================
#  Binned scores  --  Definition 5.1, binned sub-family
# =============================================================================

def degree_bins(c, K):
    order = np.argsort(c["deg"], kind="stable")
    b = np.empty(c["n"], np.int64)
    b[order] = np.minimum((np.arange(c["n"]) * K) // c["n"], K - 1)
    return b


def bin_profiles(c, binof, K):
    """t_j(e) and n_j(e) for edges with t>0. n_j by inclusion-exclusion."""
    Eu, Ev = c["E"][:, 0], c["E"][:, 1]
    NB = np.zeros((c["n"], K))
    np.add.at(NB, (Eu, binof[Ev]), 1.0)
    np.add.at(NB, (Ev, binof[Eu]), 1.0)
    tj = np.zeros((c["m"], K))
    np.add.at(tj, (c["TE"], binof[c["TW"]]), 1.0)
    nj = NB[Eu] + NB[Ev] - tj
    nj[np.arange(c["m"]), binof[Eu]] -= 1.0
    nj[np.arange(c["m"]), binof[Ev]] -= 1.0
    nj = np.maximum(nj, 0.0)
    assert np.all(nj >= tj - 1e-9), "n_j < t_j: inclusion-exclusion wrong"
    assert np.max(np.abs(nj.sum(1) - c["Ssz"])) < 1e-6, "sum_j n_j != |S(e)|"
    p = c["pos"]
    return tj[p], nj[p]


def bin_C(u, tj, nj):
    u = np.clip(u, -25, 25)
    return (nj @ np.exp(u)) * (tj @ np.exp(-u))


def bin_p(u, tj, nj):
    u = np.clip(u, -25, 25)
    g = np.exp(u)
    return (tj @ g) / np.maximum(nj @ g, 1e-300)


def _bin_grad(u, tj, nj):
    u = np.clip(u, -25, 25)
    g, gi = np.exp(u), np.exp(-u)
    Z, Q = nj @ g, tj @ gi
    return (nj * g).T @ Q - (tj * gi).T @ Z


def fit_bins(tj, nj, u0=None, maxiter=200, objective="cost"):
    """u = log-scores, pinned at u[0]=0 (the objective is scale invariant).
    gamma == 1 reproduces Arm 1, so the cost objective returns G >= 1."""
    K = tj.shape[1]
    x0 = np.zeros(K - 1) if u0 is None else np.asarray(u0)[1:]
    if objective == "cost":
        f = lambda x: float(bin_C(np.concatenate([[0.], x]), tj, nj).sum())
        jac = lambda x: _bin_grad(np.concatenate([[0.], x]), tj, nj)[1:]
    else:
        f = lambda x: -float(bin_p(np.concatenate([[0.], x]), tj, nj).sum())
        jac = None
    r = minimize(f, x0, jac=jac, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return np.concatenate([[0.], r.x])


def C_infinity(tj, nj):
    """Per-edge optimum, Theorem 5.5:  ( sum_j sqrt(t_j n_j) )^2 ,
    attained at gamma_j ~ sqrt(t_j/n_j).  Overfits; see the permutation null."""
    return np.sqrt(tj * nj).sum(1) ** 2


def advice_curve(c, tj, nj, bmax, objective="cost", seed=0, sweeps=5):
    """Codebook of 2^b weightings, warm-started by splitting, encoder given T(e).
    Upper-bounds every b-bit scheme."""
    rng = np.random.default_rng(seed)
    me, K = tj.shape
    num = M1_numerator(c)
    U = fit_bins(tj, nj, objective=objective)[None, :]
    out = {}
    for b in range(bmax + 1):
        if b:
            U = np.repeat(U, 2, 0)
            U[1::2] += rng.normal(0, 0.25, (U.shape[0] // 2, K))
        for _ in range(sweeps):
            Cs = np.stack([bin_C(u, tj, nj) for u in U], 1)
            Ps = np.stack([bin_p(u, tj, nj) for u in U], 1)
            a = np.argmin(Cs, 1) if objective == "cost" else np.argmax(Ps, 1)
            for k in range(U.shape[0]):
                s = a == k
                if s.sum() >= 2:
                    U[k] = fit_bins(tj[s], nj[s], U[k], maxiter=80, objective=objective)
        Ce = np.stack([bin_C(u, tj, nj) for u in U], 1)[np.arange(me), a]
        pe = np.stack([bin_p(u, tj, nj) for u in U], 1)[np.arange(me), a]
        Chat = c["t_pos"].astype(float) ** 2 / np.maximum(pe, 1e-300)
        out[b] = dict(G=float(num / Ce.sum()), pbar=float(pe.mean()),
                      rho=float(Ce.sum() / Chat.sum()),
                      logC=float(np.log10(max(Ce.sum(), 1e-300))))
    return out


# =============================================================================
#  Suffix model  --  Lemmas 4.3-4.5
# =============================================================================

def suffix_closed(c) -> float:
    """kappa~_suf from the closed form (Corollary 4.6).

    E[t_suf^2]     = t/3 + t(t-1)/5
    E[t_suf S_suf] = t/3 + 3t(t-1)/10 + t(S-t)/4
    Ratio of the summed expectations; this IS E[N]/E[D] exactly, by linearity.
    """
    t = c["t_pos"].astype(float)
    S = c["S_pos"].astype(float)
    num = np.sum(t) * 2 / 15 + np.sum(t * t) / 5
    den = np.sum(t) / 30 + np.sum(t * t) / 20 + np.sum(t * S) / 4
    return float(num / den)


def suffix_simulate(c, reps=40, seed0=3000, per_edge=False):
    """Direct simulation over random stream orders. This is the ESTIMATOR, not
    the reference; on hub-dominated graphs N is right-skewed and few-sample runs
    agree with one another while sitting low (Section 6.6)."""
    m, n, E = c["m"], c["n"], c["E"]
    Eu, Ev = E[:, 0], E[:, 1]
    Ns, Ds = [], []
    s_t2 = np.zeros(m)
    s_tS = np.zeros(m)
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        pos = np.empty(m, np.int64)
        pos[rng.permutation(m)] = np.arange(m)
        order = np.argsort(pos)
        cnt = np.zeros(n, np.int64)
        au = np.empty(m, np.int64)
        av = np.empty(m, np.int64)
        for k in range(m - 1, -1, -1):          # one reverse sweep gives a_u(e)
            e = order[k]
            u, v = Eu[e], Ev[e]
            au[e], av[e] = cnt[u], cnt[v]
            cnt[u] += 1
            cnt[v] += 1
        pe = pos[c["TE"]]
        both = (pos[c["TU"]] > pe) & (pos[c["TV"]] > pe)
        ts = np.bincount(c["TE"], weights=both.astype(float), minlength=m)
        Ss = (au + av).astype(float) - ts       # Lemma 4.4
        assert Ss.min() >= -1e-9, "S_suf negative: counting identity broken"
        Ns.append(float(np.sum(ts * ts)))
        Ds.append(float(np.sum(ts * Ss)))
        if per_edge:
            s_t2 += ts * ts
            s_tS += ts * Ss
    out = dict(N=np.array(Ns), D=np.array(Ds))
    if per_edge:
        out["m2"] = s_t2 / reps
        out["mx"] = s_tS / reps
    return out


def participation_ratio(c) -> float:
    x = (c["t_pos"].astype(float) * c["S_pos"].astype(float))
    return float(x.sum() ** 2 / np.sum(x * x)) / len(x)


# =============================================================================
#  Cost model  --  Definition 4.8, Theorem 4.9
# =============================================================================

def theta_star(G_prime: float, Dbar_: float, gamma: float = 0.0) -> float:
    """Largest predictor inference cost, in stream operations, that pays off."""
    return 2.0 * (G_prime - 1.0) * (1.0 + gamma * Dbar_ / 2.0) / Dbar_


# =============================================================================
#  Experiments
# =============================================================================

def exp_identities(graphs, quick=False):
    """Section 6.2: Monte Carlo validation, telescoping, ceiling violations."""
    print("\n=== 6.2  The identities ===\n")

    # --- Monte Carlo against the closed forms, on a small instance
    Gsm = nx.watts_strogatz_graph(250, 8, 0.15, seed=11)
    c = build(Gsm)
    n_samples = 100_000 if quick else 500_000
    rng = np.random.default_rng(42)
    adj, E, m, n = c["adj"], c["E"], c["m"], c["n"]
    Slist, qlist, inT = [], [], []
    for i, (u, v) in enumerate(E):
        Sset = (adj[u] | adj[v]) - {u, v}
        S = np.fromiter(Sset, np.int64, len(Sset))
        Tset = adj[u] & adj[v]
        mask = np.array([w in Tset for w in S], bool)
        k = mask.sum()
        q = np.full(len(S), 1.0 / max(len(S), 1))
        if len(S) and k:
            q = np.empty(len(S))
            q[mask] = 0.7 / k + 0.3 / len(S)
            q[~mask] = 0.3 / len(S)
            q /= q.sum()
        Slist.append(S); qlist.append(q); inT.append(mask)
    tot = tot2 = 0.0
    for i in rng.integers(0, m, n_samples):
        S, q = Slist[i], qlist[i]
        if len(S) == 0:
            continue
        j = rng.choice(len(S), p=q)
        x = (m / 3.0) / q[j] if inT[i][j] else 0.0
        tot += x; tot2 += x * x
    Ce, _ = pred_mixture(c, 0.3)
    m2 = c["m"] / 9.0 * Ce.sum()
    print(f"  Monte Carlo (mixture eta=0.3, N={n_samples}):")
    print(f"    E[X]   predicted {c['T']:10d}   measured {tot/n_samples:12.2f}")
    print(f"    E[X^2] predicted {m2:12.4e}   measured {tot2/n_samples:12.4e}"
          f"   rel.dev {tot2/n_samples/m2-1:+.4f}")

    # --- telescoping and ceiling
    print(f"\n  {'graph':24s} {'telescope err':>14s} {'ceiling':>9s} "
          f"{'oracle/ceiling':>15s} {'violations':>11s}")
    fams: list[tuple[str, Callable]] = [
        ("oracle", pred_oracle),
        ("mixture 0.1", lambda x: pred_mixture(x, 0.1)),
        ("mixture 0.5", lambda x: pred_mixture(x, 0.5)),
        ("mixture 0.9", lambda x: pred_mixture(x, 0.9)),
        ("random", lambda x: pred_random(x, np.random.default_rng(7))),
        ("spiked", lambda x: pred_spiked(x, np.random.default_rng(7))),
    ]
    total_viol = 0
    for name, G in graphs.items():
        c = build(G)
        A, _ = arm0(c); B, _ = arm1(c)
        O, _ = pred_oracle(c)
        gl, gp, gt = gain(c, B) and (A.sum() / B.sum()), B.sum() / O.sum(), A.sum() / O.sum()
        err = abs(gl * gp - gt) / gt
        ceil = 1.0 / kappa_tilde(c)
        viol = 0
        for fname, f in fams:
            Ce, _ = f(c)
            if gain(c, Ce) > ceil * (1 + 1e-9):
                viol += 1
                print(f"      VIOLATION: {name} / {fname}")
        total_viol += viol
        print(f"  {name:24s} {err:14.2e} {ceil:9.3f} {gain(c, O)/ceil:15.6f} {viol:11d}")
    print(f"\n  total ceiling violations: {total_viol} (expected 0)")


def exp_channels(graphs, quick=False):
    """Section 6.3: the three channels."""
    print("\n=== 6.3  Channel decomposition ===\n")
    print(f"  {'graph':24s} {'kappa~':>8s} {'1/kappa~':>9s} {'G_loc':>8s} "
          f"{'G_stat':>8s} {'residual':>9s} {'transit':>8s}")
    res = {}
    for name, G in graphs.items():
        c = build(G, with_core=True)
        kt = kappa_tilde(c)
        g1, _ = optimise_score(c, feature_sets(c)["F1 deg"], rng=np.random.default_rng(17))
        res[name] = dict(kappa=kt, ceiling=1 / kt, G_loc=G_loc(c), G_stat=g1,
                         residual=1 / (kt * g1), transitivity=nx.transitivity(G),
                         n=c["n"], m=c["m"], tri=c["T"])
        r = res[name]
        print(f"  {name:24s} {kt:8.4f} {1/kt:9.2f} {r['G_loc']:8.1f} "
              f"{g1:8.3f} {r['residual']:9.2f} {r['transitivity']:8.4f}")
    return res


def exp_features(graphs, quick=False):
    """Section 6.4: binning resolution and feature admissibility."""
    print("\n=== 6.4  Binning resolution and feature admissibility ===\n")
    KS = [4, 12, 24, 64] if quick else [4, 8, 12, 16, 24, 32, 48, 64]
    print(f"  {'graph':24s} " + " ".join(f"K={k:<6d}" for k in KS) +
          f" {'F1':>7s} {'F2':>7s} {'F3*':>7s}")
    res = {}
    for name, G in graphs.items():
        c = build(G, with_core=True)
        vals = []
        for K in KS:
            tj, nj = bin_profiles(c, degree_bins(c, K), K)
            u = fit_bins(tj, nj)
            vals.append(float(M1_numerator(c) / bin_C(u, tj, nj).sum()))
        fs = feature_sets(c)
        cont = {k: optimise_score(c, v, rng=np.random.default_rng(17))[0]
                for k, v in fs.items()}
        res[name] = dict(K=KS, binned=vals, **{f"cont_{k.split()[0]}": v
                                               for k, v in cont.items()})
        print(f"  {name:24s} " + " ".join(f"{v:<8.4f}" for v in vals) +
              f" {cont['F1 deg']:7.4f} {cont['F2 deg+core']:7.4f} "
              f"{cont['F3 +nodetri (INADMISSIBLE)']:7.4f}")
    print("\n  F3 is INADMISSIBLE (node triangle counts are the problem itself);")
    print("  it is reported only to quantify what an unchecked feature set buys.")
    return res


def exp_suffix(graphs, quick=False):
    """Sections 6.5-6.6: one-pass ceiling, closed form vs simulation, skew."""
    print("\n=== 6.5-6.6  One pass ===\n")
    reps = 8 if quick else 40
    print(f"  {'graph':24s} {'kappa~':>8s} {'k~_suf cf':>10s} {'E[N]/E[D]':>10s} "
          f"{'err%':>7s} {'rise':>6s} {'sd(N)/E[N]':>11s} {'PR/m':>8s}")
    res = {}
    for name, G in graphs.items():
        c = build(G, with_edge_ids=True)
        kt = kappa_tilde(c)
        cf = suffix_closed(c)
        s = suffix_simulate(c, reps=reps, per_edge=True)
        rom = s["N"].mean() / s["D"].mean()
        rat = s["N"] / s["D"]
        res[name] = dict(kappa=kt, kappa_suf_cf=cf, rom=rom,
                         err_pct=100 * abs(cf - rom) / rom,
                         rise=kt / cf, relsd_N=float(s["N"].std() / s["N"].mean()),
                         sd_ratio=float(rat.std() / rat.mean()),
                         PR_over_m=participation_ratio(c))
        r = res[name]
        print(f"  {name:24s} {kt:8.4f} {cf:10.5f} {rom:10.5f} {r['err_pct']:7.3f} "
              f"{r['rise']:6.3f} {r['relsd_N']:11.4f} {r['PR_over_m']:8.4f}")

        # per-edge moments grouped by t(e): tests the lemma, not the aggregate
        t = c["t"].astype(float); S = c["Ssz"].astype(float)
        p2 = t / 3 + t * (t - 1) / 5
        px = t / 3 + 0.3 * t * (t - 1) + 0.25 * t * (S - t)
        use = t >= 2
        g = np.floor(np.log2(np.maximum(t[use], 1))).astype(int)
        for gg in np.unique(g):
            sel = g == gg
            if sel.sum() < 20:
                continue
            e2 = (s["m2"][use][sel].sum() - p2[use][sel].sum()) / p2[use][sel].sum()
            ex = (s["mx"][use][sel].sum() - px[use][sel].sum()) / px[use][sel].sum()
            print(f"      t in [{2**gg:4d},{2**(gg+1)-1:4d}]  edges={sel.sum():6d}  "
                  f"err E[t^2]={100*e2:+7.3f}%  err E[tS]={100*ex:+7.3f}%")
    print("\n  Closed form = E[N]/E[D] exactly, by linearity; simulation is the estimator.")
    return res


def exp_cost(graphs, quick=False):
    """Section 6.7: cost budget and the admissible region."""
    print("\n=== 6.7  Cost and the admissible region ===\n")
    print(f"  {'graph':24s} {'Dbar':>8s} {'mean t':>7s} {'residual':>9s} "
          f"{'theta*':>8s}  admissible")
    res = {}
    for name, G in graphs.items():
        c = build(G, with_core=True)
        kt = kappa_tilde(c)
        g1, _ = optimise_score(c, feature_sets(c)["F1 deg"], rng=np.random.default_rng(17))
        resid = 1 / (kt * g1)
        Db = Dbar(c)
        th = theta_star(resid, Db)
        adm = "yes" if (resid > 2 and th > 0.3) else ("marginal" if th > 0.09 else "no")
        res[name] = dict(Dbar=Db, mean_t=float(c["t_pos"].mean()),
                         residual=resid, theta_star=th, admissible=adm)
        print(f"  {name:24s} {Db:8.1f} {c['t_pos'].mean():7.2f} {resid:9.2f} "
              f"{th:8.4f}  {adm}")
    return res


def exp_metric(graphs, quick=False):
    """Section 6.8: hit-rate optimisation is inadmissible (Theorem 5.2)."""
    print("\n=== 6.8  The reported metric ===\n")
    K = 12
    print(f"  {'graph':24s} {'obj':9s} {'pbar':>8s} {'log10 sum C':>12s} {'rho':>8s}")
    res = {}
    for name, G in graphs.items():
        c = build(G)
        tj, nj = bin_profiles(c, degree_bins(c, K), K)
        bmax = 1 if quick else (2 if len(c["t_pos"]) > 100_000 else 3)
        r = {}
        for obj in ("cost", "hitrate"):
            cur = advice_curve(c, tj, nj, bmax, objective=obj, seed=5)
            r[obj] = cur
            b = max(cur)
            print(f"  {name:24s} {obj:9s} {cur[b]['pbar']:8.4f} "
                  f"{cur[b]['logC']:12.2f} "
                  f"{cur[b]['rho'] if obj=='cost' else float('nan'):8.4f}")
        b = max(r["cost"])
        blow = r["hitrate"][b]["logC"] - r["cost"][b]["logC"]
        res[name] = dict(cost=r["cost"], hitrate=r["hitrate"], blowup_log10=blow)
        print(f"  {'':24s} {'-> variance blow-up 10^' + f'{blow:.1f}':>40s}"
              f"   {'INADMISSIBLE' if blow > 3 else 'finite'}\n")
    return res


def exp_controls(graphs, quick=False):
    """Section 6.9: the three hypotheses of ours that these measurements killed."""
    print("\n=== 6.9  Negative controls ===\n")
    K, nperm = 12, (5 if quick else 20)
    print(f"  {'graph':24s} {'G_inf':>8s} {'null(b=inf)':>12s} {'1+K/4t':>8s} "
          f"{'G(b=0)':>8s} {'null(b=0)':>10s} {'cont null':>10s}")
    res = {}
    for name, G in graphs.items():
        c = build(G, with_core=True)
        b0 = degree_bins(c, K)
        tj, nj = bin_profiles(c, b0, K)
        num = M1_numerator(c)
        g_inf = float(num / C_infinity(tj, nj).sum())
        g_b0 = float(num / bin_C(fit_bins(tj, nj), tj, nj).sum())
        nulls_inf, nulls_b0 = [], []
        for r in range(nperm):
            rng = np.random.default_rng(9000 + r)
            bp = b0[rng.permutation(c["n"])]
            tp, np_ = bin_profiles(c, bp, K)
            nulls_inf.append(float(num / C_infinity(tp, np_).sum()))
            nulls_b0.append(float(num / bin_C(fit_bins(tp, np_), tp, np_).sum()))
        feats = feature_sets(c)["F1 deg"]
        _, th = optimise_score(c, feats, rng=np.random.default_rng(17))
        cnull, _ = score_null(c, feats, th)
        mt = float(c["t_pos"].mean())
        res[name] = dict(G_inf=g_inf, null_inf=float(np.mean(nulls_inf)),
                         pred=1 + K / (4 * mt), G_b0=g_b0,
                         null_b0=float(np.mean(nulls_b0)), cont_null=cnull)
        r = res[name]
        print(f"  {name:24s} {g_inf:8.4f} {r['null_inf']:12.4f} {r['pred']:8.3f} "
              f"{g_b0:8.4f} {r['null_b0']:10.4f} {cnull:10.4f}")
    print("\n  null(b=inf) > 1  : per-edge fitting overfits, order K/mean_t (Thm 5.5)")
    print("  null(b=0)  ~ 1   : a single global score vector does not overfit")
    print("  cont null  < 1   : a permuted score is WORSE than uniform, so the")
    print("                     null is not a floor and must not be a denominator")
    return res


def exp_tables(graphs, quick=False):
    """Emit the LaTeX tables of Sections 6.3 and 6.7."""
    ch = exp_channels(graphs, quick)
    co = exp_cost(graphs, quick)
    order = sorted(ch, key=lambda n: -ch[n]["kappa"])
    esc = lambda s: s.replace("_", r"\_")
    print("\n% ---- Table: channel decomposition ----")
    print(r"\begin{tabular}{lrrrrrr}\toprule")
    print(r"graph & $\tilde\kappa$ & $1/\tilde\kappa$ & $G_\loc$ & $G_\stat$ & "
          r"residual & transitivity\\\midrule")
    for n in order:
        r = ch[n]
        print(f"{esc(n)} & {r['kappa']:.4f} & {r['ceiling']:.2f} & {r['G_loc']:.1f} & "
              f"{r['G_stat']:.3f} & {r['residual']:.2f} & {r['transitivity']:.4f}\\\\")
    print(r"\bottomrule\end{tabular}")
    print("\n% ---- Table: cost budget ----")
    print(r"\begin{tabular}{lrrrrl}\toprule")
    print(r"graph & $\bar D$ & $\bar t$ & residual & $\theta^\ast$ & admissible\\\midrule")
    for n in order:
        r = co[n]
        print(f"{esc(n)} & {r['Dbar']:.1f} & {r['mean_t']:.2f} & {r['residual']:.2f} & "
              f"{r['theta_star']:.3f} & {r['admissible']}\\\\")
    print(r"\bottomrule\end{tabular}")
    return dict(channels=ch, cost=co)



# =============================================================================
#  Transfer to Tonic's estimator family  --  Section 7
# =============================================================================

TONIC_ALPHA, TONIC_BETA, TONIC_KFRAC = 0.05, 0.2, 0.10   # Boldrin & Vandin's own values


def captured_heaviness(c, score, h, reps=8, seed=100):
    """psi: fraction of total heaviness mass sum_e Delta(e) = 3T captured by the
    top-h edges under `score`.  Ties are broken uniformly at RANDOM -- never by
    Delta(e), which would smuggle the exact oracle back into the free score."""
    delta = c["t"].astype(float)
    score = np.asarray(score, float)
    total = delta.sum()
    vals = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        order = np.lexsort((rng.random(len(score)), -score))
        vals.append(delta[order[:h]].sum() / total)
    return float(np.mean(vals)), float(np.std(vals))


def tonic_threshold(p=0.1, pp=0.09, cc=1.5, rho=10.0):
    """Boldrin & Vandin Prop. 1: Tonic beats the unaugmented baseline if
    T^H/T^L exceeds this.  Decreasing in rho, with infimum 5/24 at their p,p',c."""
    num = 3 * ((1 / pp ** 2 - 1 / p ** 2) + cc * rho * (1 / pp - 1 / p))
    den = (1 / p - 1) * (3 + 4 * rho / cc)
    return num / den


def exp_tonic(graphs, quick=False):
    """Section 7: does the channel decomposition transfer to a deployed algorithm?"""
    print("\n=== 7  Transfer to Tonic's family ===\n")
    KS = [0.01, 0.10, 0.20] if quick else [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    inf_thr = tonic_threshold(rho=1e9)
    print(f"  Prop.1 threshold at rho=10: {tonic_threshold(rho=10):.4f}   "
          f"rho=100: {tonic_threshold(rho=100):.4f}   infimum: {inf_thr:.4f}\n")
    print(f"  {'graph':22s} {'h':>7s} {'psi_rand':>9s} {'psi_deg':>9s} {'psi_max':>9s} "
          f"{'share':>7s} {'TH/TL max':>10s} {'clears?':>8s}")
    res = {}
    for name, G in graphs.items():
        c = build(G)
        deg = c["deg"]
        mindeg = np.minimum(deg[c["E"][:, 0]], deg[c["E"][:, 1]])
        h = max(1, int(round(TONIC_KFRAC * c["m"] * (1 - TONIC_ALPHA) * TONIC_BETA)))
        p_rand = h / c["m"]
        p_deg, sd = captured_heaviness(c, mindeg, h)
        p_max, _ = captured_heaviness(c, c["t"].astype(float), h, reps=1)
        share = ((p_deg - p_rand) / (p_max - p_rand)) if p_max > p_rand else float("nan")
        thtl = p_max / (1 - p_max)
        sweep = []
        for f in KS:
            hh = max(1, int(round(f * c["m"] * (1 - TONIC_ALPHA) * TONIC_BETA)))
            d_, _ = captured_heaviness(c, mindeg, hh, reps=4)
            m_, _ = captured_heaviness(c, c["t"].astype(float), hh, reps=1)
            sweep.append(d_ / m_ if m_ > 0 else float("nan"))
        res[name] = dict(h=h, psi_rand=p_rand, psi_deg=p_deg, psi_deg_sd=sd,
                         psi_max=p_max, share=share, TH_TL_max=thtl,
                         kappa=kappa_tilde(c), budget_sweep=sweep,
                         clears_any_rho=bool(thtl > inf_thr))
        print(f"  {name:22s} {h:7d} {p_rand:9.4f} {p_deg:9.4f} {p_max:9.4f} "
              f"{share:7.3f} {thtl:10.4f} {str(thtl > inf_thr):>8s}")
    ok = [r["share"] for r in res.values() if r["share"] == r["share"]]
    print(f"\n  share of attainable lift taken by the FREE statistic: "
          f"[{min(ok):.3f}, {max(ok):.3f}]")
    print("  'clears?' = does even a PERFECT oracle meet Prop.1's sufficient condition")
    print("  for some rho?  Note Prop.1 is sufficient, not necessary.")
    return res



def exp_advice_null(graphs, quick=False):
    """Section 6.10: advice curve under the COST objective against a permutation
    null at every b.  The encoder sees T(e) in both arms, so fitting freedom
    contributes equally and cancels in the difference."""
    print("\n=== 6.10  Advice curve vs its null at every b ===\n")
    K = 12
    print(f"  {'graph':22s} {'b':>2s} {'G(b)':>8s} {'null(b)':>9s} {'sd':>7s} {'gap':>9s}")
    res = {}
    for name, G in graphs.items():
        c = build(G)
        big = len(c["t_pos"]) > 100_000
        bmax, nperm = (2, 2) if quick else ((3, 3) if big else (4, 5))
        b0 = degree_bins(c, K)
        tj, nj = bin_profiles(c, b0, K)
        real = [advice_curve(c, tj, nj, bb, objective="cost", seed=5)[bb]["G"]
                for bb in range(bmax + 1)]
        nulls = []
        for r in range(nperm):
            rng = np.random.default_rng(9000 + r)
            tp, np_ = bin_profiles(c, b0[rng.permutation(c["n"])], K)
            nulls.append([advice_curve(c, tp, np_, bb, objective="cost", seed=5)[bb]["G"]
                          for bb in range(bmax + 1)])
        nulls = np.array(nulls)
        gap = np.array(real) - nulls.mean(0)
        res[name] = dict(G=real, G_null=nulls.mean(0).tolist(),
                         null_sd=nulls.std(0).tolist(), gap=gap.tolist(),
                         mean_t=float(c["t_pos"].mean()),
                         slope=float(np.polyfit(np.arange(len(real)), gap, 1)[0]))
        for bb in range(bmax + 1):
            print(f"  {name:22s} {bb:2d} {real[bb]:8.4f} {nulls.mean(0)[bb]:9.4f} "
                  f"{nulls.std(0)[bb]:7.4f} {gap[bb]:9.4f}")
        print(f"  {'':22s}    null growth {nulls.mean(0)[-1]-nulls.mean(0)[0]:+.4f}  "
              f"(1/mean_t = {1/res[name]['mean_t']:.4f});  gap slope "
              f"{res[name]['slope']:+.4f}/bit\n")
    print("  The null's growth tracks 1/mean_t (Thm 5.5 in the b dimension).")
    print("  A positive, stable gap is a real advice channel; a shrinking gap means")
    print("  the codebook's fitting freedom has outgrown the signal.")
    return res



def exp_advice_rate(graphs, quick=False):
    """Section 6.11: the self-normalised advice curve.

    G(b) - G_null(b) is a defective comparison because the two arms saturate at
    different levels, so the difference turns negative regardless of signal.
    Normalising each arm by its own asymptote removes the mismatch:

        F(b) = (G(b) - G(0)) / (G(inf) - G(0))  in [0,1]

    No fitting enters the normaliser: G(inf) is the closed form of Thm 5.5.
    F measures the RATE at which advice bits convert into gain, not the size of
    the gain -- always read the printed `range` = G(inf) - G(0) alongside it.
    """
    print("\n=== 6.11  Self-normalised advice rate ===\n")
    K = 12
    print(f"  {'graph':22s} {'range':>8s} {'F_real':>8s} {'F_null':>8s} {'gap':>8s} "
          f"{'lam_r':>7s} {'lam_n':>7s} {'b_1/2':>7s}")
    res = {}
    for name, G in graphs.items():
        c = build(G)
        big = len(c["t_pos"]) > 100_000
        bmax, nperm = (2, 2) if quick else ((3, 3) if big else (4, 5))
        b0 = degree_bins(c, K)
        tj, nj = bin_profiles(c, b0, K)

        Gr = [advice_curve(c, tj, nj, bb, objective="cost", seed=5)[bb]["G"]
              for bb in range(bmax + 1)]
        Gr_inf = float(M1_numerator(c) / C_infinity(tj, nj).sum())
        dr = Gr_inf - Gr[0]
        Fr = [(v - Gr[0]) / dr if abs(dr) > 1e-9 else float("nan") for v in Gr]

        Fn = []
        for r in range(nperm):
            rng = np.random.default_rng(9000 + r)
            tp, np_ = bin_profiles(c, b0[rng.permutation(c["n"])], K)
            gn = [advice_curve(c, tp, np_, bb, objective="cost", seed=5)[bb]["G"]
                  for bb in range(bmax + 1)]
            gi = float(M1_numerator(c) / C_infinity(tp, np_).sum())
            d = gi - gn[0]
            Fn.append([(v - gn[0]) / d if abs(d) > 1e-9 else float("nan") for v in gn])
        Fn_m = np.nanmean(np.array(Fn), 0)

        def rate(F):
            b = np.arange(len(F), dtype=float)
            y = np.clip(np.asarray(F, float), -0.99, 0.999)
            ok = np.isfinite(y) & (y < 0.999) & (b > 0)
            return float(-np.polyfit(b[ok], np.log1p(-y[ok]), 1)[0]) if ok.sum() >= 2 else np.nan

        lr, ln = rate(Fr), rate(Fn_m)
        half = np.log(2) / lr if lr and lr > 0 else float("nan")
        res[name] = dict(range=dr, F_real=Fr, F_null=Fn_m.tolist(),
                         gap=Fr[-1] - Fn_m[-1], lam_real=lr, lam_null=ln, b_half=half)
        print(f"  {name:22s} {dr:8.4f} {Fr[-1]:8.4f} {Fn_m[-1]:8.4f} "
              f"{Fr[-1]-Fn_m[-1]:8.4f} {lr:7.4f} {ln:7.4f} {half:7.2f}")
    print("\n  gap > 0 means the feature arm converts bits into gain faster than the")
    print("  null arm converts bits into overfit.  b_1/2 = bits reaching half the")
    print("  attainable gain.  A large F over a tiny `range` is a fast climb to a low")
    print("  ceiling, not a strong result.")
    return res



def exp_scale(graphs, quick=False):
    """Section 6.12: are the reported quantities scale-stable?

    Every quantity here needs the EXACT per-edge count t(e), so the analysis
    cannot be scaled to the sizes these algorithms actually run on -- doing so
    would mean solving the problem they exist to approximate.  Two probes:

      A. stream prefixes  -- a prefix is a SPARSER graph, not a smaller one:
         t(e) falls like f^2 while |S(e)| falls like f, so kappa~(f) ~ f*kappa~(1).
         This probe does not bear on extrapolation; it is here to show why.
      B. generator scaling at fixed m/n -- this one does.
    """
    print("\n=== 6.12  Scale stability ===\n")

    print("  Probe A: stream prefixes (expect kappa~(f) ~ f * kappa~(1))")
    print(f"  {'graph':22s} {'f':>4s} {'m':>8s} {'mean_t':>7s} {'kappa~':>8s} {'ratio vs f=1':>13s}")
    FR = [0.2, 0.5, 1.0] if quick else [0.2, 0.4, 0.6, 0.8, 1.0]
    for name, G in graphs.items():
        E = [(u, v) for u, v in G.edges()]
        rng = np.random.default_rng(7)
        E = [E[i] for i in rng.permutation(len(E))]
        full = None
        for f in FR:
            H = nx.Graph(); H.add_edges_from(E[:int(f * len(E))])
            c = build(H); kt = kappa_tilde(c)
            if f == 1.0: full = kt
            print(f"  {name:22s} {f:4.1f} {c['m']:8d} {c['t_pos'].mean():7.2f} {kt:8.4f} "
                  f"{(full/kt if full else float('nan')):13.2f}")
        print()

    print("  Probe B: generator scaling at fixed m/n")
    NS = [3000, 10000, 30000] if quick else [3000, 10000, 30000, 60000]
    gens = {"BA(n,5)":      lambda n: nx.barabasi_albert_graph(n, 5, seed=2),
            "PowerlawClus": lambda n: nx.powerlaw_cluster_graph(n, 5, 0.6, seed=7),
            "ER(n,10/n)":   lambda n: nx.gnp_random_graph(n, 10.0 / n, seed=1)}
    res = {}
    for gname, fn_ in gens.items():
        print(f"  {gname}")
        print(f"    {'n':>7s} {'m':>8s} {'mean_t':>7s} {'kappa~':>8s} {'share':>8s}")
        row = []
        for n in NS:
            G = fn_(n); G.remove_edges_from(nx.selfloop_edges(G))
            c = build(G)
            kt = kappa_tilde(c)
            mindeg = np.minimum(c["deg"][c["E"][:, 0]], c["deg"][c["E"][:, 1]])
            h = max(1, int(round(TONIC_KFRAC * c["m"] * (1 - TONIC_ALPHA) * TONIC_BETA)))
            pr = h / c["m"]
            pd_, _ = captured_heaviness(c, mindeg, h, reps=4)
            pm_, _ = captured_heaviness(c, c["t"].astype(float), h, reps=1)
            sh = (pd_ - pr) / (pm_ - pr) if pm_ > pr else float("nan")
            row.append(dict(n=n, m=c["m"], mean_t=float(c["t_pos"].mean()),
                            kappa=kt, share=sh))
            print(f"    {n:7d} {c['m']:8d} {c['t_pos'].mean():7.2f} {kt:8.4f} {sh:8.4f}")
        res[gname] = row
        k = [r["kappa"] for r in row]
        print(f"    tail drift in kappa~: {(k[-1]-k[-2])/k[-2]:+.4f}\n")
    print("  A quantity still moving at the largest computable size is not a graph")
    print("  constant.  The THEOREMS are unaffected -- they hold at any kappa~ -- but the")
    print("  empirical headline is measured on these instances, not extrapolated.")
    return res



def kappa_identity(c):
    """Proposition 6.x:  sum_e t|S| = Lambda - sum_e t^2 - 6T,  Lambda = 2 sum_v d(v) tri(v).

    Follows from sum_{e incident to v} t(e) = 2 tri(v): each triangle at v has two
    edges at v.  Exact -- no approximation anywhere.  Returns the identity residual,
    the degree-triangle coupling rho_dt (= 1 when degree and local triangle count are
    uncorrelated), and the aggregate upper bound of Corollary 6.y.
    """
    E, deg, t = c["E"], c["deg"].astype(float), c["t"].astype(float)
    n, m = c["n"], c["m"]
    tri = np.zeros(n)
    np.add.at(tri, E[:, 0], t); np.add.at(tri, E[:, 1], t)
    tri /= 2.0                                   # tri(v) = triangles containing v
    Lam = 2.0 * float(np.sum(deg * tri))
    st2 = float(np.sum(t * t)); T = float(t.sum()) / 3.0
    S = deg[E[:, 0]] + deg[E[:, 1]] - t - 2.0
    lhs = float(np.sum(t * S)); rhs = Lam - st2 - 6 * T
    dbar = float(deg.mean())
    rho = float(np.sum(deg * tri) / (dbar * np.sum(tri))) if tri.sum() > 0 else np.nan
    den_nc = 6 * dbar * T - st2 - 6 * T          # rho_dt = 1
    return dict(residual=abs(lhs - rhs) / max(lhs, 1e-12),
                Lambda=Lam, sum_t2=st2, T=T, dbar=dbar, rho_dt=rho,
                kappa=st2 / rhs if rhs > 0 else np.nan,
                kappa_upper=st2 / den_nc if den_nc > 0 else np.nan)


def exp_kappa_id(graphs, quick=False):
    """Section 6.12: the exact degree-triangle identity for kappa~, and what it says
    about scaling.  The drift with n is carried entirely by rho_dt."""
    print("\n=== 6.12  Degree-triangle identity for kappa~ ===\n")
    print(f"  {'graph':22s} {'residual':>10s} {'rho_dt':>8s} {'kappa~':>8s} "
          f"{'upper bd':>9s} {'holds':>6s}")
    res = {}
    for name, G in graphs.items():
        c = build(G); k = kappa_identity(c)
        res[name] = k
        print(f"  {name:22s} {k['residual']:10.2e} {k['rho_dt']:8.3f} {k['kappa']:8.5f} "
              f"{k['kappa_upper']:9.5f} {str(k['kappa'] <= k['kappa_upper'] + 1e-9):>6s}")
    print("\n  residual is |lhs-rhs|/lhs for the identity; it should be ~1e-16.")

    print("\n  Scaling: the drift in kappa~ is carried by rho_dt, not by t or d alone.")
    NS = [3000, 10000] if quick else [3000, 10000, 30000]
    gens = {"BA(n,5)":      lambda n: nx.barabasi_albert_graph(n, 5, seed=2),
            "PowerlawClus": lambda n: nx.powerlaw_cluster_graph(n, 5, 0.6, seed=7),
            "WS(n,10,0.1)": lambda n: nx.watts_strogatz_graph(n, 10, 0.1, seed=3),
            "ER(n,10/n)":   lambda n: nx.gnp_random_graph(n, 10.0 / n, seed=1)}
    for g, fn_ in gens.items():
        print(f"\n  {g}")
        print(f"    {'n':>7s} {'T/n':>8s} {'sum t^2/n':>10s} {'rho_dt':>8s} {'kappa~':>8s}")
        for n in NS:
            G = fn_(n); G.remove_edges_from(nx.selfloop_edges(G))
            c = build(G); k = kappa_identity(c)
            print(f"    {n:7d} {k['T']/n:8.3f} {k['sum_t2']/n:10.3f} {k['rho_dt']:8.3f} "
                  f"{k['kappa']:8.5f}")
    print("\n  rho_dt ~ constant  => kappa~ is a graph constant (WS, ER)")
    print("  rho_dt grows with n => kappa~ falls (BA, PowerlawCluster)")
    return res


EXPERIMENTS = {
    "identities": exp_identities,
    "channels": exp_channels,
    "features": exp_features,
    "suffix": exp_suffix,
    "cost": exp_cost,
    "metric": exp_metric,
    "controls": exp_controls,
    "advice": exp_advice_null,
    "advice_rate": exp_advice_rate,
    "tonic": exp_tonic,
    "scale": exp_scale,
    "kappa_id": exp_kappa_id,
    "tables": exp_tables,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--graphs", default="all")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    print("Loading graphs...")
    graphs = load_graphs(a.graphs)
    if not graphs:
        print("No graphs loaded; aborting.")
        return 1
    print(f"{len(graphs)} graphs.\n")

    todo = list(EXPERIMENTS) if a.experiment == "all" else [a.experiment]
    out = {}
    for name in todo:
        t0 = time.time()
        r = EXPERIMENTS[name](graphs, quick=a.quick)
        if r is not None:
            out[name] = r
        print(f"\n[{name} finished in {time.time()-t0:.1f}s]")

    if a.json:
        def conv(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(str(type(o)))
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=conv)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
