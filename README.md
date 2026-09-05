# triangle-prediction-ceiling

Verification code for the paper

> **Localization, Free Statistics, and Prediction: A Tight Ceiling for Weighted Closing-Vertex Triangle Estimators**

Every theorem, lemma and empirical claim in the paper is checked by
[`triangle_prediction_ceiling.py`](triangle_prediction_ceiling.py). The file is
self-contained — no project imports, one dependency set.

---

## What the paper claims, and where it is checked

| Paper | Claim | Command |
|---|---|---|
| Thm 3.6 | The gain factors exactly as `G_total = G_loc · G_stat · G_pred'` | `identities` |
| Thm 3.7 | `G_pred ≤ 1/κ̃`, tight iff `q_e = Unif(T(e))` | `identities` |
| Lem 3.3 | `E[X] = T`, `E[X²] = (m/9)·Σ C_q(e)` | `identities` (Monte Carlo) |
| §6.3 | The three channels are independent axes | `channels` |
| §6.4 | `K=12` binning understates the channel; node-triangle counts are inadmissible | `features` |
| Lem 4.3–4.5 | Exact suffix moments; one-pass ceiling in closed form | `suffix` |
| §6.6 | Skew, and why simulation is not the reference | `suffix` |
| Thm 4.9 | Break-even predictor cost `θ* = Θ(1/t̄)`; the admissible region | `cost` |
| Thm 5.2 | Maximizing hit rate gives a biased, unbounded-variance estimator | `metric` |
| Thm 5.5 | Per-edge fitting overfits by order `K/t̄` | `controls` |
| §6.9 | The permutation null lies **below** uniform and is not a denominator | `controls` |
| §6.10 | The advice curve against a permutation null at every `b` | `advice` |
| §6.11 | Self-normalised advice rate; `b_1/2` and the null separation | `advice_rate` |
| §6.12 | Scale stability of `kappa~` and the free-statistic share | `scale` |
| §6.12 | Exact degree-triangle identity for `kappa~`; the drift is carried by `rho_dt` | `kappa_id` |
| §7 | The decomposition transfers to Tonic; captured heaviness `psi` | `tonic` |
| §7.5 | Residual channel measured by running Tonic's own binary | `tonic_three_arms.ipynb` |

## Install and run

```bash
pip install numpy scipy networkx
python triangle_prediction_ceiling.py all
```

Options:

```
--graphs real|synthetic|all|"<name>;<name>"   default all; separator is ";" because
                                              graph names contain commas
--json out.json                               dump raw results
--quick                                       smaller sweeps, for a smoke test
```

Examples:

```bash
python triangle_prediction_ceiling.py identities --graphs real
python triangle_prediction_ceiling.py suffix --graphs "oregon1;email-Enron"
python triangle_prediction_ceiling.py tables --json results.json
python triangle_prediction_ceiling.py tonic --graphs real
```

The `tonic` command reproduces Section 7: it computes, at Boldrin & Vandin's own
memory budget (`k=m/10, alpha=0.05, beta=0.2`, so `h=0.019m`), how much of the
total heaviness mass a free min-degree score places in the heavy-edge set versus
an exact oracle. Ties in the free score are broken uniformly at random, never by
the true heaviness — breaking them by heaviness would smuggle the oracle back in
and inflate the free score.

The six SNAP graphs are downloaded on first use and cached in `snap_cache/`.
If a download fails the run continues with whatever loaded, so the synthetic
families always work offline.

## Runtime

Roughly 20–40 minutes for `all` on the full graph set on a laptop; `--quick`
brings it under five minutes. The heaviest steps are the suffix simulation
(40 stream orders, one reverse sweep over the edge list each) and the advice
curves (non-convex alternating optimisation).

## Two implementation notes

**`S(e)` is never enumerated.** Both `C_q(e)` and `p_q(e)` depend on `q_e` only
through its restriction to `T(e)`, so the code enumerates `T(e)` — total size
`3T` — and otherwise uses only `|S(e)| = d(u)+d(v)-t(e)-2`. For binned scores,
the per-bin counts `n_j(e)` follow by inclusion–exclusion from per-vertex
neighbourhood bin counts. This is what makes exact evaluation feasible on
`email-Enron` (m = 183k, 3T = 2.2M).

**Normalise each arm by its own asymptote.** `advice` subtracts the permutation
null from the measured curve, which is a defective comparison: the two arms
saturate at different levels, so their difference turns negative regardless of
signal. `advice_rate` divides instead, using the exact closed form for
`G(inf)`, and three instances that looked flat or negative under subtraction
show a clear separation. Both commands are kept because the contrast is the
point.

**Closed forms are the reference; simulation is the estimator.** The one-pass
ceiling has an exact closed form because `Σ_e E[t_suf(e)²] = E[Σ_e t_suf(e)²]`
by linearity. On hub-dominated graphs the simulated value is right-skewed —
`sd(N)/E[N] = 0.23` on `oregon1` against `0.003` on `ca-HepTh` — so few-sample
runs agree with one another while all sitting below the mean. An early version
of this work read that agreement as evidence of a 25% error in the closed form.
It was not. `suffix` reports the dispersion alongside the value for this reason.

## Reproducing the paper's tables

```bash
python triangle_prediction_ceiling.py tables --graphs real
```

emits the LaTeX for the channel-decomposition and cost-budget tables directly.

## A limit worth knowing before you run this

Every quantity needs the **exact** per-edge triangle count. That is why nothing here
runs on billion-edge graphs: computing `t(e)` for every edge *is* the triangle
counting problem. `scale` measures how `kappa~` and the free-statistic share drift
over the range that is computable, and reports a trend that runs against the
paper's empirical headline. The theorems are unaffected — they hold at any
`kappa~` — but the measured numbers are properties of these instances.

## Running the deployed algorithm

Section 7.5 of the paper measures the residual prediction channel by running the
released Tonic binary (`github.com/VandinLab/Tonic`) with its predictor varied
and everything else fixed. That is not part of this repository — it needs their
C++ build — so it ships as a separate notebook, `tonic_three_arms.ipynb`, which
clones and builds their code, generates both predictor files, and reports
`Var[MinDegree]/Var[OracleExact]` with bootstrap intervals.

Two things it records that are worth knowing before you rerun it. Their binary
rejects `beta = 0`, and a small `beta` is *not* an inert predictor — the variance
falls monotonically as `beta` grows, with no plateau — so there is no way to get
an unaugmented baseline arm out of it, and only the residual channel is
measurable. And the estimate appears on stdout as `Estimated count T = ...`; the
declared output file is never created.

## Data

Graphs come from the [SNAP collection](https://snap.stanford.edu/data/):
`facebook_combined`, `ca-GrQc`, `ca-HepTh`, `p2p-Gnutella08`, `oregon1_010331`,
`email-Enron`. Synthetic families are generated by NetworkX with fixed seeds, so
every number in the paper's synthetic rows is reproducible without network
access.

## License

MIT.
