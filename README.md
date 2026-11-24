# Best Arm Identification with Unknown Post-Action Context Distribution

Code of the experiments of the paper. Every experiment stores its raw results in `results/`, and `plot.py` draws
every figure and table from them, so figures can be changed without rerunning anything.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The stored results were made with Python 3.9 and the versions of `requirements.txt`.

## Figures and tables of the paper

| paper | file in `figures/` | script | instances |
|---|---|---|---|
| Figure 2 and Table 4 (tightness of the relaxation) | `bound_tightness.pdf`, `table_bound_tightness.tex` | `bound_tightness.py` | random and hard instances, drawn in the script |
| Figure 3 (GLR statistic, relaxations and thresholds along a run) | `glr_trace.pdf` | `glr_trace.py` | 2 and 9 |
| Figure 4 (sample complexity on the two instances of the main text) | `sample_complexity.pdf` | `simulate.py` | 4 and 5 |
| Figure 6 (heatmaps of F(w)) | `heatmaps.pdf` | `heatmaps.py` | 6 |
| Figure 7 (sample complexity on random instances) | `sample_complexity_grid.pdf` | `simulate.py` | 10 n + k |

`figures/tables.md` also has the mean and median stopping time, the error rate and the running time of every stored
run.

## Instances

`instances/instance{i}.json` holds `n`, `k`, `confidence` (delta), `mus` and `A`. `A` is given row by row: one row per
arm, with the probabilities of the k contexts (the transpose of the k x n matrix of the paper).

| index | n, k, delta | used for |
|---|---|---|
| 0 | 3, 2, 1e-2 | a small sample instance, the default of the scripts (every algorithm stops within a minute) |
| 2 | 3, 3, 1e-3 | Figure 3, left |
| 9 | 3, 3, 1e-9 | Figure 3, right: instance 2 with a smaller delta |
| 4, 5 | 3, 2, 1e-3 | Figure 4, the instances of eq. (10) |
| 6 | 3, 3, 1e-3 | Figure 6, the instance of eq. (33) |
| 10 n + k | n, k, 1e-3 | Figure 7, n in {4, 8, 12, 16, 20} and k in {2, 4, 6, 8}: 42, 44, ..., 206, 208 |

The Figure 7 instances have the rows of `A` drawn by `utils.context_initiator(n, k)` and `mus` uniform in [0, 1].
For each (n, k) we took the first `np.random.seed(s)`, s = 0, 1, ..., for which the two best arms are at least 0.05
apart and every entry of `A` is at least 0.005 (so that the initial exploration, which waits for every arm-context
pair, stays short). Instances 122 and 124 were drawn the same way before these two conditions were added (124 with
`np.random.seed(2)`) and satisfy them. `io_utils.add_instance_to_json` writes a new instance file.

## The code

| file | content |
|---|---|
| `lowerbound.py` | the closed-form relaxation `L_bound` (`L_i` of eq. (8), Theorem 7) and the naive relaxation `L_naive` (Lemma 24), the proven factors of `L_bound` (Theorems 8 and 9), and the convex sampling rule `optimal_w` |
| `optimization.py` | the non-convex GLR: `exact_GLR_k2` (closed form for k = 2, Proposition 5), `plain_GLR` (local solver from several starting points) and `reference_GLR` (exact for k = 2, the solver with 20 random starting points otherwise: the reference value of Appendix 15.2, not used by any algorithm), and the w-optimization `optimize_bfgs` used for k = 2 |
| `algorithms/` | `ASTS` (GTS-A), `MuSTS` (GTS-mu), `STS` (GTS-Unknown, `glr="lb"` or `glr="naive"`), `SGTS` (GTS-SG); `TS.py` has the G-tracking rule |
| `environment.py` | the bandit and the run loop (`Environment.run`) |
| `utils.py` | `C_exp` of Kaufmann and Koolen (2021) and `W_bar` of Tuynman and Degenne (2025), used in the thresholds |
| `simulate.py` | many runs of one algorithm on one instance (Figures 4 and 7) |
| `glr_trace.py` | Figure 3 |
| `heatmaps.py` | Figure 6 |
| `bound_tightness.py` | Figure 2 and Table 4 |
| `plot.py` | all figures and tables from the stored results |
| `run_all.sh` | every experiment of the paper |

Algorithm names used on the command line and in the result files:

| name | paper | stopping | sampling |
|---|---|---|---|
| `ASTS` | GTS-A | exact GLR (A known) | exact (convex) |
| `MuSTS` | GTS-mu | exact GLR (mu known) | approximate maximizer of `F_mu`: a search over the weight of the best arm, the other weights filled greedily |
| `STS-LB` | GTS-Unknown (ours) | `min_i L_i(t) > beta(t, delta)`; for k = 2 the exact GLR (closed form) | maximizer of `w -> min_i L_i(w, A w)` (convex); for k = 2 the exact GLR inside BFGS |
| `STS-LB-Naive` | GTS-Unknown (Naive) | `min_i L_naive_i(t) > naive_beta(t, delta)`, for every k | maximizer of `w -> min_i L_naive_i(w, A w)` (convex) |
| `SGTS` | GTS-SG | sub-Gaussian GLR, threshold of Tuynman and Degenne (2025) | exact (convex) |

All algorithms recompute the optimal `w` only every `10^(d-2)` rounds once `t` has `d` digits (`--no_fast` turns this
off) and check their stopping rule at every round.

## Running the experiments

`run_all.sh` has every experiment, and every command can be stopped and run again: `simulate.py` keeps one entry per
run (seed = run index) in its result file and only runs the missing seeds.

```bash
./run_all.sh light               # Figures 2, 3 and 6 and Table 4, about 5 minutes on a laptop
JOBS=32 ./run_all.sh complexity  # Figure 4, 100 runs per algorithm
JOBS=32 ./run_all.sh grid        # Figure 7, 20 runs per algorithm
python plot.py all               # figures/ and figures/tables.md
```

`job.sh` is an example of a SLURM job that calls `run_all.sh`.

Single experiments (without arguments, `simulate.py`, `glr_trace.py` and `heatmaps.py` run on the sample instance 0):

```bash
python simulate.py                                                  # one run of GTS-Unknown (ours) on instance 0
python simulate.py --algorithm STS-LB --instance_index 4 --runs 100 --jobs 8
python simulate.py --algorithm ASTS SGTS --instance_index 42 44 --runs 20 --jobs 8
python glr_trace.py --instance_index 2 --log_period 40 --jobs 8
python heatmaps.py --instance_index 6 --div 61 --jobs 8
python bound_tightness.py --jobs 8
```

`simulate.py` also takes several algorithms and instances (`--algorithm STS-LB ASTS --instance_index 4 5`), all
their missing runs then share one pool, the slowest algorithm first. Do not run the same (instance, algorithm) in two
commands at the same time: both would write the same result file. To stop the runs, use Ctrl-C (or
`pkill -f simulate.py`; on macOS the workers are `multiprocessing.spawn` processes and need to be killed too).

Every run is seeded, and running the same seed twice on one machine gives the same stopping time. On another machine
or with other versions of the solvers, the convex programs of the sampling rule are solved to slightly different
values, so single runs can stop at other times than the stored ones.

### The GLR trace on another instance

1. Write `instances/instance11.json` (any unused number, see [Instances](#instances) for the format). All entries of `A`
   must be positive, its rows sum to 1, `"mus"` is in [0, 1] and the best arm is unique.
2. Run the trace, then draw it (the paper figure `glr_trace.pdf` is not touched):

```bash
python glr_trace.py --instance_index 11 --log_period 20 --jobs 8
python plot.py trace 11          # figures/glr_trace_11.pdf
```

`glr_trace.py` prints when each statistic crosses each threshold. The run continues until both lower bounds crossed the
naive threshold, so a harder instance (smaller gap, smaller delta) runs longer; keep about 1000 to 2000 logged rounds
with `--log_period` (the run length divided by 1500). `--delta` overrides the delta of the instance, `--seed` changes
the sample path, and `--reference_starts` (20 by default) is the number of random starting points of the reference GLR
for k >= 3 (for k = 2 it is the exact closed form). On the plot the filled circles and vertical lines are the stopping
times of the methods (`L` at our threshold, `L_naive` at the naive threshold), the open circles where each bound
crosses the other threshold. When all panels have the same n and k (the same instance with several deltas, as in
the paper), they share the y scale so that the thresholds can be compared. With 8 cores it takes about 1 min for
instance 2.

### Expected runtimes

Measured on one core of a laptop (Apple M5, all cores busy), a server core can be up to 2x slower. `figures/tables.md`
has the measured time of every stored run.

| algorithm | time per run | |
|---|---|---|
| `ASTS`, `SGTS`, `STS-LB`, `STS-LB-Naive` | 1 s to 13 min (`STS-LB` on instance 208; seconds for `STS-LB` with k = 2) | |
| `MuSTS` | 0.5 min to 1.5 h (instance 208) | almost all of it is the stopping rule: n - 1 convex programs per round |

| experiment | server CPU-hours |
|---|---|
| `complexity` (100 runs) | about 4 |
| `grid` (20 runs) | about 415 (365 of it `MuSTS`, up to 5 h per run on instance 208) |

With 72 cores `grid` takes at least the longest `MuSTS` run: those of the largest instances start first and set the end.

## Results

| directory | content |
|---|---|
| `results/simulation/instance{i}_{algorithm}.json` | one entry per run: seed, stopping time `T`, recommended arm, correctness, runtime (total, sampling, stopping) |
| `results/glr_trace/instance{i}.json` | Figure 3: every statistic along one run |
| `results/heatmaps/instance{i}.json` | Figure 6: `F(w)` on a grid of the simplex for the four settings |
| `results/bound_tightness/pairs.json` | Figure 2 and Table 4: the reference value of `F_i` (`Lambda`), `L_i`, `L_naive_i` and the proven factor of every pair (i, i*) |

## License

The code is released under the MIT License, see [LICENSE](LICENSE).
