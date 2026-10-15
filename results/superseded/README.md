# Superseded artifacts

Runs here are kept for provenance and for reuse as **diagnostics**. None may enter a
publication table, and none may be pooled with artifacts produced under the current code.
Nothing here was deleted: `docs/methodology.md` and `RUNBOOK.md` cite these paths as the
origin of published numbers, and those citations must keep resolving.

Audited 2026-08-27 (last updated 2026-10-13) on branch `conformal-per-cell-geometry`.

| Directory | Was | Why it cannot be used |
| --- | --- | --- |
| `headroom/` (3 seeds) | ETTh1-h336 RevIN CISSN runs + `conditioning_headroom_revin.json` | `git_dirty: true` at commit `d749709`, i.e. produced **before** the conditioning fitting-set fix. Cluster SCCP's partition was fit on train states (~6481 windows) while `sigma` was fit on the calibration half (~696) — a ~9x asymmetry that confounds every cluster-vs-scale ordering measured from them. |
| `percell/` (1 seed) | ETTh1-h336 RevIN, `--conformal_conditioning scale --scale_geometry per_cell` | Same commit, same `git_dirty: true`, same asymmetry. The per-cell number `3.6916` comes from here. |
| `CISSN_ETTh1_h336_seed42_preRevIN/` | the original Step 3 end-to-end run, plus its `conditioning_headroom.json` | Two independent problems: it is pre-RevIN (amplitude collapse, variance ratio ~0.07, test MSE 1.374), and its `metrics.json` carries only `interval` — no `interval_flat_cp`, `interval_cluster_cp`, or `interval_state_scaled` — so it predates the three-mechanism reporting contract entirely. |
| `pre_revin_scalar_grid/` (41 runs + 6 aggregate files) | exposed-dataset CISSN grid launched with `revin: false` and `scale_geometry: scalar` | Wrong regime: flat CP beats the primary mechanism on all four datasets, marginal coverage 0.72-0.83 against 0.90. Moved 2026-10-13 so no aggregation script reads it. Its results were read, so the ETTh2, weather, and exchange-rate test splits are exposed (see `docs/prespecification.md`). |
| `dlinear_cd8cb6a_duplicate/` (29 runs) | second `dlinear` launch at commit `cd8cb6a`, ETTh1 and ETTh2 | Duplicate of the complete `b8edcc9` set, which matches the other single-model baselines. Rule fixed before any metric was read. `generate_publication_tables.py` cannot choose between two copies. |
| `baselines_no_revin/` (216 runs) | `dlinear`, `patchtst`, `mc_dropout`, `deep_ensemble` without RevIN (workstation, `b8edcc9`/`efadb63`/`a1add62`/`6711026`) | R1 selected RevIN (`results/selection/r1_decision.json`), so R5 reran these four models with `--revin`. Moved 2026-10-15 as `experiments.md` R5 prescribes, so one table never mixes regimes. DeepState stays in `publication/`: it has no RevIN arm. |
| `weather_h192_h100_duplicate/` (2 runs) | second launch of R2 (CISSN weather h192, seeds 42 and 456) on the H100 at `6711026`, JSON only | Duplicate of the R2 runs on the workstation at `a1add62`. The workstation copies are kept: they are the first complete launch, and Windows never takes the compiled encoder path that `6adc24b` fixed. Rule fixed before the analysis. The primary Winkler of both copies was printed during the check (seed 42: 2.239 workstation, 2.268 H100; seed 456: 2.290, 2.251); the rule does not use these values. |

The directory `results/_superseded_bs1_bug/` (outside this folder) holds runs made on Linux at
`6711026`, before `6adc24b`, that sent a one-window batch through the compiled encoder: the 9
first ECL runs (test-origin stride path) and 12 R5 runs, `mc_dropout` and `deep_ensemble` on
ETTh1 and ETTh2 at h192, where the validation, calibration, and test splits (2689, 1537, 2689
windows) leave a final batch of 1 at batch size 128. Every other H100 cell leaves no batch of 1.
The ECL cells were rerun at `6adc24b`; the 12 R5 cells are rerun on the cluster.

Common to the first three: `protocol.json` has keys `config/dataset/protocol/protocol_hash/source`
and **no `evidence` block**, so every run here is a development artifact regardless of its
numbers.

## What these are still good for

- The pre-RevIN run against the RevIN runs in `headroom/` is the evidence that state
  conditioning is **regime-dependent**: cluster SCCP is `+0.151` Winkler vs flat CP on 0/4
  cuts pre-RevIN, and `-0.117` on 4/4 cuts under RevIN. That contrast survives the fitting
  fix, because both arms were measured under the same asymmetric scheme — the asymmetry is
  held constant across the comparison, so it cannot by itself produce a sign flip between
  regimes.
- The cluster-occupancy figure `[94, 482, 1958, 0, 11]` and the sparse-cluster fallback
  discussion in `docs/methodology.md` come from the pre-RevIN run.

## What they are not good for

The ordering flat `3.7869` < scalar `3.7877` < per-cell `3.6916` < cluster `3.5962` is
**confounded** and must be re-measured under the current code before any use. It survives
in `docs/methodology.md` only as retracted context. The fixed call site is
`run_benchmark.py:754-761`.

## Replacement

`RUNBOOK.md` Step 3 re-run under `--revin` on current code, then Step 3b.0 against those
fresh artifacts. See `experiments.md` at the repository root.
