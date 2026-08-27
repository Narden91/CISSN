# Superseded artifacts

Runs here are kept for provenance and for reuse as **diagnostics**. None may enter a
publication table, and none may be pooled with artifacts produced under the current code.
Nothing here was deleted: `docs/methodology.md` and `RUNBOOK.md` cite these paths as the
origin of published numbers, and those citations must keep resolving.

Audited 2026-08-27 on branch `conformal-per-cell-geometry`.

| Directory | Was | Why it cannot be used |
| --- | --- | --- |
| `headroom/` (3 seeds) | ETTh1-h336 RevIN CISSN runs + `conditioning_headroom_revin.json` | `git_dirty: true` at commit `d749709`, i.e. produced **before** the conditioning fitting-set fix. Cluster SCCP's partition was fit on train states (~6481 windows) while `sigma` was fit on the calibration half (~696) — a ~9x asymmetry that confounds every cluster-vs-scale ordering measured from them. |
| `percell/` (1 seed) | ETTh1-h336 RevIN, `--conformal_conditioning scale --scale_geometry per_cell` | Same commit, same `git_dirty: true`, same asymmetry. The per-cell number `3.6916` comes from here. |
| `CISSN_ETTh1_h336_seed42_preRevIN/` | the original Step 3 end-to-end run, plus its `conditioning_headroom.json` | Two independent problems: it is pre-RevIN (amplitude collapse, variance ratio ~0.07, test MSE 1.374), and its `metrics.json` carries only `interval` — no `interval_flat_cp`, `interval_cluster_cp`, or `interval_state_scaled` — so it predates the three-mechanism reporting contract entirely. |

Common to all three: `protocol.json` has keys `config/dataset/protocol/protocol_hash/source`
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
fresh artifacts. See `commands.md` at the repository root.
