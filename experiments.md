# Historical CISSN training campaign (closed)

Every block of the historical training campaign has finished. One new block is open: `p5`,
Solar DLinear and PatchTST-lite (12 runs, `plan.md` Phase 5). What to run now is in
[HPC_JOBS.md](HPC_JOBS.md).

| Block | What | Evidence on disk (checked 2026-11-16) |
| --- | --- | --- |
| R1 to R7 | Selection, main grid, ablations, analysis | `results/selection`, `results/publication`, `RUNBOOK.md` status table |
| `r5` | 12 baseline reruns after the batch-of-one encoder fix (`6adc24b`) | 216 `BASELINE_*fullrevin__*` runs outside ECL |
| `p1` | DLinear, PatchTST-lite, MC-Dropout, Deep Ensemble on ECL | 24 `BASELINE_*_ECL_*fullrevin__*` runs |
| `p2`, `p3b`, exposed ACI | ACI against flat CP | `aci_vs_flat_ECL.json`, `aci_vs_flat_solar.json`, `aci_vs_flat_exposed.json` |
| `p3s`, `p3` | Second confirmatory test on Solar | 6 `CISSN_solar_*` runs |

The R5 baselines span two commits: `6711026` and the later rerun commit with the encoder fix.
The manuscript provenance table states both.

The old commands are in git history:

- Step 7 blocks, `pack_small` transfer helper, Solar paired analysis: `git show 75cbfae:experiments.md`.
- R1 to R7: `git show a1add62:experiments.md`.

Rules that still bind any new training launch:

- Confirmation runs need `--evidence_role confirmation --immutable_artifacts --strict_determinism --require_clean_git`. `slurm/cells.py` sets them.
- Do not change code, pull, or switch branch on the cluster while jobs run.
- Run on H100 nodes, like the earlier R5 runs. H100 and RTX 5080 results differ beyond floating-point noise under the same seed.
- A new experiment needs a `RUNBOOK.md` step before it gets a launch command.
