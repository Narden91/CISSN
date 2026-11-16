# What to run now: workstation and cluster

This is the only launch checklist. Run the parts in order: A, B, C. Part D is optional.
Every step says where it runs:

- **[workstation]**: PowerShell, repository root, `.\.venv\Scripts\python.exe`.
- **[cluster]**: bash on the Cineca login node, root of the cluster checkout, `.venv/bin/python`.

Reference material is elsewhere. Do not copy commands from those files:

- `slurm/CINECA_RUNBOOK.md`: one-time cluster setup, watch and cancel commands, failure table.
- `plan.md`: why each run exists, and the result checks (section 7) before any comparison is read.
- `RUNBOOK.md`: the protocol of the historical confirmatory runs.

## Status (2026-11-16)

| Work | Where | State |
| - | - | - |
| Historical training blocks R1 to R7, P1, P2, P3 | cluster | Done. Results are in `results/publication` (216 R5 baselines, 24 ECL baselines, 6 Solar CISSN runs, 3 ACI files). |
| A. Local follow-up scoring on ETTh2, Weather, Exchange | workstation | Done. 18 runs in `results/paper_strengthening_v2` (GPU, code of commit `f44da2e`). |
| B. Follow-up scoring on ECL and Solar | cluster | Done. 24 runs, CPU. Solar ran at commit `7117b83`, ECL at `f44da2e`; the two commits differ only in `HPC_JOBS.md`, `build_strengthening.py`, and a test. |
| C. Results on the workstation | workstation | Done. The merged report inputs are in `results/cluster_complete_20261116_161806/paper_cluster`. Checks 7.1 and 7.2 of `plan.md` pass; the reading rule gives outcome A on ECL and on Solar. |
| D. Solar DLinear and PatchTST-lite training, then Solar re-scoring | workstation GPU, then cluster CPU | Open. Block `p5` in `slurm/cells.py` (12 runs, about 4.6 h on the RTX 5080). Commands below. |
| E. Phase 6 statistics | workstation | Done 2026-11-16 (seconds, no cluster). Outputs in `results/phase6_statistics` and `manuscript/figures/origin_difference.pdf`. |

Result folders that the manuscript reads:

| Folder | Content | Table and macro prefix |
| - | - | - |
| `results/cluster_complete_20261116_161806/paper_cluster` | ECL (CISSN, DLinear, PatchTST-lite) and Solar (CISSN) | `cluster` |
| `results/paper_strengthening_v2` | ETTh2, Weather, Exchange (CISSN) | `strengthening` |

Regenerate the manuscript inputs with:

```powershell
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py --results results/cluster_complete_20261116_161806/paper_cluster --prefix cluster
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py --results results/paper_strengthening_v2
```

Removed on 2026-11-16 after a per-file hash check against the folder above:
`results/cluster_download_20261116_144016`, `results/cluster_download_ECL_20261116_154301`,
`results/paper_cluster` (a partial local cache), and `results/paper_cluster_ECL.tar.gz`. The first ECL
scoring (commit `a004960`, superseded by the `f44da2e` rerun, same summaries) is kept as
`results/cluster_complete_20261116_161806/superseded_first_scoring_ECL_a004960.tar.gz`.
Also removed on 2026-11-16: `results/paper_strengthening` (the October six-method follow-up; its
scores equal the v2 scores), `results/paper_strengthening_v2_timing`, the October snapshot
`results_14_10_2026` (every run in it also exists under `results/`; its `superseded/` diagnostics
are retired), and the `_*` scratch files in `results/`.

The follow-up scores 9 interval methods on frozen models, with no training: `flat_paired`, `flat_full`,
`phase`, `phase_coarse`, `state`, `phase_state`, `phase_coarse_state`, `flat_scaled`, and capped
delayed-feedback `aci`. Alphas 0.05, 0.10, 0.20; horizons 96 and 336; seeds 42, 123, 456.

## A. [workstation] Local scoring and commit

A1. Run the tests. Expect no failures.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_paper_suite.py tests/test_paper_jobs.py -q -p no:cacheprovider --basetemp "results/_pytest_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

The timestamp gives a new, unused `--basetemp` folder each time; pytest clears it.

A2. Score the three exposed datasets: 18 CISSN runs, about 30 minutes (estimate from one 102-second run).
The agent asks before it starts this run.

```powershell
.\.venv\Scripts\python.exe -m experiments.run_paper_suite --config experiments/paper_suite_v2.yaml
```

Output goes to `results/paper_strengthening_v2`. Never write into `results/paper_strengthening`.

A3. Apply checks 7.1 and 7.2 of `plan.md`. Then build the tables:

```powershell
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py --results results/paper_strengthening_v2
```

A4. Commit the code and the dated design text (`plan.md` Stop point 1). Push the branch
`conformal-per-cell-geometry`. Write down the commit hash. The cluster runs this commit.

## B. [cluster] Score ECL and Solar

B1. Update the checkout to the commit of step A4. Do not copy files by hand.

```bash
git status --short        # must print nothing
git pull --ff-only
git log -1 --oneline      # must show the commit of step A4
```

If this is the first session on Cineca, do section 1 of `slurm/CINECA_RUNBOOK.md` first.

B2. Put `ACCOUNT` and `PARTITION` in the untracked repo-root `.env` (`slurm/env.sh` loads it; do not
commit it). `QOS` and `MAIL_USER` may stay blank. Leave `GRES` empty so every job runs on CPU.
`PAPER_TIME` default is `24:00:00`, the `dcgp_fua_prod` maximum. `--gres=gpu:1` on that partition is
rejected as an unavailable node configuration.

B3. Check every input. The last line must be `Paper preflight OK`.

```bash
bash slurm/paper_preflight.sh --datasets ECL,solar
```

The check covers weights, dataset hashes, configurations, environment, disk space, and scheduler settings.
Fix each reported failure, then run it again.

B4. Preview, then submit. The preview must show two `sbatch` commands, one per dataset.

```bash
bash slurm/launch_paper.sh --datasets ECL,solar --dry-run
bash slurm/launch_paper.sh --datasets ECL,solar
```

| Job name | Models scored | Frozen runs | Test-origin stride |
| --- | --- | --- | --- |
| `cissn_paper_ECL` | CISSN, DLinear, PatchTST-lite | 18 | 23 |
| `cissn_paper_solar` | CISSN | 6 | 25 |

Write down the two job IDs. Do not edit code, pull, or switch branch until both jobs end.

B5. Watch the jobs.

```bash
squeue -u "$(id -un)"
tail -n 20 $(ls -t logs/slurm/cissn_paper_*.out | head -n 2)   # the two newest paper logs
```

A successful log ends with `End UTC:`. If one job fails, fix the cause in its log and resubmit only
that dataset, for example `bash slurm/launch_paper.sh --datasets ECL`. Complete prediction caches are
reused. The launcher refuses a dataset that is already queued.

CPU reloads can move an interval boundary across a target relative to the original GPU results.
The frozen-score check allows one cell of absolute coverage drift per origin
(`1 / (horizon * scored channels)`), or the existing `1e-4` relative tolerance, whichever is larger.
Width and Winkler retain the `1e-4` relative tolerance. Each `scores/<run>/reproduction_*.json`
records the tolerances, per-metric drift, and pass/fail result. Nonfinite scores still fail.
After updating this check, resubmit only the failed dataset; source changes automatically rebuild
its prediction caches. A CPU job accepts either a CPU-only or CUDA-enabled PyTorch build.

B6. After both jobs succeed, merge and package the report inputs. The merge stops if any run is incomplete.

```bash
.venv/bin/python slurm/paper_jobs.py merge --datasets ECL,solar --output results/paper_cluster
tar --exclude='cache' -czf results/paper_cluster_report.tar.gz -C results paper_cluster
```

Expect `summary.csv`, `runs.csv`, `uncertainty.csv`, `phase_effects.csv`, and `inventory.json` in
`results/paper_cluster/`.
The archive includes the per-dataset `scores/` directories needed for the report and excludes
the large prediction caches. Recreate it after rerunning or merging new results.

## C. [workstation] Bring the results back

C1. Download the archive into a fresh local directory and build the report on the laptop.
Use the updated report builder: it supports per-dataset score directories and needs no prediction
caches, datasets, or checkpoints. In the first line, replace the three quoted connection values.

```powershell
$env:CU = "<cineca user>"; $env:LOGIN = "<login host>"; $env:WORKDIR = "<work folder>"
$bundleDir = "results/cluster_download_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $bundleDir | Out-Null
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/results/paper_cluster_report.tar.gz" "$bundleDir/"
tar -xzf "$bundleDir/paper_cluster_report.tar.gz" -C "$bundleDir"
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py --results "$bundleDir/paper_cluster" --prefix cluster
```

Never copy whole run folders or `results/publication` from the cluster.

C2. Apply `plan.md` section 7 in order: integrity (7.1), control identities (7.2), then the
comparison table and outcome A, B, or C (7.3). Read no comparison before 7.1 and 7.2 pass.

C3. Record the commit of step A4 and the device next to the cluster results. GPU jobs print
`nvidia-smi -L` at the top of each log; CPU jobs print that CUDA is not required.

## D. Solar DLinear and PatchTST-lite: train, re-score, bring back

Block `p5` trains 12 runs (DLinear and PatchTST-lite, `H` 336 and 96, seeds 42, 123, 456) with RevIN,
the sealed flags, and Solar stride 25. The cluster account has no GPU, so the runs train on the
workstation RTX 5080 Laptop GPU. The CISSN Solar checkpoints exist only on the cluster, so the
re-scoring stays there, on CPU.

Measured on 2026-11-16, one epoch at `H=336`, seed 42, non-sealed probe: DLinear 31.7 s per epoch
plus 23 s test and about 13 s setup; PatchTST-lite 103.5 s per epoch plus 29 s test. The saved ECL
baselines all ran the full 20 epochs. Estimate with 20 epochs and `H=96` taken as no slower than
`H=336` (not measured): about 11 min per DLinear run and 35 min per PatchTST-lite run, so about
4.6 hours for all 12. Early stopping can only shorten this. The strict determinism flag can slow it.

D1. **[workstation]** Commit and push the working tree. The runs are sealed (`--require_clean_git`),
so `git status --short` must print nothing before D2. Include `slurm/cells.py`, `slurm/launch.sh`,
`slurm/env.sh`, `slurm/preflight.sh`, `slurm/paper_jobs.py`, and the new
`manuscript/analysis/plot_origin_difference.py`. Write down the commit hash.

D2. **[workstation]** Train the 12 runs on the local GPU. `GRES` makes each run require a GPU. The
command skips finished runs, so rerun the same line after an interruption.

```powershell
git status --short                                   # must print nothing
.\.venv\Scripts\python.exe slurm/cells.py list p5    # 12 cells, 12 to do
$env:GRES = "gpu:1"; .\.venv\Scripts\python.exe slurm/cells.py run p5 --task 0 --ntasks 1 *> "logs/p5_$(Get-Date -Format yyyyMMdd_HHmmss).log"
.\.venv\Scripts\python.exe slurm/cells.py list p5    # 12 cells, 0 to do
```

D3. **[workstation]** Pack the 12 run configurations (no `.npy` arrays) and the 24 weight files, then
upload them. In the first line, replace the three quoted connection values, as in step C1.

```powershell
$env:CU = "<cineca user>"; $env:LOGIN = "<login host>"; $env:WORKDIR = "<work folder>"
$runs = Get-ChildItem results/publication -Directory | Where-Object Name -match '^BASELINE_(dlinear|patchtst)_solar_'
$weights = Get-ChildItem checkpoints/publication -Directory | Where-Object Name -match '^BASELINE_(dlinear|patchtst)_solar_' | Get-ChildItem -File -Filter *.pth
"$($runs.Count) runs, $($weights.Count) weight files"   # expect 12 and 24
$paths = @($runs) + @($weights) | ForEach-Object { $_.FullName.Substring($PWD.Path.Length + 1) -replace '\\', '/' }
tar -czf results/p5_solar_baselines.tar.gz --exclude '*.npy' @paths
scp results/p5_solar_baselines.tar.gz "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/"
```

D4. **[cluster]** Update the checkout and unpack. `results/` is gitignored and so are `*.pth` files, so
the tree stays clean.

```bash
git status --short        # must print nothing
git pull --ff-only
git log -1 --oneline      # must show the commit of step D1
tar -xzf p5_solar_baselines.tar.gz && rm p5_solar_baselines.tar.gz
git status --short        # must still print nothing
.venv/bin/python slurm/cells.py list p5    # 12 cells, 0 to do
```

D5. **[cluster]** Re-score Solar with all three forecasters, on CPU (`GRES=""`,
`PARTITION="dcgp_fua_prod"` in `.env`, as before).

```bash
bash slurm/paper_preflight.sh --datasets solar
bash slurm/launch_paper.sh --datasets solar --dry-run
bash slurm/launch_paper.sh --datasets solar
tail -n 20 $(ls -t logs/slurm/cissn_paper_solar*.out | head -n 1)   # ends with End UTC:
```

D6. **[cluster]** Merge ECL and Solar and package the report inputs, as in step B6.

```bash
.venv/bin/python slurm/paper_jobs.py merge --datasets ECL,solar --output results/paper_cluster
tar --exclude='cache' -czf results/paper_cluster_report.tar.gz -C results paper_cluster
```

D7. **[workstation]** Download into a new folder and rebuild the tables, in the same PowerShell
session as D3 (it reuses the connection values).

```powershell
$bundleDir = "results/cluster_complete_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $bundleDir | Out-Null
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/results/paper_cluster_report.tar.gz" "$bundleDir/"
tar -xzf "$bundleDir/paper_cluster_report.tar.gz" -C "$bundleDir"
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py --results "$bundleDir/paper_cluster" --prefix cluster
```

Then the agent applies `plan.md` section 7 again. It updates the folder name in this file, `CLAUDE.md`,
and `manuscript/README.md`. It also updates the manuscript sentences that say no DLinear or PatchTST-lite
model exists for Solar (`sections/07_results_strong_forecaster_aci.tex`, `sections/08_discussion.tex`).

## E. [workstation] Phase 6 statistics

Prespecified statistic on the historical runs at block multipliers 1, 2, and 4, with the
ratio-of-means sensitivity interval beside it (key `ratio_of_means`). Multiplier 1 reproduces
`results/publication/paired_interval_inference_ECL.json` exactly. Runs in seconds.

```powershell
$o = "results/phase6_statistics"; New-Item -ItemType Directory -Force $o | Out-Null
foreach ($m in 1, 2, 4) { .\.venv\Scripts\python.exe scripts/paired_interval_inference.py --results_root results/publication --datasets ECL --block_multiplier $m --n_boot 5000 --seed 0 --output "$o/ECL_primary_m$m.json" }
foreach ($m in 1, 2, 4) { foreach ($meth in "primary", "state_scaled") { .\.venv\Scripts\python.exe scripts/paired_interval_inference.py --results_root results/publication --datasets solar --method $meth --block_multiplier $m --n_boot 5000 --seed 0 --output "$o/solar_${meth}_m$m.json" } }
.\.venv\Scripts\python.exe manuscript/analysis/plot_origin_difference.py
```
