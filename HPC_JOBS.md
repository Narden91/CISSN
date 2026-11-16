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
| A. Local follow-up scoring on ETTh2, Weather, Exchange | workstation | Open. `results/paper_strengthening_v2` does not exist yet. |
| B. Follow-up scoring on ECL and Solar | cluster | Open. Needs the ECL and Solar checkpoints, which exist only on the cluster. |
| D. Solar DLinear and PatchTST-lite training | cluster | Not launchable. `slurm/cells.py` has no Solar baseline block yet (`plan.md` Phase 5). |

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

B6. After both jobs succeed, merge and build. The merge stops if any run is incomplete.

```bash
.venv/bin/python slurm/paper_jobs.py merge --datasets ECL,solar --output results/paper_cluster
.venv/bin/python manuscript/analysis/build_strengthening.py --results results/paper_cluster --prefix cluster
```

Expect `summary.csv`, `runs.csv`, `uncertainty.csv`, `phase_effects.csv`, and `inventory.json` in
`results/paper_cluster/`.

## C. [workstation] Bring the results back

C1. Copy the merged results and the generated files. In the first line, replace the three
quoted values with your own; they are the only values you edit in this file.

```powershell
$env:CU = "<cineca user>"; $env:LOGIN = "<login host>"; $env:WORKDIR = "<work folder>"
scp -r "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/results/paper_cluster" results/
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/manuscript/generated/*cluster*" manuscript/generated/
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/manuscript/tables/*cluster*" manuscript/tables/
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/manuscript/figures/cluster_*" manuscript/figures/
```

Never copy whole run folders or `results/publication` from the cluster.

C2. Apply `plan.md` section 7 in order: integrity (7.1), control identities (7.2), then the
comparison table and outcome A, B, or C (7.3). Read no comparison before 7.1 and 7.2 pass.

C3. Record the commit of step A4 and the device next to the cluster results. GPU jobs print
`nvidia-smi -L` at the top of each log; CPU jobs print that CUDA is not required.

## D. [cluster] Optional: Solar DLinear and PatchTST-lite training

Not launchable yet. First, `plan.md` Phase 5 adds a Solar baseline block to `slurm/cells.py`
(12 runs, about 70 minutes of GPU time, estimate). After that block exists and is committed, launch it
with section 4 of `slurm/CINECA_RUNBOOK.md`. Then rerun step B4 with `--datasets solar`.
