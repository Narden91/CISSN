# CISSN on the cluster

Runs R1-R6 of `experiments.md` as Slurm GPU array jobs. R7 (analysis) runs on the workstation.
All commands run from the repository root, in bash. Never `chmod` the scripts and never edit a
tracked file on the cluster: `--require_clean_git` rejects a dirty tree, and a mode change counts.

## 1. On the workstation, once

```powershell
git add slurm/
git commit -m "Add Slurm scripts"
git push                  # the branch you run: conformal-per-cell-geometry
```

Do not change `cissn/` or `experiments/` after this commit. Every run records the commit.

## 2. On the cluster login node, once

```bash
cd <your scratch folder>
git clone <repo-url> CISSN && cd CISSN && git checkout conformal-per-cell-geometry

# uv, if missing (login nodes have internet, compute nodes do not):
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

bash slurm/setup_cluster.sh        # builds .venv from uv.lock (about 3 GB of torch)
```

`data/` and `results/` are gitignored, so a clone has neither. Create them, then copy from the workstation
(PowerShell does not expand `*` for `scp`; the names below are exact):

```bash
mkdir -p data results/publication          # on the cluster, in CISSN/
```

```powershell
# Data, byte for byte: sealed runs need exact checksums, so do not re-download.
scp -r data/ETT data/electricity.csv data/weather.csv data/exchange_rate.csv <user>@<login-host>:<scratch>/CISSN/data/

# R1 and R2 are already done on the workstation (cells.py status shows 12/12 and 2/2). Copy them so the
# cluster does not rerun them. A rerun would put a second, different-hash run of each weather-h192 cell in
# results/publication, and the cluster R1 would replace the record of the run that chose R3 or R5.
scp -r results/selection <user>@<login-host>:<scratch>/CISSN/results/
scp -r results/publication/CISSN_weather_M_sl96_pl192_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed42__1cc2e8e45bee results/publication/CISSN_weather_M_sl96_pl192_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed456__025daa56de32 <user>@<login-host>:<scratch>/CISSN/results/publication/
```

With these copies, `bash slurm/launch.sh` submits only r4, r4b, r5 and r6 (R1 says RevIN wins, so R3 is
excluded). To rerun R1 and R2 on the cluster instead, skip the last two `scp` lines.

## 3. Check and launch

```bash
bash slurm/preflight.sh            # read-only; fix every FAIL line
bash slurm/launch.sh --dry-run     # prints the sbatch commands
bash slurm/launch.sh               # preflight again, then submits
```

`launch.sh` first submits a 1-epoch GPU smoke job. Every block waits for it (`afterok`). If it fails, the
blocks stay pending: read `logs/slurm/cissn_smoke_*.out`, fix the cause, and cancel them (the command is
printed by `launch.sh`). If the log ends with exit 5 ("torch cannot see a GPU"), compare `nvidia-smi` with the
driver that CUDA 12.8 wheels need. The fix changes `uv.lock`, so it needs a new commit and a new clone.

| Block | Cells | Tasks | After |
| --- | --- | --- | --- |
| r1 RevIN selection | 12 | 4 | smoke |
| r2 CISSN weather h192 | 2 | 2 | smoke |
| r3 deep_ensemble gaps | 14 | 4 | r1, and only if RevIN loses |
| r4 ECL confirmatory | 9 | 9 | smoke |
| r4b ECL hybrid vs base | 1 | 1 | r4 |
| r5 baselines with RevIN | 216 | 12 | r1, and only if RevIN wins |
| r6 ablations ETTh1-h336 | 3 (x6 ablations) | 3 | r2-r5 |

## 4. Watch, resume

```bash
squeue -u $(id -un)
.venv/bin/python slurm/cells.py status        # done/total per block
cat logs/slurm/failed_*.txt                   # failed cells: job id, cell, exit code
bash slurm/launch.sh                          # resume: skips finished cells and empty blocks
```

Resume only when `squeue` shows no `cissn_*` job: preflight refuses to start while one is queued.

A cell that hits the time limit leaves a partial checkpoint. The next run removes it before it retries.
A failed cell never stops the other cells of its task. A structural failure (`structural_passed: false`) is
rerun once at the same seed (`docs/prespecification.md`, section 3): run `launch.sh` again.

## 5. Bring the results back (workstation)

```bash
rsync -av --ignore-existing --exclude '.*.tmp-*' --exclude 'smoke/' <user>@<login-host>:<scratch>/CISSN/results/ results/
```

The excludes skip the partial roots that a timed-out cell leaves behind.

Then run R7 from `experiments.md`, with these changes:

- Skip the `paired_point_comparison.py` line. Block r4b wrote `results/publication/hybrid_vs_base_ECL.json` on the GPU
  that trained the checkpoints; a reload on another GPU can exceed the drift tolerance.
- After R5, do the "move non-RevIN baselines" step and the `results/superseded/README.md` row locally.

Record the commit and the GPU (`nvidia-smi -L` is at the top of each log) next to every cluster result.
R2-R5 cells come from a different GPU than the 52 CISSN and 256 baseline cells run on the workstation.

## Settings

`slurm/env.sh` holds the account (`FUPB2_MEATIC`), partition (`boost_fua_prod`), GPU request, memory, tasks per
block and time limits. Override with an environment variable, e.g. `ACCOUNT=X PARTITION=Y bash slurm/launch.sh`.
`preflight.sh` asks the scheduler whether the job shape is accepted (`sbatch --test-only`).
