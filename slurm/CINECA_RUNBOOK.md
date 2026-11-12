# Cineca Leonardo runbook (copy and paste)

Runs the CISSN cluster jobs with the scripts already in `slurm/`. Fill the editable fields in `slurm/env.sh` once; every launcher reads them automatically.
Blocks marked **[workstation]** run in PowerShell on your PC. Blocks marked **[cluster]** run in bash on the login node. Facts are marked **verified** (source, date) or **not verified**.

## 0. Values to set once in the script

Edit these fields at the top of `slurm/env.sh` on the workstation:

```bash
ACCOUNT=""        # required: your project account
PARTITION=""      # required: your GPU partition
QOS=""            # optional: blank uses the scheduler default
MAIL_USER=""      # optional: blank disables email
PAPER_TIME="06:00:00"
```

No separate configuration file or account exports are needed. Keep these non-secret
script edits committed before historical training runs, which require a clean tree.
The paper follow-up records source hashes and accepts uncommitted changes.

**[workstation]** Only if copying files between machines, fill these transfer destinations:

```powershell
$env:CU = "<your Cineca user name>"; $env:LOGIN = "<login host>"; $env:WORKDIR = "<same work folder>"
```

Never paste a password, a 2FA code, an SSH key or a token into the chat, a script or this file.

## 1. First time on the cluster

**[cluster]** Skip section 1 if `CISSN/` already exists on the cluster.

1.1 Clone into your chosen work folder and build the environment (login nodes have internet, compute nodes do not):

```bash
cd <your-work-folder>
git clone <repo-url> CISSN && cd CISSN && git checkout conformal-per-cell-geometry
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
bash slurm/setup_cluster.sh        # builds .venv from uv.lock; expect: torch 2.x built for CUDA 12.8
```

1.2 Data and frozen weights. `data/`, `results/` and `*.pth` are gitignored, so a clone has none of them.

```bash
mkdir -p data results/publication checkpoints/publication
```

**[workstation]** Copy byte for byte (sealed runs check SHA-256). Login hosts such as `login01-ext.leonardo.cineca.it` are listed in the Leonardo guide (verified, 2026-11-12). 2FA applies to `scp` too (verified).

```powershell
scp -r data/ETT data/electricity.csv data/weather.csv data/exchange_rate.csv data/solar_AL.txt "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/data/"
```

The ECL and Solar CISSN weights and the RevIN baseline weights exist only on the machine that trained them (`manuscript/tables/tab_provenance.tex` names it). Copy `results/publication` and `checkpoints/publication` from that machine into the same paths. Preflight lists every missing file.

## 2. Every session

**[cluster]**

```bash
cd <your-work-folder>/CISSN
git status --short                 # empty = clean. Then:  git pull --ff-only
```

If you changed code on the workstation, the user commits and pushes first (git is read-only for the agent). `slurm/` files must be committed and use LF line endings; `bash slurm/preflight.sh` checks both.

## 3. Paper follow-up jobs (ECL, solar): no training, 1 GPU each

The settings come from `experiments/paper_suite.yaml` (strides ECL 23, solar 25; alphas 0.05, 0.10, 0.20). Follows `HPC_JOBS.md`.

```bash
bash slurm/paper_preflight.sh --datasets ECL,solar          # last line must be: Paper preflight OK
bash slurm/launch_paper.sh --datasets ECL,solar --dry-run   # expect two sbatch lines, one per dataset
bash slurm/launch_paper.sh --datasets ECL,solar             # submits; prints one job id per dataset
```

- A good log ends with `End UTC:`. Logs: `logs/slurm/cissn_paper_ECL_<jobid>.out` and `..._solar_<jobid>.out`.
- Time limit is `PAPER_TIME` in `slurm/env.sh` (default `06:00:00`; not a measured runtime). Edit that field directly to change it.
- Resubmit only a failed dataset: `bash slurm/launch_paper.sh --datasets ECL`. Finished prediction caches are reused. The launcher refuses a dataset that is already queued.

After both jobs succeed:

```bash
.venv/bin/python slurm/paper_jobs.py merge --datasets ECL,solar --output results/paper_cluster
.venv/bin/python manuscript/analysis/build_strengthening.py --results results/paper_cluster --prefix cluster
```

Expect `summary.csv`, `runs.csv`, `uncertainty.csv`, `phase_effects.csv`, `inventory.json` in `results/paper_cluster/`. The merge stops on incomplete runs. Do not edit a result in place; rerun under a new `--output`.

## 4. Training blocks (R1 to R6, P1 to P3): only when a block is open

Blocks and cell counts are in `slurm/README.md` (section 3). The launcher runs preflight, a 1-epoch smoke job, then the blocks; every block waits for the smoke job (`afterok`).

```bash
.venv/bin/python slurm/cells.py status                  # done/total per block
bash slurm/launch.sh --blocks p1,p2 --dry-run           # print the sbatch commands
bash slurm/launch.sh --blocks p1,p2                     # submit
```

- A bare `bash slurm/launch.sh` submits the blocks still open in `experiments.md`. Finished blocks need `--blocks`.
- Resume by running the same command again: finished cells are skipped, and partial checkpoints of unfinished cells are removed first.
- **Gap:** `slurm/cells.py` has no Solar baseline block (DLinear, PatchTST-lite). `plan.md` Phase 5 adds it as a copy of `block_p1`. Until then, Solar baselines cannot be launched.
- Sealed runs (`--require_clean_git --immutable_artifacts`) need a clean tree and byte-identical data.

## 5. Watch, resume, cancel

```bash
squeue -u "$(id -un)"                                    # my jobs
squeue --start -u "$(id -un)"                            # estimated start of pending jobs
sacct -u "$(id -un)" -S today -o JobID%14,JobName%22,State,Elapsed,MaxRSS,ExitCode
tail -f logs/slurm/<job-name>_<jobid>.out                # array jobs: <name>_<arrayjob>_<task>.out
cat logs/slurm/failed_*.txt                              # failed cells: job id, cell, exit code
scancel <jobid>                                          # one job
squeue -h -u "$(id -un)" -o '%i %j' | awk '$2 ~ /^cissn_/ {print $1}' | xargs scancel   # all cissn_ jobs
```

Preflight refuses to start while a `cissn_*` job is queued. Wait or cancel first.

## 6. Bring results back

**[workstation]** The two paper jobs write to `results/paper_cluster`. The training blocks write to `results/publication`.

```powershell
scp -r "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/results/paper_cluster" results/
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/manuscript/generated/*cluster*" manuscript/generated/
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/manuscript/tables/*cluster*" manuscript/tables/
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/manuscript/figures/cluster_*" manuscript/figures/
```

Record the commit and the GPU (`nvidia-smi -L` is at the top of each log) next to every cluster result.

## 7. If something fails

| Symptom | Cause | Fix |
| - | - | - |
| Preflight: `slurm/... is not committed` or `git tree not clean` | Local edits or an untracked file on the cluster | `git status`; commit on the workstation and `git pull --ff-only`, or add the file to `.gitignore`. A mode change counts as an edit. |
| Preflight: `CRLF line endings` | Windows line endings | Re-save with LF on the workstation, commit, pull. |
| Preflight: `virtual environment missing` | `setup_cluster.sh` not run | `bash slurm/setup_cluster.sh` on the login node. |
| Preflight: `torch is not a CUDA build` | Wrong wheel | Rerun setup; `uv.lock` pins `+cu128`. |
| Preflight: `Missing frozen weights` or `dataset SHA-256 disagrees` | Weights or data not copied byte for byte | Copy again from the machine that trained them (section 1.2). |
| Preflight: `scheduler rejects the job shape` | Wrong account, partition, QOS, or memory | Check `saldo -b`, then edit the account, partition, QOS, or resource fields in `slurm/env.sh`. |
| Preflight: not enough disk space | Cache estimate above free space | Free space or move the checkout to a larger work area. The estimate prints as `INFO cache estimate`. |
| Task exit 3 / 4 / 5 | Dirty tree / no venv / no GPU visible | Read the first lines of the log; `nvidia-smi -L` shows the node's GPUs. |
| Cell stopped at the time limit | Share per task too long | Raise `TIME_<BLOCK>` or `SHARDS_<BLOCK>` in `slurm/env.sh`, or `PAPER_TIME`; rerun resumes. |
| Job pending long | Queue load | `squeue --start`; select a suitable QOS directly in `slurm/env.sh`. |

## 8. Cineca facts

| Fact | Status |
| - | - |
| Booster node: 1 Xeon Platinum 8358, 32 cores, 4 A100 (64 GiB), 512 GiB RAM, no local disk. `$TMPDIR` on Booster is a 10 GB RAM disk. | Verified, docs.hpc.cineca.it/hpc/leonardo.html, 2026-11-12 |
| `boost_usr_prod` / `normal`: up to 64 nodes, 24 h. `boost_qos_dbg`: 30 min. `boost_qos_lprod`: 8 nodes, 4 days. | Verified, same page |
| Two-factor authentication is mandatory. No backup of `$HOME`. | Verified, same page |
| One GPU per job: `--gres=gpu:1 --cpus-per-task=8 --mem=64G` (8 cores per GPU = 32 / 4). | Ran in this project's scripts; the 1/4-node share is an inference |
| Compute nodes have no internet; build the venv on the login node and call `.venv/bin/python`. | From `slurm/env.sh` and `run_block.slurm`; consistent with Cineca practice, not checked in the docs |
| `saldo -b` lists budget per account. `$WORK`, `$CINECA_SCRATCH`, `$FAST` are the work areas. | Not verified (the docs page read did not cover them). Check with `saldo -b` and `echo $WORK $CINECA_SCRATCH $FAST`. |
| Login-node CPU time limit for long commands (relevant to `uv sync`). | Not verified. If `setup_cluster.sh` is killed, rerun it; downloads resume from the uv cache. |
| `slurm/env.sh` contains editable account, partition, QOS, email, and time fields. | Required fields start blank; optional blank flags are omitted. |
