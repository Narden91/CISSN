# Launch the remaining HPC jobs

Run this checklist in Bash from the existing CISSN checkout on the cluster.
It submits **two GPU jobs** using the existing frozen checkpoints.

| Job name | Point forecasts | Frozen runs | Test-origin stride |
| --- | --- | --- | --- |
| `cissn_paper_ECL` | CISSN, DLinear, PatchTST | 18 | 23 |
| `cissn_paper_solar` | CISSN | 6 | 25 |

Both jobs cover horizons 96 and 336, seeds 42/123/456, and alpha 0.05/0.10/0.20.
Each compares paired Flat, full Flat, Phase Mondrian, State SCCP, Phase+State,
and capped delayed-feedback ACI. These jobs cover the prepared follow-up analyses.
New datasets, additional conformal comparators, and matched-coverage analysis need
further work before they have launch-ready jobs.

## 1. Update the cluster checkout

Copy the new source files into their matching paths in the existing checkout:

```text
cissn/conformal/phase.py
experiments/run_paper_suite.py
experiments/paper_suite.yaml
slurm/paper_jobs.py
slurm/paper_preflight.sh
slurm/launch_paper.sh
slurm/run_paper.slurm
manuscript/analysis/build_strengthening.py
```

Keep the existing project environment, `slurm/env.sh`, datasets, publication
configurations, and checkpoints. Shell files must use LF line endings.
The scripts use the account and resource settings in `slurm/env.sh`.

## 2. Check all required files

```bash
bash slurm/paper_preflight.sh --datasets ECL,solar
```

Continue only when the command ends with `Paper preflight OK`.
Fix each reported missing file or failed check first. The checker verifies the
weights, dataset hashes, configurations, environment, disk space, and scheduler settings.

## 3. Preview the two jobs

```bash
bash slurm/launch_paper.sh --datasets ECL,solar --dry-run
```

Expect two `sbatch` commands: one for ECL and one for Solar.
Use `launch_paper.sh` for this follow-up. The older `launch.sh` runs the historical training workflow.

## 4. Submit

```bash
bash slurm/launch_paper.sh --datasets ECL,solar
```

Save the two job IDs printed by the launcher.
Keep the source and configuration files unchanged until both jobs finish.

## 5. Monitor and check completion

```bash
squeue -u "$(id -un)"
ls -lt logs/slurm/cissn_paper_*.out
```

Read the logs whose filenames contain the submitted job IDs:

- `logs/slurm/cissn_paper_ECL_<jobid>.out`
- `logs/slurm/cissn_paper_solar_<jobid>.out`

A successful log ends with `End UTC:` after checking every expected run completed.
If one job fails, fix its reported cause and resubmit only that dataset.
For ECL, use:

```bash
bash slurm/launch_paper.sh --datasets ECL
```

For Solar, replace `ECL` with `solar`. Complete prediction caches are reused when
their inputs and source hashes match. The launcher rejects a duplicate active job.

## 6. Merge the completed results

After both jobs succeed:

```bash
.venv/bin/python slurm/paper_jobs.py merge --datasets ECL,solar --output results/paper_cluster
```

The merge stops if either dataset has incomplete runs.
The merged results are in `results/paper_cluster/`:
`summary.csv`, `runs.csv`, `uncertainty.csv`, `phase_effects.csv`, and `inventory.json`.

## 7. Generate the tables and figures

```bash
.venv/bin/python manuscript/analysis/build_strengthening.py --results results/paper_cluster --prefix cluster
```

Collect the merged result files, `manuscript/generated/*cluster*`, and
`manuscript/figures/cluster_*` for review. The `cluster` prefix keeps these outputs
separate from the current manuscript's local results. Review them before changing
the manuscript's conclusions.
