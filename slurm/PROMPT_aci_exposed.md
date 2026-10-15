# Prompt: check ACI on the exposed grid (Claude Code on the Cineca login node)

Paste the block below into Claude Code, started in the CISSN clone that the Slurm jobs used, after you
pulled the Step 7 commit. It checks whether adaptive conformal inference (ACI) against flat CP can
run on the cluster for ETTh2, weather, and exchange_rate (39 CISSN runs), and runs the full pass only
if the check passes.

One risk is built in. The exposed-grid checkpoints were trained on the workstation (Windows, RTX
5080). A reload on Linux may not reproduce the saved flat CP. The ACI script stops itself if the
relative drift exceeds `1e-4`; on the workstation the drift is about `1.5e-7`. If the trial in
step 3 fails, run this pass on the workstation instead (`experiments.md`, section 5).

````text
You run on the Cineca login node, inside the CISSN clone that the Slurm jobs used. Read CLAUDE.md,
experiments.md (sections 3 to 5), and scripts/aci_paired_comparison.py before you act.

GOAL
Check whether adaptive conformal inference (ACI) against flat CP can run here for the exposed
grid: ETTh2, weather, exchange_rate (39 CISSN runs). Run the full pass only if the check passes.

RULES
- Never delete, move, or overwrite anything under results/ or checkpoints/. Never edit, stage, or
  commit a tracked file. Never pull, switch branch, or submit a Slurm job. The tree must stay clean.
- Change no setting of the ACI script (gamma 0.005, drift tolerance 1e-4) and patch no script. If a
  command fails, quote the exact error, stop that step, and continue with the next.
- Report numbers only. Do not apply any reading rule.
- Use PY=.venv/bin/python, not `uv run`. On the login node run light CPU work only. Stop and report
  any step that needs more than about 15 minutes.
- Write outputs only to results/publication/aci_vs_flat_exposed.json (step 4) and to
  OUT=$WORK/cissn_aci_exposed ($HOME/cissn_aci_exposed if $WORK is unset).

STEP 0. Preconditions. Report each result. Stop if one fails.
- `git rev-parse HEAD`; `git status --porcelain` (must be empty);
  `git cat-file -e HEAD:scripts/aci_paired_comparison.py`.
- `squeue -u $(id -un)` shows no cissn_* job.

STEP 1. Inventory.
List the CISSN runs under results/publication whose config.json has data in {ETTh2, weather,
exchange_rate}, architecture legacy, revin true, evidence_role confirmation, and whose metrics.json
has structural_passed true. Expect 39: ETTh2 5 horizons, weather 4, exchange_rate 4, each with seeds
42, 123, 456. Report the count per dataset and horizon, and per group the git_commit and
cuda_device_name from environment.json.

STEP 2. Files.
For each of the 39 runs check that pred.npy, true.npy, test_origin_indices.npy, and
per_origin_interval_flat_cp_metrics.npy exist in the run folder, and that
checkpoints/publication/<same folder name> holds checkpoint.pth, checkpoint_head.pth, and
checkpoint_revin.pth. Check the three data files with
cissn.data.registry.verify_dataset(name, strict=True, require_exact_checksum=True) for ETTh2,
weather, and exchange_rate. List every missing item. If any item is missing, stop: steps 3 and 4
need all of them.

STEP 3. Reload trial: one run per dataset, h96, seed 42.
Run this from the repository root and report its three output lines:

.venv/bin/python - <<'EOF'
import json, sys, time
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from aci_paired_comparison import score_run
for data in ("ETTh2", "weather", "exchange_rate"):
    run = next(Path("results/publication").glob(f"CISSN_{data}_M_sl96_pl96_*fullrevin*seed42__*"))
    config = json.loads((run / "config.json").read_text())
    t = time.time()
    try:
        out = score_run(run, Path("checkpoints/publication"), config, 0.005)
        print(data, "drift", out["reproduction_max_relative_drift"], "n_calibration", out["n_calibration"], "seconds", round(time.time() - t, 1))
    except ValueError as e:
        print(data, "STOP:", e)
EOF

Decision: all three drifts below 1e-4 -> continue. Any "STOP" line or any drift above 1e-4 -> stop
and quote it. Say that the checkpoints were trained on another machine (workstation, RTX 5080,
Windows) and that a failed reload means the exposed-grid pass stays on the workstation. Also report
the seconds per run and estimate the full time as the mean times 39.

STEP 4. Full pass, only if step 3 passed and the estimate is below 15 minutes.

.venv/bin/python scripts/aci_paired_comparison.py --results_root ./results/publication --checkpoints_root ./checkpoints/publication --datasets ETTh2 weather exchange_rate --gamma 0.005 --block_multiplier 1 --n_boot 5000 --seed 0 --output ./results/publication/aci_vs_flat_exposed.json

If the estimate is higher, do not run it. Write the command and the estimate, and stop.

STEP 5. Report.
Print a table with one row per dataset and horizon: relative_mean, relative_ci_low, relative_ci_high,
aci_coverage, flat_coverage, aci_width, flat_width, the largest n_capped over the seeds, and the
largest reload drift. End with the exact scp command (real host and path) for
results/publication/aci_vs_flat_exposed.json.
````

After it finishes, `pack_small` (`experiments.md`, section 4) already includes `aci_vs_flat_*.json`, so
the result comes back with the next pack.
