# CISSN — what to run

Updated 2026-10-15, branch `conformal-per-cell-geometry`. This file lists only the runs that
are still open, with their commands. `RUNBOOK.md` is the protocol and wins on any conflict;
`docs/prespecification.md` fixes the confirmatory analysis. The finished blocks R1 to R7 and
their commands are in git history: `git show a1add62:experiments.md`. Where `RUNBOOK.md`,
`docs/prespecification.md`, or `slurm/` name a block R1 to R7, they mean that version.

## 1. Open work

| Step | What | Where | Cells |
| --- | --- | --- | --- |
| E1 | Rerun the R5 baseline runs that the batch-of-one encoder bug affected | Cineca, bash | 12 |
| E2 | Pack the 12 runs and copy them to the workstation | Cineca, then workstation | - |
| E3 | Rebuild the manuscript numbers, figures, and PDF | workstation, PowerShell | - |
| P0 | Commit Step 7, copy traffic data, pull on the cluster | workstation, then Cineca | - |
| P1 | Point baselines on ECL (block `p1`) | Cineca | 24 |
| P2 | ACI against flat CP: ECL (block `p2`), exposed grid (workstation) | both | 1 + 1 |
| P3 | traffic memory check (`p3s`), confirmatory runs (`p3`), ACI pass (`p3b`) | Cineca | 1 + 6 + 1 |
| P4 | Pack, copy, and run the traffic analysis | both | - |

P0 to P4 are `RUNBOOK.md` Step 7. Start them only after E1 finishes.

The 12 cells are `mc_dropout` and `deep_ensemble` under RevIN, on ETTh1 and ETTh2, at
horizon 192, seeds 42, 123, and 456. Before `6adc24b`, on Linux, `torch.compile` returned a
wrong encoder state for a batch of one window. These splits leave a final batch of one at
batch size 128 (validation 2689, calibration 1537, test 2689 windows). The cluster already
moved the old copies to `results/_superseded_bs1_bug/` (inventory of 2026-10-15,
`manuscript/hpc/hpc_inventory.csv`). No other cell is affected.

## 2. E1. Rerun on the cluster (bash, root of the clone that the Slurm jobs used)

Check the preconditions. Stop if one fails.

```bash
git rev-parse HEAD
git status --porcelain                                   # must print nothing
git merge-base --is-ancestor 6adc24b HEAD && echo "encoder fix present"
squeue -u $(id -un)                                      # no cissn_* job may still run
.venv/bin/python slurm/cells.py check
.venv/bin/python slurm/cells.py list r5 | grep todo      # must print exactly the 12 cells
```

Launch. `launch.sh` first submits the one-epoch smoke job; the r5 cells wait for it.

```bash
bash slurm/launch.sh --blocks r5 --dry-run
bash slurm/launch.sh --blocks r5
```

Watch and resume. A second `launch.sh --blocks r5` skips finished cells.

```bash
squeue -u $(id -un)
.venv/bin/python slurm/cells.py status                   # r5 must reach 216/216
cat logs/slurm/failed_r5.txt 2>/dev/null                 # must not exist or be empty
```

Check each new run before you pack it:

```bash
for d in results/publication/BASELINE_{mc_dropout,deep_ensemble}_ETTh{1,2}_M_sl96_pl192_seed*_*fullrevin__*; do
  .venv/bin/python - "$d" <<'EOF'
import json, subprocess, sys
from pathlib import Path
d = Path(sys.argv[1])
env = json.loads((d / "environment.json").read_text())
has_fix = subprocess.run(["git", "merge-base", "--is-ancestor", "6adc24b", env["git_commit"]]).returncode == 0
structural = json.loads((d / "sanity.json").read_text())["structural_passed"]
print(d.name, env["git_commit"][:7], "dirty" if env["git_dirty"] else "clean",
      "fix" if has_fix else "NO FIX", "structural ok" if structural else "STRUCTURAL FAIL")
EOF
done
```

Each of the 12 lines must show `clean`, `fix`, and `structural ok`. A run that fails the
structural validity check is rerun once at the same seed. Never exclude a run for poor
forecast quality.

## 3. E2. Pack and copy

On the cluster:

```bash
OUT="${WORK:-$HOME}/cissn_transfer_r5rerun"; mkdir -p "$OUT"
find results/publication -maxdepth 1 -type d \( -name 'BASELINE_mc_dropout_ETTh[12]_M_sl96_pl192_seed*_*fullrevin__*' -o -name 'BASELINE_deep_ensemble_ETTh[12]_M_sl96_pl192_seed*_*fullrevin__*' \) > "$OUT/dirs.txt"
wc -l < "$OUT/dirs.txt"                                  # must print 12
xargs -a "$OUT/dirs.txt" -I{} find {} -type f -size -5121k > "$OUT/files.txt"
find logs/slurm -type f \( -name 'cissn_r5_*' -o -name 'cissn_smoke_*' -o -name 'failed_r5*' \) >> "$OUT/files.txt"
tar -czf "$OUT/cissn_small_r5rerun.tar.gz" -T "$OUT/files.txt"
sha256sum "$OUT/cissn_small_r5rerun.tar.gz"
```

On the workstation (PowerShell, repository root). Set the two variables to the real login
host and to the cluster value of `$OUT`.

```powershell
$remote = 'USER@LOGIN_HOST'; $outDir = 'OUT_DIR'
scp "${remote}:$outDir/cissn_small_r5rerun.tar.gz" .
Get-FileHash .\cissn_small_r5rerun.tar.gz -Algorithm SHA256   # must equal the sha256sum value
tar -xzkf .\cissn_small_r5rerun.tar.gz                         # -k keeps existing files
(Get-ChildItem ./results/publication -Directory -Filter 'BASELINE_*fullrevin__*').Count  # must print 216
```

## 4. E3. Rebuild the manuscript (workstation, PowerShell, repository root)

```powershell
uv run python manuscript/analysis/build_numbers.py
uv run python manuscript/analysis/make_figures.py
Push-Location manuscript; latexmk -pdf -interaction=nonstopmode main.tex; Pop-Location
Select-String -Path manuscript/generated/tab_point.tex -Pattern 'ddagger'   # must print nothing
```

`build_numbers.py` stops if a run fails the structural validity check, or if a per-origin
array does not reproduce its `metrics.json` Winkler score.

Then update the text that still describes the 12 runs as missing:

1. In `manuscript/main.tex`, section "Deviations and disclosures": replace "they are
   excluded and wait for a rerun" with a statement that they were rerun at the fix commit
   with unchanged settings.
2. In the same file, remove the `$^\ddagger$` sentence from the caption of `tab:point`, and
   remove "Twelve baseline runs await a rerun" from "Limitations".
3. State that R5 now spans two commits, `6711026` and the rerun commit, which differ only by
   the encoder fix. The provenance table shows both after the rebuild.
4. In `manuscript/README.md`, delete open item 3.

## 5. Step 7 (after E1)

### P0. Commit, copy, pull

1. Review `docs/prespecification_traffic.md`, then commit it with the Step 7 code before
   any traffic run. Without it in `HEAD`, `slurm/preflight.sh` refuses to launch.
2. Copy the traffic file to the cluster, byte for byte (sealed runs check its SHA-256):

   ```powershell
   $remote = 'USER@LOGIN_HOST'; $clone = 'SCRATCH/CISSN'
   scp data/traffic.csv "${remote}:$clone/data/"
   ```

3. On the cluster, when `squeue` shows no `cissn_*` job:

   ```bash
   git pull --ff-only
   git merge-base --is-ancestor 6adc24b HEAD && echo "encoder fix present"
   bash slurm/preflight.sh                  # every line must be OK
   ```

### P1, P2, P3. Launch on the cluster (bash)

P1 and P2 do not need the traffic data or its prespecification:

```bash
.venv/bin/python slurm/cells.py check
bash slurm/launch.sh --blocks p1,p2 --dry-run
bash slurm/launch.sh --blocks p1,p2
```

P3 only after `docs/prespecification_traffic.md` is committed and `data/traffic.csv` is on
the cluster. `p3s` trains one epoch on traffic through the validation-only controller to
check memory; `p3` starts only if it succeeds, and `p3b` only if every `p3` task succeeds.

```bash
bash slurm/launch.sh --blocks p3s,p3,p3b --dry-run
bash slurm/launch.sh --blocks p3s,p3,p3b
```

Watch and resume as in E1. After `p3s`, compare its epoch time times 20 with `TIME_P3` in
`slurm/env.sh` (an estimate) and raise the limit if needed before `p3` starts. Then check:

```bash
.venv/bin/python slurm/cells.py status                  # p1 24/24, p2 1/1, p3s 1/1, p3 6/6, p3b 1/1
cat logs/slurm/failed_p*.txt 2>/dev/null                # must not exist or be empty
.venv/bin/python -c "import glob, json; print('max reload drift', max(r['reproduction_max_relative_drift'] for f in glob.glob('results/publication/aci_vs_flat_*.json') for c in json.load(open(f))['cells'] for r in c['runs']))"
```

The ACI script stops on its own if a reload does not reproduce the saved flat CP.

### P2 on the exposed grid (workstation, PowerShell, repository root)

The exposed-grid checkpoints are on the workstation, so this pass runs here.

```powershell
uv run python scripts/aci_paired_comparison.py --results_root ./results/publication --checkpoints_root ./checkpoints/publication --datasets ETTh2 weather exchange_rate --gamma 0.005 --block_multiplier 1 --n_boot 5000 --seed 0 --output ./results/publication/aci_vs_flat_exposed.json
```

### P4. Pack, copy, analyse

On the cluster:

```bash
OUT="${WORK:-$HOME}/cissn_transfer_step7"; mkdir -p "$OUT"
find results/publication -maxdepth 1 -type d \( -name 'BASELINE_*_ECL_*fullrevin__*' -o -name 'CISSN_traffic_*' \) > "$OUT/dirs.txt"
wc -l < "$OUT/dirs.txt"                                  # must print 30 (24 baselines, 6 traffic)
xargs -a "$OUT/dirs.txt" -I{} find {} -type f -size -5121k > "$OUT/files.txt"
ls results/publication/aci_vs_flat_*.json >> "$OUT/files.txt"
find logs/slurm -type f \( -name 'cissn_p*' -o -name 'cissn_smoke_*' -o -name 'failed_p*' \) >> "$OUT/files.txt"
tar -czf "$OUT/cissn_small_step7.tar.gz" -T "$OUT/files.txt"
sha256sum "$OUT/cissn_small_step7.tar.gz"
```

On the workstation:

```powershell
$remote = 'USER@LOGIN_HOST'; $outDir = 'OUT_DIR'
scp "${remote}:$outDir/cissn_small_step7.tar.gz" .
Get-FileHash .\cissn_small_step7.tar.gz -Algorithm SHA256       # must equal the sha256sum value
tar -xzkf .\cissn_small_step7.tar.gz
foreach ($method in 'primary','state_scaled') { foreach ($conf in '0.975','0.95') {
  uv run python scripts/paired_interval_inference.py --results_root ./results/publication --datasets traffic --block_multiplier 1 --n_boot 5000 --seed 0 --method $method --confidence $conf --output "./results/publication/paired_interval_inference_traffic_${method}_$conf.json"
} }
```

Read E1 and E2 from the horizon-336 rows with the Holm steps and the reading rule of
`docs/prespecification_traffic.md`, section 3. Then rerun E3 to place the P1 runs in the
point table, and update the manuscript sentence "No baseline ran on ECL".

## 6. Rules that bind every launch

- Confirmation runs need `--evidence_role confirmation --immutable_artifacts
  --strict_determinism --require_clean_git` together. `slurm/cells.py` sets them for r5.
- Do not change code in `cissn/` or `experiments/` between the launch commit and the end of
  E1.
- Results from the H100 and the RTX 5080 differ beyond floating-point noise under the same
  seed. Run all 12 cells on the H100 nodes, like the other 204 R5 runs.
- A new experiment needs a `RUNBOOK.md` step before it gets a command here. P1 to P3 are
  `RUNBOOK.md` Step 7.
