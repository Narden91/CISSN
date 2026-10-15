# CISSN — what to run

Updated 2026-10-15, branch `conformal-per-cell-geometry`. Open runs only. `RUNBOOK.md` is the
protocol and wins on any conflict; `docs/prespecification.md` fixes the confirmatory analysis.
The finished blocks R1 to R7 and their commands are in `git show a1add62:experiments.md`.
Where other files name a block R1 to R7, they mean that version.

## 1. Open work

| Step | What | Where |
| --- | --- | --- |
| E1 | Rerun the 12 R5 baseline runs that the batch-of-one encoder bug hit | Cineca |
| E2 | Pack the 12 runs and pull them | Cineca, then workstation |
| E3 | Rebuild manuscript numbers, figures, and PDF | workstation |
| P0 | Copy the P3 dataset, pull the Step 7 commit on the cluster | workstation, then Cineca |
| P1 | Point baselines on ECL (block `p1`, 24 cells) | Cineca |
| P2 | ACI against flat CP: ECL (block `p2`) on the cluster, exposed grid on the workstation | both |
| P3 | traffic: memory check (`p3s`), confirmatory runs (`p3`, 6 cells), ACI (`p3b`) | Cineca |
| P4 | Pack, pull, and analyse traffic | both |

P0 to P4 are `RUNBOOK.md` Step 7. Start them after E1 finishes.

**P3 is blocked: its dataset is undecided.** Traffic at horizon 336 gives only 9.5 bootstrap
blocks (`RUNBOOK.md` Step 7 lists the options). The draft is already committed, so preflight
passes for P3: until the decision, do not launch `p3s`, `p3`, or `p3b`. P1 and P2 are not
affected. The traffic names in P0 and P4 change with the decision.

The 12 E1 cells are `mc_dropout` and `deep_ensemble` under RevIN, on ETTh1 and ETTh2, at
horizon 192, seeds 42, 123, 456. Before `6adc24b`, on Linux, `torch.compile` returned a wrong
encoder state for a batch of one window. These splits leave a final batch of one at batch size
128. The cluster already moved the old copies to `results/_superseded_bs1_bug/`. No other cell
is affected.

## 2. Shared commands

### 2.1 Launch and watch (cluster, bash, root of the clone that the Slurm jobs used)

Set `BLOCKS` to the blocks of the step. A second `launch.sh` call skips finished cells. The
failure check reads only these blocks, because older logs hold failures that were resolved.

```bash
BLOCKS=r5                      # E1: r5. Step 7: p1,p2 then p3s,p3,p3b
bash slurm/launch.sh --blocks $BLOCKS --dry-run
bash slurm/launch.sh --blocks $BLOCKS
squeue -u $(id -un)
.venv/bin/python slurm/cells.py status
for b in ${BLOCKS//,/ }; do cat logs/slurm/failed_$b.txt 2>/dev/null; done      # must print nothing
```

### 2.2 Pack only the files the analysis reads (cluster, bash)

A full run folder holds `pred`, `true`, `lower`, `upper`, `residuals`, and `states` arrays:
tens of MB to several GB per run. The analysis never reads them. Paste this helper once per
shell, set `BLOCKS` as in section 2.1 (it selects the logs to pack), and call it from the
repository root:

```bash
pack_small() {   # pack_small NAME EXPECTED_RUNS 'run-dir glob' ...
  local name="$1" expected="$2"; shift 2
  local pub="${PUB:-results/publication}" out="${WORK:-$HOME}/cissn_transfer"
  mkdir -p "$out"
  local dirs="$out/$name.dirs" files="$out/$name.files" tarball="$out/$name.tar.gz"
  : > "$dirs"
  for g in "$@"; do find "$pub" -maxdepth 1 -type d -name "$g" >> "$dirs"; done
  sort -u "$dirs" -o "$dirs"
  local n; n=$(wc -l < "$dirs")
  [ "$n" -eq "$expected" ] || { echo "ERROR: $n run directories match, expected $expected" >&2; return 1; }
  # Whitelist of what the analysis reads. Never pred, true, lower, upper, residuals, states.
  xargs -I{} find {} -maxdepth 1 -type f \( -name '*.json' -o -name 'per_origin_interval*_metrics.npy' -o -name test_origin_indices.npy \) < "$dirs" > "$files"
  find "$pub" -maxdepth 1 -type f -name 'aci_vs_flat_*.json' >> "$files"
  for b in smoke ${BLOCKS//,/ }; do
    find logs/slurm -maxdepth 1 -type f \( -name "cissn_${b}_*" -o -name "failed_${b}.txt" \) >> "$files"
  done
  tar -czf "$tarball" -T "$files"
  echo "$n run dirs, $(wc -l < "$files") files"; ls -lh "$tarball"; sha256sum "$tarball"; echo "OUT_DIR=$out"
  [ "$(stat -c %s "$tarball")" -lt 209715200 ] || echo "WARNING: tarball above 200 MB; find the cause with: tar -tvzf $tarball | sort -k3 -n | tail" >&2
}
```

The helper stops if the number of matching run folders differs from `EXPECTED_RUNS`, so a
partial run set never ships. The run files take about 20 to 80 KB per run. Logs add to that, and
may hold progress bars. The helper warns when the tarball exceeds 200 MB.

Two rules follow:

- An analysis that needs the large arrays runs on the cluster, and only its JSON output is
  pulled (the ECL rank diagnostic and the ACI pass work this way).
- `completion.json` lists the omitted arrays, so an integrity check against it fails on the
  workstation. Check the tarball SHA-256 instead.

### 2.3 Pull and unpack (workstation, PowerShell, repository root)

The archive extracts into `results/`. If your results folder carries the archive name
`results_14_10_2026/`, first copy its `publication/`, `selection/`, and `validation/` folders
back to `results/`; `build_numbers.py` and the checks below read them there.

Set `$name` to the pack name, `$remote` to the login host, and `$outDir` to the `OUT_DIR` value
that `pack_small` printed.

```powershell
$remote = 'USER@LOGIN_HOST'; $name = 'NAME'; $outDir = 'OUT_DIR'
scp "${remote}:$outDir/$name.tar.gz" .
Get-FileHash ".\$name.tar.gz" -Algorithm SHA256         # must equal the sha256 printed on the cluster
tar -xzkf ".\$name.tar.gz"                              # -k keeps existing files
```

## 3. E1. Rerun on the cluster

Run E1 on the cluster's current checkout. Do not `git pull` first: the Step 7 commit (`ef84e5e`)
would put these 12 runs on a different commit than the other 204 R5 runs. Check the
preconditions and stop if one fails.

```bash
git rev-parse HEAD
git status --porcelain                                   # must print nothing
git merge-base --is-ancestor 6adc24b HEAD && echo "encoder fix present"
squeue -u $(id -un)                                      # no cissn_* job may still run
.venv/bin/python slurm/cells.py check
.venv/bin/python slurm/cells.py list r5 | grep todo      # must print exactly the 12 cells
```

Launch with section 2.1, `BLOCKS` = `r5`. Status must reach `r5 216/216`. Then check each new
run:

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

All 12 lines must show `clean`, `fix`, and `structural ok`. A run that fails the structural
validity check is rerun once at the same seed. Never exclude a run for poor forecast quality.

## 4. E2. Pack and pull

On the cluster, with `pack_small` from section 2.2:

```bash
BLOCKS=r5
pack_small r5rerun 12 'BASELINE_mc_dropout_ETTh[12]_M_sl96_pl192_seed*_*fullrevin__*' 'BASELINE_deep_ensemble_ETTh[12]_M_sl96_pl192_seed*_*fullrevin__*'
```

On the workstation, run section 2.3 with `$name = 'r5rerun'`, then:

```powershell
(Get-ChildItem ./results/publication -Directory -Filter 'BASELINE_*fullrevin__*').Count   # must print 216
```

## 5. E3. Rebuild the manuscript (workstation, PowerShell, repository root)

`build_numbers.py` reads `results/publication`, `results/selection`, and `results/validation`
(see the note in section 2.3).

```powershell
uv run python manuscript/analysis/build_numbers.py
uv run python manuscript/analysis/make_figures.py
Push-Location manuscript; latexmk -pdf -interaction=nonstopmode main.tex; Pop-Location
Select-String -Path manuscript/generated/tab_point.tex -Pattern 'ddagger'   # must print nothing
```

`build_numbers.py` stops if a run fails the structural validity check, or if a per-origin
array does not reproduce its `metrics.json` Winkler score. Then update the text that still
calls the 12 runs missing:

1. In `manuscript/main.tex`, "Deviations and disclosures": replace "they are excluded and wait
   for a rerun" with a statement that they were rerun at the fix commit, settings unchanged.
2. In the same file, remove the `$^\ddagger$` sentence from the caption of `tab:point`, and
   remove "Twelve baseline runs await a rerun" from "Limitations".
3. State that R5 now spans two commits, `6711026` and the rerun commit, which differ only by
   the encoder fix. The provenance table shows both.
4. In `manuscript/README.md`, delete open item 3.

## 6. Step 7 (after E1)

### P0. Copy, pull

1. The Step 7 code and `docs/prespecification_traffic.md` are committed (`ef84e5e`). Settle the
   P3 dataset first and update the draft; it can change freely until a traffic test window
   is opened.
2. Copy the traffic file byte for byte (sealed runs check its SHA-256):

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

### P1, P2, P3. Launch on the cluster

Use section 2.1. P1 and P2 need neither the traffic data nor its prespecification:

- `BLOCKS` = `p1,p2`.
- Then, once the prespecification is committed and `traffic.csv` is on the cluster:
  `BLOCKS` = `p3s,p3,p3b`. `p3s` trains one epoch on traffic through the validation-only
  controller to check memory. `p3` starts only if it succeeds, and `p3b` only if every `p3`
  task succeeds.

After `p3s`, compare its epoch time times 20 with `TIME_P3` in `slurm/env.sh` (an estimate)
and raise the limit before `p3` starts if needed. At the end, status must read `p1 24/24`,
`p2 1/1`, `p3s 1/1`, `p3 6/6`, `p3b 1/1`. The ACI script stops on its own if a reload does not
reproduce the saved flat CP; read the largest reload drift with:

```bash
.venv/bin/python -c "import glob, json; print('max reload drift', max(r['reproduction_max_relative_drift'] for f in glob.glob('results/publication/aci_vs_flat_*.json') for c in json.load(open(f))['cells'] for r in c['runs']))"
```

### P2 on the exposed grid (workstation, PowerShell, repository root)

The exposed-grid checkpoints are on the workstation, so this pass runs here:

```powershell
uv run python scripts/aci_paired_comparison.py --results_root ./results/publication --checkpoints_root ./checkpoints/publication --datasets ETTh2 weather exchange_rate --gamma 0.005 --block_multiplier 1 --n_boot 5000 --seed 0 --output ./results/publication/aci_vs_flat_exposed.json
```

### P4. Pack, pull, analyse

On the cluster, with `pack_small` from section 2.2 (24 ECL baselines and 6 traffic runs):

```bash
BLOCKS=p1,p2,p3s,p3,p3b
pack_small step7 30 'BASELINE_*_ECL_*fullrevin__*' 'CISSN_traffic_*'
```

On the workstation, run section 2.3 with `$name = 'step7'`, then:

```powershell
foreach ($method in 'primary','state_scaled') { foreach ($conf in '0.975','0.95') {
  uv run python scripts/paired_interval_inference.py --results_root ./results/publication --datasets traffic --block_multiplier 1 --n_boot 5000 --seed 0 --method $method --confidence $conf --output "./results/publication/paired_interval_inference_traffic_${method}_$conf.json"
} }
```

Read E1 and E2 from the horizon-336 rows with the Holm steps and the reading rule of
`docs/prespecification_traffic.md`, section 3. Then rerun E3 to place the P1 runs in the point
table, and update the manuscript sentence "No baseline ran on ECL".

## 7. Rules that bind every launch

- Confirmation runs need `--evidence_role confirmation --immutable_artifacts
  --strict_determinism --require_clean_git` together. `slurm/cells.py` sets them.
- Do not change code in `cissn/` or `experiments/` between the launch commit and the end of E1.
- The H100 and the RTX 5080 differ beyond floating-point noise under the same seed. Run all 12
  E1 cells on the H100 nodes, like the other 204 R5 runs.
- Never copy a whole run folder or `results/publication` from the cluster. Use `pack_small`.
- A new experiment needs a `RUNBOOK.md` step before it gets a command here.
