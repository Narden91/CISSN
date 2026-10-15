# CISSN — what to run

Updated 2026-10-15, branch `conformal-per-cell-geometry`. Open runs only. `RUNBOOK.md` is protocol, wins on any conflict; `docs/prespecification.md` fixes confirmatory analysis.
Finished blocks R1 to R7 and commands in `git show a1add62:experiments.md`.

## 1. Workflow

1. **Workstation.** Edit code, run `uv run python tests/run_tests.py`, commit, push.
   Nothing else leaves machine.
2. **Cluster.** `git pull --ff-only`, then one launcher call submits every open experiment
   (section 3).
3. **Back on workstation.** Pack small result files on cluster, pull, rebuild
   manuscript (sections 4 and 5).

## 2. Open experiments

`bash slurm/launch.sh` submits exactly these blocks. Finished blocks never rerun.

| Block | What | Cells |
| --- | --- | --- |
| `r5` | Rerun 12 baseline runs hit by batch-of-one encoder bug | 12 |
| `p1` | Point baselines on ECL (DLinear, PatchTST-lite, MC-Dropout, Deep Ensemble; h96 and h336) | 24 |
| `p2` | ACI vs flat CP on ECL CISSN runs | 1 |
| `p3s`, `p3`, `p3b` | Second confirmatory test on solar: one-epoch memory check, 6 CISSN runs, ACI pass | 1 + 6 + 1 |

12 `r5` cells: `mc_dropout` and `deep_ensemble` under RevIN, on ETTh1 and ETTh2, horizon 192, seeds 42, 123, 456. Before `6adc24b`, on Linux, `torch.compile` returned wrong encoder state for batch of one window. These splits leave final batch of one at batch size 128. Cluster already moved old copies to `results/_superseded_bs1_bug/`.

Runs use commit you pull, later than commit of other 204 R5 runs
(`6711026`). Since then code changed only by encoder fix and Step 7 additions: `--test_origin_stride` option (default 1, same loader) and extra output files. Manuscript states both commits (section 5).

**P3 (solar).** Needs `docs/prespecification_solar.md` in `HEAD` and `data/solar_AL.txt` on the
cluster; preflight stops otherwise. Test origins use stride 25, not 24 (`RUNBOOK.md` Step 7 gives
the reason). `p3` starts only if `p3s` succeeds, `p3b` only if every `p3` task succeeds.

## 3. Launch on the cluster (bash, root of the clone that the Slurm jobs used)

Once, from the workstation (PowerShell), copy the solar file byte for byte; sealed runs check its
SHA-256:

```powershell
$remote = 'USER@LOGIN_HOST'; $clone = 'SCRATCH/CISSN'
scp data/solar_AL.txt "${remote}:$clone/data/"
```

Then on the cluster:

```bash
git pull --ff-only
git status --porcelain                                   # must print nothing
git merge-base --is-ancestor 6adc24b HEAD && echo "encoder fix present"
bash slurm/launch.sh --dry-run                           # runs preflight, prints the sbatch commands
bash slurm/launch.sh                                     # smoke job first; r5, p1, p2, p3s wait for it
```

Preflight fails → fix each `FAIL` line; nothing submitted. Smoke job fails → blocks stay pending: read `logs/slurm/cissn_smoke_*.out`, fix cause, cancel them with the
command `launch.sh` prints.

Watch and resume. Second `launch.sh` call skips finished cells. Do not `git pull` or edit code
while jobs run: every run records its commit.

```bash
squeue -u $(id -un)
.venv/bin/python slurm/cells.py status                   # r5 216/216, p1 24/24, p2 1/1, p3s 1/1, p3 6/6, p3b 1/1 when done
for b in r5 p1 p2 p3s p3 p3b; do cat logs/slurm/failed_$b.txt 2>/dev/null; done      # must print nothing
```

When jobs end, check 12 `r5` runs:

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
.venv/bin/python -c "import glob, json; print('max ACI reload drift', max(r['reproduction_max_relative_drift'] for f in glob.glob('results/publication/aci_vs_flat_*.json') for c in json.load(open(f))['cells'] for r in c['runs']))"
```

All 12 lines must show `clean`, `fix`, `structural ok`. Reload drift must stay below
`1e-4` (ACI script stops itself above that). Run failing structural validity
check → rerun once at same seed by second `launch.sh` call. Never exclude run for poor
forecast quality.

## 4. Pack and pull

Full run folder holds `pred`, `true`, `lower`, `upper`, `residuals`, `states` arrays: tens
of MB to several GB per run. Analysis never reads them. On cluster, paste this helper
once per shell, call from repository root.

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

Pack 12 reruns, 24 ECL baselines, 6 solar runs, and ACI results in one tarball:

```bash
BLOCKS=r5,p1,p2,p3s,p3,p3b
pack_small step7 42 'BASELINE_mc_dropout_ETTh[12]_M_sl96_pl192_seed*_*fullrevin__*' 'BASELINE_deep_ensemble_ETTh[12]_M_sl96_pl192_seed*_*fullrevin__*' 'BASELINE_*_ECL_*fullrevin__*' 'CISSN_solar_*'
```

Helper stops if run count differs from expected, so partial set never ships.
Run files take about 20 to 80 KB per run; logs add more. Helper warns above 200 MB.
Two rules follow. Analysis needing large arrays runs on cluster; only its JSON
output gets pulled. `completion.json` lists omitted arrays, so integrity check against it
fails on workstation; check tarball SHA-256 instead.

On workstation (PowerShell, repository root), set `$remote` to login host and `$outDir`
to `OUT_DIR` value helper printed:

```powershell
$remote = 'USER@LOGIN_HOST'; $name = 'step7'; $outDir = 'OUT_DIR'
scp "${remote}:$outDir/$name.tar.gz" .
Get-FileHash ".\$name.tar.gz" -Algorithm SHA256         # must equal the sha256 printed on the cluster
tar -xzkf ".\$name.tar.gz"                              # -k keeps existing files
$b = Get-ChildItem ./results/publication -Directory -Filter 'BASELINE_*fullrevin__*'
($b | Where-Object Name -notlike '*_ECL_*').Count       # must print 216
($b | Where-Object Name -like '*_ECL_*').Count         # must print 24
(Get-ChildItem ./results/publication -Directory -Filter 'CISSN_solar_*').Count   # must print 6
```

## 5. Rebuild the manuscript (workstation, PowerShell, repository root)

Exposed-grid checkpoints on workstation, so ACI pass for that grid runs here. To check first
whether it can run on the cluster instead, paste `slurm/PROMPT_aci_exposed.md` into Claude Code there:

```powershell
uv run python scripts/aci_paired_comparison.py --results_root ./results/publication --checkpoints_root ./checkpoints/publication --datasets ETTh2 weather exchange_rate --gamma 0.005 --block_multiplier 1 --n_boot 5000 --seed 0 --output ./results/publication/aci_vs_flat_exposed.json
uv run python manuscript/analysis/build_numbers.py
uv run python manuscript/analysis/make_figures.py
Push-Location manuscript; latexmk -pdf -interaction=nonstopmode main.tex; Pop-Location
Select-String -Path manuscript/generated/tab_point.tex -Pattern 'ddagger'   # must print nothing
```

`build_numbers.py` stops if run fails structural validity check, or if per-origin array
does not reproduce its `metrics.json` Winkler score. Then update manuscript:

1. In `manuscript/main.tex`, "Deviations and disclosures": replace "they are excluded and wait
   for a rerun" with statement that they were rerun at fix commit, settings unchanged.
2. Same file: remove `$^\ddagger$` sentence from caption of `tab:point`, remove
   "Twelve baseline runs await a rerun" from "Limitations", replace "No baseline ran on
   ECL" with P1 results.
3. State R5 now spans two commits, `6711026` and rerun commit. They differ by
   encoder fix and Step 7 additions that leave stride-1 computation unchanged. Provenance table shows both.
4. In `manuscript/README.md`, delete open item 3.
5. `build_numbers.py` has no rows yet for ACI results (`aci_vs_flat_*.json`). Add them from
   those files only, then cite them as calibrator with more information than split CP.

## 6. Solar analysis (workstation, PowerShell, repository root, after the pull)

`p3b` already ran ACI on the solar runs on the cluster; its `aci_vs_flat_solar.json` arrived with the
pack. Run the paired analysis for both endpoints at both confidence levels (the Holm steps of
`docs/prespecification_solar.md` need the 97.5% interval):

```powershell
foreach ($method in 'primary','state_scaled') { foreach ($conf in '0.975','0.95') {
  uv run python scripts/paired_interval_inference.py --results_root ./results/publication --datasets solar --block_multiplier 1 --n_boot 5000 --seed 0 --method $method --confidence $conf --output "./results/publication/paired_interval_inference_solar_${method}_$conf.json"
} }
```

Read the two endpoints from the horizon-336 rows with the Holm steps and the reading rule of
`docs/prespecification_solar.md`, section 3. If `p3s` ran longer than expected, compare its epoch
time times 20 with `TIME_P3` in `slurm/env.sh` before you rerun `p3`.

## 7. Rules that bind every launch

- Confirmation runs need `--evidence_role confirmation --immutable_artifacts
  --strict_determinism --require_clean_git` together. `slurm/cells.py` sets them.
- Never change code, pull, or switch branch on cluster while jobs run.
- H100 and RTX 5080 differ beyond floating-point noise under same seed. Run on
  H100 nodes, like other 204 R5 runs.
- Never copy whole run folder or `results/publication` from cluster. Use `pack_small`.
- New experiment needs `RUNBOOK.md` step before it gets command here.