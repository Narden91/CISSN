# Follow-up reproducibility reference

For the steps to launch only the remaining prepared HPC jobs, follow
[HPC_JOBS.md](../HPC_JOBS.md). That file covers ECL and Solar, checking files,
submission, monitoring, result merging, and table/figure generation.

This reference records the completed local work, output files, and research details.

## Local commands actually used

These commands use the existing project environment. No package was added.

```powershell
.\.venv\Scripts\python.exe -m experiments.run_paper_suite --config experiments/paper_suite.yaml --inventory-only
.\.venv\Scripts\python.exe -m experiments.run_paper_suite --config experiments/paper_suite.yaml
.\.venv\Scripts\python.exe -m experiments.synthetic_phase
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py
.\.venv\Scripts\python.exe scripts/paper_strengthening_report.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_paper_jobs.py -q -p no:cacheprovider --basetemp results/paper_strengthening/pytest-hpc-checks-20261016
```

The first full test command ran before the eight checker tests were added.
The second command ran those eight tests. For a future full run in the restricted
Windows environment, choose a new, unused workspace path for `--basetemp`.
Pytest clears that path, so do not point it at a folder containing existing files.

The shell checker was exercised through Git Bash on Windows. An ETTh2/CISSN-only
temporary configuration passed the complete resource check and launcher dry run.
The default ECL/Solar check failed on the known missing local weights, as intended.
No Slurm job was submitted here. Real scheduler acceptance remains a cluster check.

## Data flow and outputs

1. `prepare_cache` reloads frozen weights and preserves the existing loader splits.
2. It writes all calibration residuals and states to disk in chronological order.
3. It writes the selected test predictions, targets, states, phases, and origins.
4. `fit_methods` fits the partition on the first calibration half only.
5. It fits paired methods on the original second-half selection. Full Flat uses all calibration origins.
6. `score_cache` saves coverage, width, and Winkler by origin and by horizon.
7. `summarize` averages seeds at each origin, then resamples chronological paired differences.
8. `build_strengthening.py` writes numerical tables, figures, and manuscript macros.

`cache/<run>/metadata.json` records the dataset checksum, exact checkpoint hashes,
source hashes, original configuration, encoder configuration, stride, and environment.
`analysis_manifest.json` records analysis source hashes and arguments for new runs.
`protocol.yaml` records the complete analysis settings. The first local execution
predated the added analysis-manifest sidecar; its prediction-cache metadata and
frozen-reproduction reports record its provenance.

`scores/<run>/calibration_alpha*.json` contains the selected indices, every group
count, and every fallback. `diagnostics.json` contains cluster occupancy, phase
contingency, residual quantiles, horizon/channel means, and normalised mutual
information (NMI). Residual-regime edges are fitted on calibration scores only.
No test diagnostic is eligible for method selection.

`summary.csv`, `runs.csv`, `uncertainty.csv`, and `phase_effects.csv` are the numerical
outputs. Confidence intervals average seeds before resampling origins. Their block
lengths are one, two, or four horizons in origin units. Fewer than ten blocks is
flagged as too coarse for a strong reading. Phase-specific confidence intervals
and development-selected matched-coverage scaling are not implemented.

## Rebuild the paper

Preserve the original generated results and run the new artifact script before LaTeX:

```powershell
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py
Push-Location manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

The original table/figure scripts remain `manuscript/analysis/build_numbers.py`
and `manuscript/analysis/make_figures.py`. The new script does not overwrite their
tables or figures. All follow-up numbers in the text come from generated macros.

## Future confirmation

`experiments/prespecified_confirmatory.yaml` records Traffic and ILI as candidates.
They require a complete exposure-history check and identified frozen weights before
test scoring. ETTm1/ETTm2 share sources with exposed ETT datasets and are excluded.
No candidate test was scored in this task. No file was committed, under the user's
Git instructions. This is an internal prespecification, not external registration.

## Verified related-work additions

Metadata was checked on 2026-10-16 against these primary sources:

| Entry | Source |
| --- | --- |
| Guan, localized conformal prediction, 2023 | https://academic.oup.com/biomet/article/110/1/33/6647831 |
| Auer et al., HopCPT, 2023 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html |
| Xu and Xie, SPCI, 2023 | https://proceedings.mlr.press/v202/xu23r.html |
| Sun and Yu, CopulaCPTS, 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html |
| Xu and Xie, EnbPI, 2021; added volume/pages | https://proceedings.mlr.press/v139/xu21h.html |

The other pre-existing bibliography entries were not all independently rechecked.
