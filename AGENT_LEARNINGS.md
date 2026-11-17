# Agent learnings

Each entry: the concrete failure, then the rule that prevents it.

## Pytest temporary directories in the Windows sandbox

- **Failure:** tests using `tmp_path` failed during setup because pytest could not
  read `AppData/Local/Temp/pytest-of-emanu`, although ordinary tests passed.
  **Rule:** pass a new, unused `--basetemp` path inside the repository for these
  checks. Never reuse a directory containing user files: pytest clears that path.

## Editing files with LaTeX or regex strings

- **Failure:** three Python heredoc patches (`s.replace(old, new)`) to a script that writes
  LaTeX failed or corrupted it: the shell and Python each consumed a backslash level, so
  `"$^\\ast$"` became `"$^\ast$"` and `\a` turned into a bell character.
  **Rule:** change strings that hold backslashes with the Edit tool, which takes the text
  literally; do not route them through a heredoc into Python.

## PowerShell run blocks

- **Failure:** `$flag = if ($x) { @('--revin') } else { @() }` followed by `@flag` passed
  `- - r e v i n` to the program. An `if` expression unrolls a one-element array into a
  string, and splatting a string passes each character as an argument.
  **Rule:** assign arrays directly (`$flag = @(); if ($x) { $flag = @('--revin') }`) before
  splatting.
- **Failure:** `--ablations a,b,c` split into separate arguments. An unquoted comma list is a
  PowerShell array.
  **Rule:** quote comma lists, or use a value that needs no comma.
- **Failure:** run commands were written into `experiments.md` after a parse check only, and the
  first launch failed.
  **Rule:** before writing a launch command into a doc, run it through the target script's
  real argument parser with the real PowerShell block (override `uv` to call a parse-only
  wrapper). Check every skip-check wildcard against the runner's real name builder.
- **Failure:** `HPC_JOBS.md` gave `--basetemp results/_pytest_<date>_a`. The user pasted it as
  is; Windows rejects `<` and `>` in folder names, so 9 `tmp_path` tests errored at setup.
  **Rule:** a copy-and-paste command has no placeholder the user must edit. Compute the value
  in the command (`$(Get-Date -Format yyyyMMdd_HHmmss)`) and run the exact text once.

## Dataset-specific facts in review notes

- **Failure:** a review plan stated that the phase control has 24 groups on every exposed
  dataset. The phase definition differs by dataset: hour of day for hourly data, the native
  within-day slot for subhourly data, and weekday for Exchange-rate. The group count, the
  fallback count, and the stride therefore differ per dataset.
  **Rule:** before writing a group count, fallback count, or stride for a dataset, read the
  phase definition in the code (`first_target_phases`, `phase_from_timestamps`) and the saved
  calibration counts for that dataset. Do not carry one dataset's value to the others.
- **Failure:** a plan said to skip the coarse phase control when `period <= G` and named
  Exchange-rate as the example. Exchange has period 7 and `G` is at most 5, so the rule never
  skips it. The rule was written without computing it for each dataset.
  **Rule:** when a plan states a skip or fallback rule, evaluate it for every dataset
  (ECL 24, ETTh2 24, Weather 144, Solar 144, Exchange 7) before you write it down.

## Scoring code that silently accepts new names

- **Failure:** in `experiments/run_paper_suite.py::score_cache`, the last `else` sends every
  method name that is not flat, state, or phase to `adaptive_conformal_quantiles`. A new
  method added to `METHODS` without its own branch is scored as ACI and raises no error.
  **Rule:** when you add a method name, end each dispatch with an explicit `elif` and a
  `raise ValueError`, and add a test that every name in `METHODS` returns widths of the right
  shape. Grep `manuscript/analysis/build_strengthening.py` for hard-coded method lists too.
- **Failure:** `run_paper_suite.py --inventory-only` still writes `protocol.yaml`,
  `analysis_manifest.json`, and `inventory.json` into `output_root`. Run against the real
  config, it overwrites the provenance of a finished run.
  **Rule:** test every run command with a copy of the config whose `output_root` is a scratch
  folder outside `results/`.

## Numbers in review notes

- **Failure:** a review plan gave the forecast-head rank as "up to 37" without a computation.
  The value came from an estimate.
  **Rule:** compute every numeric claim with a script or cite the saved artifact. If the value
  is an estimate, label it as one or leave it out.

## Write hook on an existing plan file

- **Failure:** a Write call that replaced `plan.md` returned a Fact-Forcing Gate error. The
  same call passed on retry. The cause was not confirmed. It happened again on 2026-11-11 for
  `plan_ICML.md`, `plan.md`, and this file (the hook asks for callers, overlap, data files, and
  the user's words), and for the first Bash command of the session.
  **Rule:** when a hook blocks a write, present the facts the hook asks for, then retry the
  same call. For a small change to an existing file, use Edit.

- **Failure:** on 2026-11-16 the same hook denied the first Edit or Write of each file in the
  session, 15 times, one file at a time (generator, test, every manuscript section, `plan.md`).
  **Rule:** before a multi-file change, state the hook facts for all files in one message
  (callers, affected functions, data fields, the user's words), then send each first edit alone
  and retry it once. Do not batch a second edit of the same file with the denied first edit.

## Two generated macro files in one LaTeX document

- **Failure:** `build_strengthening.py --prefix cluster` and the default prefix wrote the same
  macro names (`\FollowupRuns`, ...) into two files. `main.tex` could not input both, because
  `\newcommand` rejects a duplicate name.
  **Rule:** a generator that runs once per result folder puts the folder's prefix in every macro
  name. After you add a macro file to `main.tex`, build the PDF; a clean build is the check.

## Overwriting untracked files

- **Failure:** `plan.md` and `plan_ICML.md` are untracked, so Git holds no copy. A Write
  replaced the first `plan_ICML.md` and dropped items (the ICML call details and the verified
  citation table) that the later merge had to rebuild from the chat.
  **Rule:** before a Write replaces an untracked file, copy it to the scratchpad, and carry
  every unique item into the new file. Ask before you delete the second file of a merge.

## Cluster resources versus local inventory

- **Failure:** a plan said "train DLinear on ECL and Solar, 12 runs" because
  `strengthening_status.tex` counted 60 blocked runs. That count came from the local
  inventory. `HPC_JOBS.md` lists 18 frozen ECL runs (CISSN, DLinear, PatchTST) on the cluster,
  and `slurm/paper_jobs.py` says only Solar lacks DLinear and PatchTST configurations.
  **Rule:** before you plan a training run, read `plan.md` section 5, `TASKS.md`, `slurm/paper_jobs.py`, and the
  cluster preflight output. A local "blocked" count describes this machine only.
- **Failure:** a plan described the local weights from the folder names alone. The local
  `checkpoints/publication` holds DLinear and PatchTST folders, but they are the runs without
  RevIN (no `fullrevin` in the name). The 36 blocked baseline runs on ETTh2, Weather, and
  Exchange need the RevIN weights, which are not on this machine.
  **Rule:** to state which weights exist, read the `reason` field of
  the suite's `inventory.json` (the October copy in `results/paper_strengthening` was deleted on
2026-11-16) and match the full run name, with its
  `fullrevin` part and hash, against `checkpoints/publication`.

## Diagnostics for calibrator design

- **Failure:** a plan chose a calendar grouping (`phase_ranked`) from one residual level per
  phase, averaged over the whole horizon. Calendar calibrators fit one quantile per group and
  horizon step, and the target phase moves with the step. The horizon average hid this and
  favoured the wrong grouping on Solar (0.95 against 0.78). The per-step profile reversed the
  order (0.40 to 0.50 against 0.82).
  **Rule:** before you compare partitions for a calibrator, measure them on the same unit the
  calibrator fits: per group, horizon step, and (where possible) channel. Score on the other
  calibration half, and give the chance level for random groups.

## Backups and removed files that documents cite

- **Failure:** `plan.md` and `manuscript/README.md` said that the old Elsevier manuscript
  sources were backed up under `results/_manuscript_layout_backup_20261112/`. On 2026-11-17 the
  folder was not on disk, after a cleanup of the `_*` scratch folders in `results/`. The
  documents kept the claim, and `manuscript/` was not in Git, so no copy is left.
  **Rule:** do not put a backup in a folder whose name matches a cleanup pattern
  (`results/_*`). Before you delete a folder or a document, grep the repository for its path,
  and correct every sentence that names it in the same change.
