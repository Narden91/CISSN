# TASKS: finish the TMLR submission

This is the only to-do list. Follow it from top to bottom. Target: submit by 15 January 2027.

This file holds the open steps and their commands. `plan.md` holds the decisions, the evidence,
the fixed design, the result checks, and the rules. No other file is a plan or a to-do list.

Each step says who does it: **[you]** or **[agent]**. Workstation commands are PowerShell from the
repository root. Cluster commands are bash from the cluster checkout.

## State on 2026-11-17 (each line checked today)

- The evidence for the paper is complete. The reading rule gives outcome A on ECL and on Solar:
  a calendar partition scored at least as well as the learned state. The paper is a
  matched-control study with a negative result.
- The manuscript builds with no undefined reference: 46 pages. The main text ends on page 17,
  the appendix starts on page 20. The target is 12 pages of main text.
- Tests: 60 passed (`tests/test_paper_suite.py`, `tests/test_paper_jobs.py`).
- The working tree is not clean. The `.gitignore` edit makes `manuscript/` and `plan.md`
  visible to Git as untracked files, and the document merge of 2026-11-17 is not committed
  (`plan.md` Appendix C). The sealed training runs refuse a tree that is not clean.
- Solar DLinear and PatchTST-lite (block `p5`): 0 of 12 runs trained.
- The CISSN Solar weights are not on the workstation. They are on the cluster only.
- No manuscript text has the words `pending`, `todo`, or `not yet`.

Only one experiment is open: the Solar strong-forecaster comparison (steps 1 to 3). It is
optional. Without it the paper keeps the sentence "No DLinear or PatchTST-lite model was trained
on Solar". Steps 4 and 5 do not wait for it.

## Step 0. [you] Commit, so the tree is clean (5 minutes)

`manuscript/submission_metadata.md` holds your name and declaration. The first line keeps it out
of Git, so an archive made from the repository stays anonymous.

- [ ] Run:

```powershell
Add-Content .gitignore "`nmanuscript/submission_metadata.md"
git add -A
git status --short
```

- [ ] Read the staged list. It must show no `build/` file, no `.pdf`, and no
  `submission_metadata.md`. It shows `HPC_JOBS.md`, `experiments.md`, and `venue.md` as deleted,
  `RUNBOOK.md` as deleted or renamed to `docs/RUNBOOK.md`, and `plan.md`, `TASKS.md`, and the
  `manuscript/` sources as new.
- [ ] Run:

```powershell
git commit -m "Track manuscript sources; merge run documents into plan.md and TASKS.md"
git push
git status --short
git log -1 --oneline
```

- [ ] `git status --short` prints nothing. Write the commit hash here: `________`

## Step 1. [you] Train the 12 Solar runs on the RTX 5080 (estimate: 4.6 hours)

The estimate comes from a one-epoch measurement: about 11 minutes per DLinear run and 35 minutes
per PatchTST-lite run at 20 epochs. The command skips each finished run, so after an
interruption run the same line again. Keep the laptop on mains power and stop it from sleeping.

- [ ] Run:

```powershell
powercfg /change standby-timeout-ac 0
git status --short
.\.venv\Scripts\python.exe slurm/cells.py list p5
$env:GRES = "gpu:1"; .\.venv\Scripts\python.exe slurm/cells.py run p5 --task 0 --ntasks 1 *> "logs/p5_$(Get-Date -Format yyyyMMdd_HHmmss).log"
.\.venv\Scripts\python.exe slurm/cells.py list p5
```

- [ ] Before the run, the list ends with `12 cells, 12 to do`. After it: `12 cells, 0 to do`.
- [ ] To watch progress from a second terminal:

```powershell
Get-Content (Get-ChildItem logs/p5_*.log | Sort-Object LastWriteTime | Select-Object -Last 1) -Tail 5 -Wait
```

Do not edit code or commit while the runs train. Each run records the commit and needs a clean
tree when it starts.

## Step 2. [you] Score Solar with the three forecasters (cluster, CPU only)

The scoring needs the CISSN Solar weights, and those are on the cluster. The job trains nothing
and needs no GPU. Do not train CISSN on Solar again: new weights would replace the model of the
prespecified Solar test.

The cluster login works (user, 2026-11-17). The one-time setup, the watch and cancel commands,
and the failure table are in `slurm/CINECA_RUNBOOK.md`.

- [ ] **Workstation.** Replace the three values in the first line, then run the block:

```powershell
$env:CU = "<cineca user>"; $env:LOGIN = "<login host>"; $env:WORKDIR = "<work folder>"
$runs = Get-ChildItem results/publication -Directory | Where-Object Name -match '^BASELINE_(dlinear|patchtst)_solar_'
$weights = Get-ChildItem checkpoints/publication -Directory | Where-Object Name -match '^BASELINE_(dlinear|patchtst)_solar_' | Get-ChildItem -File -Filter *.pth
"$($runs.Count) runs, $($weights.Count) weight files"
$paths = @($runs) + @($weights) | ForEach-Object { $_.FullName.Substring($PWD.Path.Length + 1) -replace '\\', '/' }
tar -czf results/p5_solar_baselines.tar.gz --exclude '*.npy' @paths
scp results/p5_solar_baselines.tar.gz "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/"
```

- [ ] The count line prints `12 runs, 24 weight files`.
- [ ] **Cluster.** Update the checkout and unpack:

```bash
git status --short
git pull --ff-only
git log -1 --oneline
tar -xzf p5_solar_baselines.tar.gz && rm p5_solar_baselines.tar.gz
git status --short
.venv/bin/python slurm/cells.py list p5
```

- [ ] Both `git status --short` calls print nothing. The log shows the step 0 commit. The list
  ends with `12 cells, 0 to do`.
- [ ] **Cluster.** Check the inputs, preview, submit (`GRES=""` in `.env`, as on 2026-11-16):

```bash
bash slurm/paper_preflight.sh --datasets solar
bash slurm/launch_paper.sh --datasets solar --dry-run
bash slurm/launch_paper.sh --datasets solar
```

- [ ] The preflight ends with `Paper preflight OK`.
- [ ] **Cluster.** Wait for the job, then check the log:

```bash
squeue -u "$(id -un)"
tail -n 20 $(ls -t logs/slurm/cissn_paper_solar*.out | head -n 1)
```

- [ ] The log ends with `End UTC:`. If the job failed, read the log, then tell the agent the
  last 20 lines.
- [ ] **Cluster.** Merge ECL and Solar and pack the report inputs:

```bash
.venv/bin/python slurm/paper_jobs.py merge --datasets ECL,solar --output results/paper_cluster
tar --exclude='cache' -czf results/paper_cluster_report.tar.gz -C results paper_cluster
```

- [ ] **Workstation.** Same PowerShell session as the upload (it holds the three values):

```powershell
$bundleDir = "results/cluster_complete_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $bundleDir | Out-Null
scp "${env:CU}@${env:LOGIN}:${env:WORKDIR}/CISSN/results/paper_cluster_report.tar.gz" "$bundleDir/"
tar -xzf "$bundleDir/paper_cluster_report.tar.gz" -C "$bundleDir"
"$bundleDir"
```

- [ ] Write the folder name here: `results/cluster_complete_________`

## Step 3. [agent] Read the Solar result and update the paper

Ask: "Do TASKS.md step 3 with folder `<name from step 2>`."

- [ ] Apply `plan.md` section 7.1 and 7.2 to the new folder before any comparison is read.
- [ ] Rebuild the `cluster` tables and macros from the new folder.
- [ ] Put the new folder name in `plan.md` section 5, `CLAUDE.md`, `manuscript/README.md`, and
  step 5 of this file.
- [ ] Add a new table for the Solar point results. Do not change `tab_point.tex`.
- [ ] Replace the sentence "No DLinear or PatchTST-lite model was trained on Solar"
  (`sections/07_results_strong_forecaster_aci.tex`, lines 58 to 59) and the matching limit in
  `sections/08_discussion.tex` with the result. The result is descriptive.
- [ ] State in the provenance appendix that the 12 Solar baseline runs trained on an RTX 5080
  at the step 0 commit, while every other publication run trained on H100 nodes. Seeds give
  different weights on the two GPUs.
- [ ] Rebuild the PDF. No undefined reference.

## Step 4. [agent] Manuscript work that can start now

Ask: "Do TASKS.md step 4." None of this waits for steps 1 to 3.

- [ ] Write the Phase 6 numbers as generated macros: the intervals at block multipliers 2 and 4
  and the ratio-of-means intervals, from `results/phase6_statistics`. Today the appendix has
  only the figure and one qualitative sentence.
- [ ] Shorten the main text. It ends on page 17 and the target is 12. Move tables to the
  appendix in the order of `plan.md` section 8.4. Keep every table that an abstract claim
  rests on. The target is not a condition: above 12 pages TMLR allows reviewers 4 weeks, not 2.
- [ ] Check that every number in the Results and Discussion is a macro or a fixed constant.
- [ ] Confirm that the text has each replacement of `plan.md` sections 8.2 and 8.3.
- [ ] Replace the Elsevier heading of the AI statement in `sections/09_conclusion.tex` with a
  plain one. The statement is one sentence already.
- [ ] Rebuild the PDF. Record the page on which the references start: `____`

## Step 5. [you and agent] Final checks, then submit

- [ ] [agent] Regenerate every input and build. The figures and CSV files are not in Git, so a
  clean copy needs these commands before `latexmk`. After step 3, use the new folder name in the
  third command:

```powershell
.\.venv\Scripts\python.exe manuscript/analysis/build_numbers.py
.\.venv\Scripts\python.exe manuscript/analysis/make_figures.py
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py --results results/cluster_complete_20261116_161806/paper_cluster --prefix cluster
.\.venv\Scripts\python.exe manuscript/analysis/build_strengthening.py --results results/paper_strengthening_v2
.\.venv\Scripts\python.exe manuscript/analysis/plot_origin_difference.py
Push-Location manuscript; latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; Pop-Location
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp "results/_pytest_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

- [ ] [agent] Search `manuscript` for `pending`, `todo`, `not yet`, and
  `International Journal of Forecasting`. None may stay outside the references.
- [ ] [agent] Build the anonymous archive, under 100 MB: code, generated files, run
  configurations, environment files, regeneration commands, and a list of the omitted weights.
  Search it for your name, e-mail, Windows user name, machine names, and cluster account. Check
  `analysis_manifest.json`, `environment.json`, `config.json`, and `slurm/env.sh` first.
- [ ] [you] Give the PDF to one reader outside the project. List their questions.
- [ ] [you] Read the abstract against `plan.md` section 4.3 (allowed claims for outcome A).
- [ ] [you] Submit on the TMLR OpenReview page: PDF and supplementary archive.

## Closed: do not reopen without a reviewer request

- No new dataset and no new prespecified test.
- No CISSN training on Solar or ECL on the workstation.
- No untrained-encoder control, no ResCP, KOWCPI, CT-SSF, or RareCP comparison. The related
  work says that the paper compares with none of them.
- No search for the RevIN baseline weights of ETTh2, Weather, and Exchange.
- No wider `flat_scaled` grid.

## Schedule

| Week of | Work |
| - | - |
| 17 Nov | Steps 0 and 1 (you). Step 4 (agent). |
| 24 Nov | Step 2 (you). Step 3 (agent). |
| 1 to 15 Dec | Read and revise the full text. |
| 5 Jan | Step 5. |
| 15 Jan 2027 | Submit. |
