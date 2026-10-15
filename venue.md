# CISSN paper — venue choice and how to raise the chance of acceptance

Updated 2026-10-15. Manuscript: `manuscript/main.tex` (IMRAD-style layout, 16 pages in the
neutral `article` class). Open runs: `experiments.md`.

## 1. Recommendation

1. Run the open experiments in `experiments.md` (sections 3 to 5) and finish the required items in section 4.
2. Submit to **COPA 2027** if the second confirmatory dataset (proposal P3, section 3) is
   ready by mid-April 2027. Otherwise submit to **TMLR** as soon as step 1 is done.
3. Use the **Journal of Forecasting** if both reject the paper, or if you want a forecasting
   audience instead of a conformal or ML audience.

Submit to one venue at a time. COPA proceedings in PMLR are archival, so a later journal
version must add substantial new content.

## 2. The three venues

| | COPA 2027 | TMLR | Journal of Forecasting |
| --- | --- | --- | --- |
| Type | Symposium, proceedings in PMLR | Journal, rolling submission | Journal (Wiley), rolling submission |
| Date | Deadline 1 May 2027; symposium 14–16 Sep 2027, University of Genoa | None | None |
| Length | "Typical papers should not exceed 20 pages" | No limit; length must fit the content | Journal limits; check the author guide |
| Review | Check the 2027 call | Double-blind; anonymise the PDF and the code link | Check the author guide |
| Template | PMLR | TMLR style file (required) | Wiley |
| Why it fits | Reviewers work on conformal prediction; cluster SCCP is a Mondrian calibrator and state-scaled CP is a normalised score | Acceptance rests on claims that match the evidence; novelty and state-of-the-art results are not required | Readers work on forecast evaluation: Winkler score, block bootstrap, comparison with DLinear |
| Known odds | No published rate | No published rate | About 22% listed (unit not stated) |
| Main risk | Reviewers expect a comparison with time-series conformal methods (P2) and may ask for a coverage statement | Reviewers ask for more experiments, first ECL baselines (P1) | Readers know less about conformal prediction; the background must grow |

A higher-impact option with lower odds: the International Journal of Forecasting accepted
10.6% of submissions in 2023 and gives 8–15% for recent years.

Sources: [COPA 2027](https://copa-conference.com/),
[TMLR acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html),
[TMLR author guide](https://jmlr.org/tmlr/author-guide.html),
[Journal of Forecasting on PJIP](https://www.pjip.org/journal/1003172/Journal%20of%20Forecasting),
[IJF 2023 editorial report](https://forecasters.org/wp-content/uploads/IJFreport2023.pdf).

## 3. Added experiments (`RUNBOOK.md` Step 7)

P1 to P3 are now `RUNBOOK.md` Step 7, with code, Slurm blocks `p1`, `p2`, `p3s`, `p3`, `p3b`, and
commands in `experiments.md`. P3 uses solar at stride 25; its design is in
`docs/prespecification_solar.md`. None of them changes the prespecified ECL endpoint, its
analysis, or its reading rule. A result added after the confirmatory analysis is reported as
such.

| ID | Experiment | What it fixes | Helps most at | Cost and conditions |
| --- | --- | --- | --- | --- |
| P1 | DLinear, PatchTST-lite, MC-Dropout, and Deep Ensemble under RevIN on ECL, h96 and h336, seeds 42, 123, 456 (24 runs) | The paper says "No baseline ran on ECL", so the point-accuracy result has no comparator on the confirmatory dataset | TMLR, J. Forecasting | Done in code: `run_baseline.py` takes `--test_origin_stride` (default 1) and scores the same 24-hour origins as CISSN. Descriptive only; the ECL test split is already open |
| P2 | Adaptive conformal inference (Gibbs and Candès, 2021) as a further calibrator on the same residuals of the same models | Related work cites time-series conformal methods but compares none | COPA | New code (the repo has no ACI). ACI updates with realised test errors, so it uses more information than the split calibrators; the text must say so. EnbPI needs bootstrap ensembles and costs much more |
| P3 | A second prespecified confirmatory test on `solar` (137 channels, 52,560 rows, untouched). Endpoint: cluster SCCP against flat CP at h336, as on ECL; per-cell state-scaled CP as a second endpoint with a multiplicity correction. Test origins at stride 25, which covers every time of day | The confirmatory evidence is one dataset at one horizon, and the Conclusions name a prespecified test of the per-cell scale as the next step | All three; most at COPA | Chosen over traffic: at h336 traffic gives 9.5 bootstrap blocks, solar 29.1. Draft in `docs/prespecification_solar.md`. `ETTm1` and `ETTm2` come from the same source as ETTh1 and ETTh2, so they are weaker choices |

## 4. Improvements without new experiments

Required for every venue:

1. Fill the `\todo` items: affiliation, co-authors, and the code and data statement.
2. Archive the code and the small run artifacts with a DOI (for example Zenodo), and cite it
   in the code and data statement. For TMLR, give an anonymised link at submission.
3. Check every `refs.bib` entry (pages, volume, DOI) against the publisher.
4. After E3, check that no text still calls the 12 baseline runs missing.

Per venue:

| Venue | Change |
| --- | --- |
| COPA | Move to the PMLR template and stay under 20 pages. State as a short proposition that cluster SCCP covers marginally within each cluster under within-cluster exchangeability (the Methods text already says it), with the short proof. Keep the statement that overlapping windows break exchangeability |
| TMLR | Move to the TMLR style file and anonymise the author block, the code link, and self-references. TMLR has no length limit, so all tables can stay in the main text |
| J. Forecasting | Move to the Wiley template. Add one paragraph that explains split conformal prediction and the Winkler score for forecasters. Keep point accuracy and interval results in separate subsections, as now |

Writing checks for every venue:

- The abstract and the introduction lead with the prespecified design and the ~1% result,
  as now. Do not let a revision turn "within the margin" into "better beyond the margin".
- Every conditioning result names the regime (RevIN).
- The reused components stay listed as reused (introduction, contributions list).
