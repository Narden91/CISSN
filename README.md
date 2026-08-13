# CISSN

CISSN is a state-space forecasting model with state-conditional conformal intervals. The publication protocol is defined only in [RUNBOOK.md](RUNBOOK.md).

## Start here

```powershell
uv sync
uv run python scripts/verify_datasets.py
uv run python tests/run_tests.py
```

Run a smoke test:

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 24 --train_epochs 2 --patience 2 --require_gpu --checkpoints ./checkpoints/smoke --results_dir ./results/smoke
```

## Method

The encoder produces a five-dimensional latent state: level, trend, a seasonal pair, and residual. A forecast head maps the final state to the horizon. The best validation checkpoint is restored before interval construction.

The state partition is fitted on train-split states, then frozen. Split-conformal quantiles are estimated on a later chronological calibration split. The default `per_feature` score produces marginal horizon-feature coverage; `max` is available for simultaneous block coverage. Temporal dependence is diagnosed and reported, never “corrected” by heuristic interval inflation.

## Implemented comparators

`dlinear`, `patchtst`, `deepstate`, `mc_dropout`, `deep_ensemble`, and the `flat_cp` ablation are launched through `experiments/`. Flat CP shares CISSN’s score geometry. MC-Dropout and Deep Ensemble are conformalized by default; their raw intervals are secondary diagnostics.

## Results contract

Each completed run saves predictions, targets, interval bounds when applicable, metrics, sanity checks, history, runtime, configuration, environment, dataset verification, and a protocol hash. Use `scripts/generate_publication_tables.py`, `scripts/generate_publication_figures.py`, and `scripts/generate_reproducibility_appendix.py` only after the required grid is complete.

## Documentation

- [RUNBOOK.md](RUNBOOK.md): runnable publication protocol and acceptance criteria.
- [docs/architecture.md](docs/architecture.md): implementation architecture.
- [docs/datasets.md](docs/datasets.md): datasets and split protocol.
- [docs/methodology.md](docs/methodology.md): scientific scope and reporting.
- [docs/flow.md](docs/flow.md): execution flow.
- [CLAUDE.md](CLAUDE.md): concise repository instructions.
