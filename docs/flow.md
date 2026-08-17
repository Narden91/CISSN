# Execution flow

`verify_dataset` -> chronological train / calibration / validation / test splits -> train with validation early stopping -> restore best checkpoint -> fit conditioning predictors (cluster partition and sigma regression) on the calibration-half `conditioning_states` -> calibrate quantiles on the remaining calibration half -> evaluate once on test -> write complete artifacts -> aggregate only completed protocol-matched runs.

See `RUNBOOK.md` for the runnable commands and publication acceptance criteria.
