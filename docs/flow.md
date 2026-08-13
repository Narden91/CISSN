# Execution flow

`verify_dataset` -> chronological train / validation / calibration / test splits -> train with validation early stopping -> restore best checkpoint -> fit state partition on train states -> calibrate on calibration residuals -> evaluate once on test -> write complete artifacts -> aggregate only completed protocol-matched runs.

See `RUNBOOK.md` for the runnable commands and publication acceptance criteria.
