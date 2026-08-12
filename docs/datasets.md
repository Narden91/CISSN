# Dataset protocol

Publication runs use four registered datasets: `ETTh1`, `ETTh2`, `weather`, and `exchange_rate`. Their paths, dimensions, frequencies, horizons, integrity fingerprints, and structural checks live in `cissn/data/registry.py`.

Each run verifies the dataset before loading. The saved `protocol.json` records the observed SHA-256, registered fingerprint, configuration, source revision, environment, and deterministic protocol hash. A missing or structurally invalid dataset stops the run.

Splits remain chronological. The configured `cal_fraction` carves a calibration tail from the canonical training window; validation remains separate for early stopping and test data is not used for training, calibration, or tuning.
