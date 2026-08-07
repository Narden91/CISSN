#!/usr/bin/env python
"""
Launcher for CISSN Conference Paper Experiments.
Executes the targeted grid defined in manuscript/conference_paper_plan.md.
Ensures clean, robust, and trackable execution using Python's subprocess.
"""
import subprocess
import sys
import time
from pathlib import Path

# ==============================================================================
# Experiment Configuration
# ==============================================================================
DATASETS = ["ETTh1", "weather"]
HORIZONS = [96, 336]
SEEDS = "42,123,456"
BASELINES = ["deepstate", "patchtst", "dlinear"]
ABLATIONS = "full,no_structured_A,state_dim_4"
CONFORMAL_ALPHA = "0.1"

def run_cmd(cmd: list[str]) -> None:
    """Run a shell command and handle potential errors cleanly."""
    print(f"\n[{time.strftime('%H:%M:%S')}] RUNNING: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}:")
        print(f"  {' '.join(cmd)}")
        print("Continuing to the next experiment...")

def main():
    print("=====================================================")
    print(" Starting CISSN Conference Paper Experiment Grid")
    print("=====================================================\n")
    print(f"Datasets : {DATASETS}")
    print(f"Horizons : {HORIZONS}")
    print(f"Seeds    : {SEEDS}")
    print(f"Baselines: {BASELINES}")
    print(f"Ablations: {ABLATIONS}")
    print("=====================================================\n")
    
    t0 = time.time()
    
    # 1. Main CISSN Benchmark
    print("\n--- Phase 1: CISSN Main Benchmark ---")
    for dataset in DATASETS:
        for h in HORIZONS:
            cmd = [
                sys.executable, "experiments/run_multiseed.py",
                "--data", dataset,
                "--pred_len", str(h),
                "--seeds", SEEDS,
                "--multivariate_strategy", "max",
                "--conformal_alpha", CONFORMAL_ALPHA,
                "--patience", "5",
                "--n_clusters", "5",
                "--output", f"./results/conference/cissn_{dataset}_h{h}.json",
                "--raw_csv", f"./results/conference/cissn_{dataset}_h{h}_raw.csv"
            ]
            run_cmd(cmd)
            
    # 2. Baselines
    print("\n--- Phase 2: Baselines ---")
    for model in BASELINES:
        for dataset in DATASETS:
            for h in HORIZONS:
                for seed in SEEDS.split(','):
                    cmd = [
                        sys.executable, "experiments/run_baseline.py",
                        "--model", model,
                        "--data", dataset,
                        "--pred_len", str(h),
                        "--seed", seed,
                        "--conformal_alpha", CONFORMAL_ALPHA,
                        "--patience", "5",
                        "--checkpoints", "./checkpoints/conference/baselines",
                        "--results_dir", "./results/conference/baselines"
                    ]
                    run_cmd(cmd)
                    
    # 3. Ablations
    print("\n--- Phase 3: Ablations ---")
    for dataset in DATASETS:
        for h in HORIZONS:
            for seed in SEEDS.split(','):
                # Ensure the ablation directory exists before saving json
                Path(f"./results/conference/ablations").mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, "experiments/run_ablation.py",
                    "--data", dataset,
                    "--pred_len", str(h),
                    "--seed", seed,
                    "--ablations", ABLATIONS,
                    "--output", f"./results/conference/ablations/{dataset}_h{h}_s{seed}.json"
                ]
                run_cmd(cmd)
                
    elapsed = (time.time() - t0) / 60
    print(f"\n=====================================================")
    print(f" All Conference Experiments Completed in {elapsed:.1f} minutes!")
    print(" Results saved in ./results/conference/")
    print("=====================================================")

if __name__ == "__main__":
    main()
