#!/usr/bin/env python
"""Cell manifest for the cluster runs R1-R6 and P1-P3 (RUNBOOK.md; P1-P3 are Step 7).

One place lists every cell (one runner call). Each cell is parsed with the runner's own
argument parser, and its done-marker comes from the runner's own name builder, so a flag
typo or a naming change fails here and not hours into a job.

    python slurm/cells.py check                 # parse every cell, assert the cell counts
    python slurm/cells.py list BLOCK [-v]       # print the cells of one block and their state
    python slurm/cells.py status                # done/total per block
    python slurm/cells.py todo BLOCK            # number of cells left, or "undecided" (r3/r5 before R1)
    python slurm/cells.py decide                # R1 outcome: does RevIN win?
    python slurm/cells.py run BLOCK --task K --ntasks N   # what one array task executes

Blocks: smoke r1 r2 r3 r4 r4b r5 r6 p1 p2 p3s p3 p3b. r3 and r5 depend on the R1 outcome:
R5 runs only if R1 selects RevIN, R3 runs only if it does not. p1 (ECL baselines), p2 (ACI on
ECL), p3s (one-epoch memory check on the P3 dataset), p3 (P3 confirmatory runs), and p3b (ACI on
P3) are RUNBOOK Step 7. Run with the project venv python.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # every path below is relative to the repository root, as in experiments.md

PUB = "./results/publication"
CKPT_PUB = "./checkpoints/publication"
SEEDS = (42, 123, 456)
EXPECTED = {"smoke": 5, "r1": 12, "r2": 2, "r3": 14, "r4": 9, "r4b": 1, "r5": 216, "r6": 3,
            "p1": 24, "p2": 1, "p3s": 1, "p3": 6, "p3b": 1}
P3_DATA = "solar"  # the dataset of docs/prespecification_<P3_DATA>.md; preflight reads it from here
LOG_DIR = ROOT / "logs" / "slurm"  # inside the gitignored logs/ so --require_clean_git stays satisfied

TRAIN = ["--train_epochs", "20", "--patience", "5", "--lradj", "cosine"]
SEALED = ["--require_gpu", "--require_clean_git", "--immutable_artifacts", "--strict_determinism",
          "--evidence_role", "confirmation", "--checkpoints", CKPT_PUB, "--results_dir", PUB]
CISSN_COMMON = [*TRAIN, "--batch_size", "128", "--conformal_alpha", "0.1", "--n_clusters", "5",
                "--multivariate_strategy", "per_feature", "--conformal_conditioning", "cluster"]
BASELINE_COMMON = [*TRAIN, "--batch_size", "128", "--conformal_alpha", "0.1",
                   "--multivariate_strategy", "per_feature", *SEALED]
HORIZONS = {"ETTh1": (24, 96, 192, 336, 720), "ETTh2": (24, 96, 192, 336, 720),
            "weather": (96, 192, 336, 720), "exchange_rate": (96, 192, 336, 720)}
DAILY_ORIGINS = ["--test_origin_stride", "24"]  # one test origin per day on hourly data (prespecification)
# P3 stride 25 shares no factor with solar's daily period of 144, so the test origins cycle through
# every time of day. A stride equal to the period fixes one phase (docs/prespecification_solar.md).
P3_ORIGINS = ["--test_origin_stride", "25"]
ACI_ANALYSIS = ["--gamma", "0.005", "--block_multiplier", "1", "--n_boot", "5000", "--seed", "0"]


@dataclass
class Cell:
    block: str
    name: str            # human label
    runner: str          # path relative to ROOT
    argv: list[str]
    done: str            # glob (relative to ROOT) of the file the runner writes last
    orphans: str = ""    # glob of checkpoint dirs to delete when the cell is not done (immutable runs)
    always: bool = False  # smoke cells ignore the done marker


def _argv(*parts) -> list[str]:
    out: list[str] = []
    for part in parts:
        out += _argv(*part) if isinstance(part, (list, tuple)) else [str(part)]
    return out


def _benchmark(block, argv, root=PUB):
    from experiments.run_benchmark import build_setting_name, parse_args
    name = build_setting_name(parse_args(argv))
    return Cell(block, name, "experiments/run_benchmark.py", argv, f"{root}/{name}__*/metrics.json",
                f"{CKPT_PUB}/{name}__*")


def _selection(block, argv, root="./results/selection"):
    from experiments.run_benchmark import build_setting_name
    from experiments.run_selection import parse_args
    name = build_setting_name(parse_args(argv))
    return Cell(block, name, "experiments/run_selection.py", argv, f"{root}/{name}/selection.json")


def _baseline(block, argv, root=PUB):
    from experiments import run_baseline
    saved = sys.argv
    sys.argv = ["run_baseline.py", *argv]  # run_baseline.parse_args reads sys.argv
    try:
        name = run_baseline.build_setting_name(run_baseline.parse_args())
    finally:
        sys.argv = saved
    return Cell(block, name, "experiments/run_baseline.py", argv, f"{root}/{name}__*/metrics.json",
                f"{CKPT_PUB}/{name}__*")


def _ablation(block, argv, output):
    from experiments.run_ablation import parse_ablation_args
    parse_ablation_args(argv)
    return Cell(block, Path(output).name, "experiments/run_ablation.py", argv, output)


def block_smoke() -> list[Cell]:
    # Short runs into results/smoke, deleted afterwards. ECL goes through the validation-only
    # controller so its test split stays sealed; ETTh1 test is already exposed.
    out = ["--checkpoints", "./checkpoints/smoke", "--results_dir", "./results/smoke"]
    one = ["--train_epochs", "1", "--patience", "5", "--lradj", "cosine", "--batch_size", "128"]
    gpu = ["--require_gpu", "--require_clean_git"]
    sealed = [*gpu, "--immutable_artifacts", "--strict_determinism", "--evidence_role", "confirmation", *out]
    cells = [
        _selection("smoke", _argv("--data", "ECL", "--pred_len", 336, "--seed", 0, "--revin", one, gpu, out),
                   root="./results/smoke"),
        _selection("smoke", _argv("--data", "ECL", "--pred_len", 336, "--seed", 0, "--architecture", "hybrid",
                                  "--state_dynamics", "anchored", "--lambda_cov", 0, "--lambda_temp", 0, one, gpu,
                                  out), root="./results/smoke"),
        _benchmark("smoke", _argv("--data", "ETTh1", "--pred_len", 96, "--seed", 0, "--revin", one,
                                  "--conformal_alpha", "0.1", "--n_clusters", 5, sealed), root="./results/smoke"),
        _baseline("smoke", _argv("--model", "dlinear", "--data", "ETTh1", "--pred_len", 96, "--seed", 0, one,
                                 sealed), root="./results/smoke"),
        _baseline("smoke", _argv("--model", "deep_ensemble", "--revin", "--data", "ETTh1", "--pred_len", 96,
                                 "--seed", 0, one, "--conformal_alpha", "0.1", sealed), root="./results/smoke"),
    ]
    for cell in cells:
        cell.always = True
    return cells


def block_r1() -> list[Cell]:
    return [_selection("r1", _argv(
        "--data", data, "--pred_len", 336, "--seed", seed, ["--revin"] if revin else [], TRAIN,
        "--require_gpu", "--require_clean_git", "--checkpoints", "./checkpoints/selection",
        "--results_dir", "./results/selection"))
        for data in ("ETTh2", "weather") for revin in (True, False) for seed in SEEDS]


def block_r2() -> list[Cell]:
    return [_benchmark("r2", _argv("--data", "weather", "--pred_len", 192, "--seed", seed, "--revin",
                                   CISSN_COMMON, "--scale_geometry", "scalar", SEALED)) for seed in (42, 456)]


def block_r3() -> list[Cell]:
    grid = [("weather", 720, seed) for seed in (123, 456)]
    grid += [("exchange_rate", h, seed) for h in HORIZONS["exchange_rate"] for seed in SEEDS]
    return [_baseline("r3", _argv("--model", "deep_ensemble", "--data", data, "--pred_len", h, "--seed", seed,
                                  BASELINE_COMMON)) for data, h, seed in grid]


def block_r4() -> list[Cell]:
    ecl = [CISSN_COMMON, "--test_origin_stride", 24, SEALED]
    cells = [_benchmark("r4", _argv("--data", "ECL", "--pred_len", h, "--seed", seed, "--revin",
                                    "--scale_geometry", "scalar", ecl)) for h in (336, 96) for seed in SEEDS]
    cells += [_benchmark("r4", _argv("--data", "ECL", "--pred_len", 336, "--seed", seed, "--architecture",
                                     "hybrid", "--state_dynamics", "anchored", "--lambda_cov", 0, "--lambda_temp", 0,
                                     ecl)) for seed in SEEDS]
    return cells


def block_r4b() -> list[Cell]:
    # Reloads the R4 hybrid checkpoints and compares them with the saved forecasts (tolerance 1e-3).
    # It runs here, on the GPU type that trained them, not on the workstation.
    out = f"{PUB}/hybrid_vs_base_ECL.json"
    argv = _argv("--results_root", PUB, "--checkpoints_root", CKPT_PUB, "--data", "ECL", "--pred_len", 336,
                 "--seeds", *SEEDS, "--output", out)
    return [Cell("r4b", "hybrid_vs_base_ECL", "scripts/paired_point_comparison.py", argv, out)]


def block_r5() -> list[Cell]:
    return [_baseline("r5", _argv("--model", model, "--revin", "--data", data, "--pred_len", h, "--seed", seed,
                                  BASELINE_COMMON))
            for model in ("dlinear", "patchtst", "mc_dropout", "deep_ensemble")
            for data, hs in HORIZONS.items() for h in hs for seed in SEEDS]


def block_r6() -> list[Cell]:
    cells = []
    for seed in SEEDS:
        out = f"{PUB}/ablations_ETTh1_h336_s{seed}.json"
        argv = _argv("--data", "ETTh1", "--pred_len", 336, "--seed", seed, "--ablations", "all", TRAIN,
                     "--batch_size", 128, "--conformal_alpha", "0.1", "--multivariate_strategy", "per_feature",
                     "--require_gpu", "--require_clean_git", "--output", out)
        cells.append(_ablation("r6", argv, out))
    return cells


def block_p1() -> list[Cell]:
    # P1: point baselines on ECL, scored on the same daily origins as the R4 CISSN runs.
    return [_baseline("p1", _argv("--model", model, "--revin", "--data", "ECL", "--pred_len", h, "--seed", seed,
                                  DAILY_ORIGINS, BASELINE_COMMON))
            for model in ("dlinear", "patchtst", "mc_dropout", "deep_ensemble") for h in (336, 96) for seed in SEEDS]


def _aci(block: str, data: str) -> Cell:
    # Reloads the CISSN checkpoints, so it runs here, on the GPU type that trained them.
    out = f"{PUB}/aci_vs_flat_{data}.json"
    argv = _argv("--results_root", PUB, "--checkpoints_root", CKPT_PUB, "--datasets", data, ACI_ANALYSIS,
                 "--output", out)
    return Cell(block, f"aci_vs_flat_{data}", "scripts/aci_paired_comparison.py", argv, out)


def block_p2() -> list[Cell]:
    return [_aci("p2", "ECL")]


def block_p3s() -> list[Cell]:
    # One epoch on the P3 dataset through the validation-only controller, before P3 starts: it
    # checks memory with the per-cell scale regression and never builds the test loader.
    out = ["--checkpoints", "./checkpoints/smoke_p3", "--results_dir", "./results/smoke_p3"]
    return [_selection("p3s", _argv("--data", P3_DATA, "--pred_len", 336, "--seed", 0, "--revin",
                                    "--scale_geometry", "per_cell", "--train_epochs", 1, "--patience", 5,
                                    "--lradj", "cosine", "--batch_size", 128, "--require_gpu",
                                    "--require_clean_git", out), root="./results/smoke_p3")]


def block_p3() -> list[Cell]:
    # P3: second prespecified confirmatory test (docs/prespecification_<P3_DATA>.md). Per-cell
    # geometry makes the run's state-scaled CP the secondary endpoint's calibrator.
    return [_benchmark("p3", _argv("--data", P3_DATA, "--pred_len", h, "--seed", seed, "--revin", CISSN_COMMON,
                                   "--scale_geometry", "per_cell", P3_ORIGINS, SEALED))
            for h in (336, 96) for seed in SEEDS]


def block_p3b() -> list[Cell]:
    return [_aci("p3b", P3_DATA)]


BLOCKS = {"smoke": block_smoke, "r1": block_r1, "r2": block_r2, "r3": block_r3, "r4": block_r4,
          "r4b": block_r4b, "r5": block_r5, "r6": block_r6,
          "p1": block_p1, "p2": block_p2, "p3s": block_p3s, "p3": block_p3, "p3b": block_p3b}


def decide() -> dict:
    """R1 rule (experiments.md): RevIN wins only if, on both datasets, its mean validation MSE is
    not higher and its mean primary coverage is not lower than the non-RevIN arm."""
    arms: dict[tuple[str, bool], list[tuple[float, float]]] = {}
    missing = []
    for cell in block_r1():
        files = glob.glob(cell.done)
        if not files:
            missing.append(cell.name)
            continue
        run = json.loads(Path(files[0]).read_text(encoding="utf-8"))
        key = (run["config"]["data"], bool(run["config"]["revin"]))
        arms.setdefault(key, []).append((run["validation_mse"], run["interval"]["coverage"]))
    if missing:
        raise SystemExit(f"R1 incomplete: {len(missing)} of {EXPECTED['r1']} selection.json missing, "
                         f"first {missing[0]}")

    def mean(rows, i):
        return sum(r[i] for r in rows) / len(rows)

    detail, wins = {}, True
    for data in ("ETTh2", "weather"):
        on, off = arms[(data, True)], arms[(data, False)]
        ok = mean(on, 0) <= mean(off, 0) and mean(on, 1) >= mean(off, 1)
        detail[data] = {"val_mse_revin": mean(on, 0), "val_mse_no_revin": mean(off, 0),
                        "coverage_revin": mean(on, 1), "coverage_no_revin": mean(off, 1), "revin_not_worse": ok}
        wins = wins and ok
    result = {"revin_selected": wins, "datasets": detail}
    Path("./results/selection/r1_decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cells_for(block: str, force: bool = False) -> list[Cell]:
    """Cells to run for a block. r3/r5 are empty when the R1 outcome excludes them."""
    if block in ("r3", "r5") and not force:
        if (block == "r5") != decide()["revin_selected"]:
            return []
    return BLOCKS[block]()


def is_done(cell: Cell) -> bool:
    return not cell.always and bool(glob.glob(cell.done))


def cmd_check(_args) -> int:
    bad = 0
    for block, build in BLOCKS.items():
        cells = build()
        names = [c.name for c in cells]
        dup = len(names) != len(set(names))
        ok = len(cells) == EXPECTED[block] and not dup
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {block:<6} {len(cells):>4} cells (expected {EXPECTED[block]})"
              f"{' DUPLICATE NAMES' if dup else ''}")
    return 1 if bad else 0


def cmd_list(args) -> int:
    cells = cells_for(args.block, force=args.force)
    for i, cell in enumerate(cells):
        print(f"{i:>3} {'done' if is_done(cell) else 'todo'} {cell.name}")
        if args.verbose:
            print(f"      python {cell.runner} {shlex.join(cell.argv)}")
    print(f"{len(cells)} cells, {sum(not is_done(c) for c in cells)} to do")
    return 0


def cmd_status(_args) -> int:
    for block in BLOCKS:
        if block == "smoke":
            continue
        try:
            cells = cells_for(block)
            note = "" if cells else " (excluded by the R1 outcome)"
        except SystemExit:
            cells, note = BLOCKS[block](), " (R1 not finished: inclusion undecided)"
        print(f"{block:<4} {sum(is_done(c) for c in cells):>3}/{len(cells):<3}{note}")
    return 0


def cmd_todo(args) -> int:
    try:
        print(sum(not is_done(c) for c in cells_for(args.block)))
    except SystemExit:
        print("undecided")
    return 0


def cmd_decide(_args) -> int:
    print(json.dumps(decide(), indent=2))
    return 0


def _remove_orphans(cell: Cell) -> None:
    # An interrupted immutable run leaves a checkpoint dir, and the runner refuses to start over an
    # existing one. The cell is not done (no metrics.json), so that directory is a partial run.
    for path in glob.glob(cell.orphans) if cell.orphans else []:
        print(f"[cells] removing partial checkpoint of an unfinished run: {path}", flush=True)
        shutil.rmtree(path)


def _clean_smoke() -> None:
    for d in ("results/smoke", "checkpoints/smoke"):
        shutil.rmtree(d, ignore_errors=True)


def cmd_run(args) -> int:
    cells = cells_for(args.block)[args.task::args.ntasks]  # partition first, then skip finished cells
    print(f"[cells] block {args.block} task {args.task}/{args.ntasks}: {len(cells)} cells", flush=True)
    if args.block == "smoke":
        _clean_smoke()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for cell in cells:
        if is_done(cell):
            print(f"[cells] skip (done) {cell.name}", flush=True)
            continue
        _remove_orphans(cell)
        print(f"[cells] start {cell.name}\n[cells] python {cell.runner} {shlex.join(cell.argv)}", flush=True)
        code = subprocess.run([sys.executable, cell.runner, *cell.argv]).returncode
        if code == 0 and not glob.glob(cell.done):
            code = 99  # exit 0 but the done marker is missing
        print(f"[cells] end {cell.name} exit {code}", flush=True)
        if code != 0:
            failed.append(cell.name)
            with open(LOG_DIR / f"failed_{args.block}.txt", "a", encoding="utf-8") as f:
                f.write(f"{os.environ.get('SLURM_JOB_ID', 'manual')}\t{cell.name}\texit {code}\n")
    if args.block == "smoke":
        _clean_smoke()
    if failed:
        print(f"[cells] {len(failed)} of {len(cells)} cells failed: {failed}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("decide").set_defaults(func=cmd_decide)
    p = sub.add_parser("todo")
    p.add_argument("block", choices=BLOCKS)
    p.set_defaults(func=cmd_todo)
    p = sub.add_parser("list")
    p.add_argument("block", choices=BLOCKS)
    p.add_argument("-v", "--verbose", action="store_true", help="print the full command")
    p.add_argument("--force", action="store_true", help="ignore the R1 outcome for r3/r5")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("run")
    p.add_argument("block", choices=BLOCKS)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--ntasks", type=int, default=1)
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
