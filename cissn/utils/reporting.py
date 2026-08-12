"""Concise terminal reporting shared by experiment runners."""
from __future__ import annotations


def print_run_header(title: str, args, setting: str) -> None:
    protocol = getattr(args, "protocol", {}) or {}
    source = protocol.get("source", {})
    protocol_hash = protocol.get("protocol_hash", "unknown")[:12]
    model = getattr(args, "model", "cissn")
    print(f"\n{'=' * 78}")
    print(f"{title}: {model} | {args.data} | horizon={args.pred_len} | seed={args.seed}")
    print(f"Run:      {setting}")
    print(f"Device:   {source.get('device', 'unknown')} | Protocol: {protocol_hash}")
    print(
        f"Training: {args.train_epochs} epochs | batch={args.batch_size} | "
        f"lr={args.learning_rate:g} ({args.lradj}) | patience={args.patience}"
    )
    print(
        f"Intervals: alpha={args.conformal_alpha:g} | {args.multivariate_strategy} | "
        f"calibration={args.cal_fraction:g}"
    )
    print(f"Artifacts: {args.results_dir}")
    print(f"{'=' * 78}")


def print_epoch_summary(
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
    elapsed_seconds: float,
    improved: bool,
    patience_counter: int,
    patience: int,
) -> None:
    if improved:
        status = "best"
    elif patience_counter >= patience:
        status = "stopped"
    else:
        status = f"wait {patience_counter}/{patience}"
    print(
        f"Epoch {epoch:02d}/{total_epochs} | train={train_loss:.6f} | "
        f"val={validation_loss:.6f} | lr={learning_rate:.2e} | "
        f"{status} | {elapsed_seconds:.1f}s"
    )
