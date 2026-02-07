"""
Grid search runner for pitch_cnn_lstm_atten.py.

Runs a small hyperparameter grid and saves results to CSV/JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class RunSpec:
    max_seq_len: int
    hidden_size: int
    lr: float
    dropout: float
    epochs: int


def run_one(spec: RunSpec, base_out: Path, device: str) -> dict:
    run_name = (
        f"len{spec.max_seq_len}_hid{spec.hidden_size}_"
        f"uni_lr{spec.lr}_drop{spec.dropout}"
    )
    out_dir = base_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "baseball-z/pitch_cnn_lstm_atten.py",
        "--device",
        device,
        "--attention",
        "simple",
        "--epochs",
        str(spec.epochs),
        "--max-seq-len",
        str(spec.max_seq_len),
        "--hidden-size",
        str(spec.hidden_size),
        "--lr",
        str(spec.lr),
        "--dropout",
        str(spec.dropout),
        "--out-dir",
        str(out_dir),
        "--no-class-weights",
    ]

    print(f"Running: {run_name}")
    subprocess.run(cmd, check=True)

    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        best_val_acc = meta.get("best_val_acc", None)
    else:
        best_val_acc = None

    return {
        "run_name": run_name,
        **asdict(spec),
        "best_val_acc": best_val_acc,
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out-dir", default="baseball-z/outputs/grid_search_cnn_lstm_atten")
    args = parser.parse_args()

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    grid = {
        "max_seq_len": [180, 240, 300, 330],
        "hidden_size": [128, 192],
        "lr": [3e-4, 1e-4],
        "dropout": [0.2, 0.3],
    }

    specs: List[RunSpec] = []
    for values in itertools.product(
        grid["max_seq_len"],
        grid["hidden_size"],
        grid["lr"],
        grid["dropout"],
    ):
        specs.append(
            RunSpec(
                max_seq_len=values[0],
                hidden_size=values[1],
                lr=values[2],
                dropout=values[3],
                epochs=args.epochs,
            )
        )

    results = []
    for spec in specs:
        results.append(run_one(spec, base_out, args.device))

    # Save results
    json_path = base_out / "grid_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    # CSV
    csv_path = base_out / "grid_results.csv"
    headers = list(results[0].keys()) if results else []
    lines = [",".join(headers)]
    for r in results:
        lines.append(",".join(str(r[h]) for h in headers))
    csv_path.write_text("\n".join(lines))

    # Print best
    best = max(results, key=lambda r: r["best_val_acc"] or -1)
    print("Best run:", best)


if __name__ == "__main__":
    main()
