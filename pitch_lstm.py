"""
Train an LSTM classifier to predict PitchType from pose sequences.

Data sources:
- Labels: data/pitch_labels.csv (columns: PitchType, ID)
- Pose sequences: data/new_poses/<ID>.npy (shape: [T, 99])

Note: Not all videos have pose data; this script filters to IDs with .npy files.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    labels_csv: str
    poses_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float
    max_seq_len: int | None
    num_workers: int
    seed: int
    device: str
    grad_clip: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PoseDataset(Dataset):
    def __init__(
        self,
        ids: List[str],
        labels: List[int],
        poses_dir: Path,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        self.ids = ids
        self.labels = labels
        self.poses_dir = poses_dir
        self.mean = mean
        self.std = std
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        vid = self.ids[idx]
        arr = np.load(self.poses_dir / f"{vid}.npy").astype(np.float32)

        if self.max_seq_len is not None and arr.shape[0] > self.max_seq_len:
            arr = arr[-self.max_seq_len :]

        if self.mean is not None and self.std is not None:
            arr = (arr - self.mean) / (self.std + 1e-8)

        return torch.from_numpy(arr), int(self.labels[idx])


def collate_batch(batch: List[Tuple[torch.Tensor, int]]):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    # sort by length for pack_padded_sequence
    lengths, perm_idx = lengths.sort(descending=True)
    sequences = [sequences[i] for i in perm_idx]
    labels = torch.tensor([labels[i] for i in perm_idx], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, labels


class PitchLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            # h_n: (num_layers * 2, batch, hidden)
            last_fwd = h_n[-2]
            last_bwd = h_n[-1]
            feats = torch.cat([last_fwd, last_bwd], dim=1)
        else:
            feats = h_n[-1]

        return self.fc(feats)


def compute_stats(ids: List[str], poses_dir: Path, max_seq_len: int | None) -> Tuple[np.ndarray, np.ndarray]:
    total = np.zeros(99, dtype=np.float64)
    total_sq = np.zeros(99, dtype=np.float64)
    count = 0
    for vid in ids:
        arr = np.load(poses_dir / f"{vid}.npy").astype(np.float32)
        if max_seq_len is not None and arr.shape[0] > max_seq_len:
            arr = arr[-max_seq_len :]
        total += arr.sum(axis=0)
        total_sq += (arr ** 2).sum(axis=0)
        count += arr.shape[0]
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean**2
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, lengths, y in loader:
        # sanitize inputs in float32
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -5.0, 5.0)
        x = x.to(device=device, dtype=torch.float32)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        loss = criterion(logits.float(), y)
        if not torch.isfinite(loss):
            # Skip unstable batch
            continue
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, lengths, y in loader:
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -5.0, 5.0)
        x = x.to(device=device, dtype=torch.float32)
        lengths = lengths.to(device)
        y = y.to(device)
        logits = model(x, lengths)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        loss = criterion(logits.float(), y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_per_class_accuracy(model, loader, device, num_classes: int):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    for x, lengths, y in loader:
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -5.0, 5.0)
        x = x.to(device=device, dtype=torch.float32)
        lengths = lengths.to(device)
        y = y.to(device)
        logits = model(x, lengths)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        preds = logits.argmax(dim=1)
        for cls in range(num_classes):
            mask = y == cls
            total[cls] += mask.sum().item()
            correct[cls] += (preds[mask] == y[mask]).sum().item()
    acc = {int(i): (correct[i] / total[i] if total[i] > 0 else 0.0) for i in range(num_classes)}
    return acc, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM on pose sequences to predict PitchType")
    parser.add_argument("--labels-csv", default="data/pitch_labels.csv")
    parser.add_argument("--poses-dir", default="data/new_poses")
    parser.add_argument("--out-dir", default="baseball-z/outputs/pitch_lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-seq-len", type=int, default=330)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device; MPS enabled. Use 'mps' for Apple Silicon.",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        labels_csv=args.labels_csv,
        poses_dir=args.poses_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        grad_clip=args.grad_clip,
    )

    set_seed(cfg.seed)

    labels_df = pd.read_csv(cfg.labels_csv)
    labels_df = labels_df.drop(columns=[c for c in labels_df.columns if c.startswith("Unnamed")], errors="ignore")

    poses_dir = Path(cfg.poses_dir)
    pose_ids = {p.stem for p in poses_dir.glob("*.npy")}
    total_labels = len(labels_df)
    labels_df = labels_df[labels_df["ID"].isin(pose_ids)].copy()
    kept_labels = len(labels_df)
    dropped_labels = total_labels - kept_labels
    print(
        f"Labels: total {total_labels}, with poses {kept_labels}, dropped {dropped_labels} (no pose file)"
    )

    if labels_df.empty:
        raise RuntimeError("No labels match available pose files in data/new_poses.")

    label_encoder = LabelEncoder()
    labels_df["label"] = label_encoder.fit_transform(labels_df["PitchType"].astype(str))
    class_counts = labels_df["PitchType"].value_counts().sort_index()
    print(f"Classes ({len(label_encoder.classes_)}): {label_encoder.classes_.tolist()}")
    print("Class counts:", dict(class_counts))

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        labels_df["ID"].tolist(),
        labels_df["label"].tolist(),
        test_size=0.2,
        random_state=cfg.seed,
        stratify=labels_df["label"],
    )

    mean, std = compute_stats(train_ids, poses_dir, cfg.max_seq_len)

    train_ds = PoseDataset(train_ids, train_labels, poses_dir, mean, std, cfg.max_seq_len)
    val_ds = PoseDataset(val_ids, val_labels, poses_dir, mean, std, cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )

    if cfg.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    # Use float32 everywhere.
    torch.set_default_dtype(torch.float32)
    model = PitchLSTM(
        input_size=99,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=len(label_encoder.classes_),
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
    ).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg.grad_clip
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    meta = {
        "label_classes": label_encoder.classes_.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "config": asdict(cfg),
        "best_val_acc": best_acc,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    per_class_acc, per_class_total = eval_per_class_accuracy(
        model, val_loader, device, num_classes=len(label_encoder.classes_)
    )
    print("Per-class val accuracy:")
    for idx, name in enumerate(label_encoder.classes_):
        print(f"  {name}: {per_class_acc[idx]:.3f} (n={per_class_total[idx]})")

    # Plot training curves
    epochs = list(range(1, cfg.epochs + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png")
    plt.close()

    print(f"Best val acc: {best_acc:.3f}")
    print(f"Saved model to {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
