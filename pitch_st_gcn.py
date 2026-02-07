"""
ST-GCN model for PitchType classification.

Data sources:
- Labels: data/pitch_labels.csv (columns: PitchType, ID)
- Pose sequences: data/new_poses/<ID>.npy (shape: [T, 99])

Note: Not all videos have pose data; this script filters to IDs with .npy files.
"""

from __future__ import annotations

import argparse
import json
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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    labels_csv: str
    poses_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
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
    lengths, perm_idx = lengths.sort(descending=True)
    sequences = [sequences[i] for i in perm_idx]
    labels = torch.tensor([labels[i] for i in perm_idx], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, labels


def compute_stats(ids: List[str], poses_dir: Path, max_seq_len: int | None) -> Tuple[np.ndarray, np.ndarray]:
    total = np.zeros(99, dtype=np.float64)
    total_sq = np.zeros(99, dtype=np.float64)
    count = 0
    for vid in ids:
        arr = np.load(poses_dir / f"{vid}.npy").astype(np.float32)
        if max_seq_len is not None and arr.shape[0] > max_seq_len:
            arr = arr[-max_seq_len :]
        total += arr.sum(axis=0)
        total_sq += (arr**2).sum(axis=0)
        count += arr.shape[0]
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean**2
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class Graph:
    def __init__(self, num_nodes: int = 33):
        self.num_nodes = num_nodes
        self.edges = self._build_mediapipe_edges()
        self.A = self._build_adjacency(self.edges, num_nodes)

    @staticmethod
    def _build_mediapipe_edges():
        # MediaPipe Pose 33-keypoint skeleton connections (undirected)
        return [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            (11, 12),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24),
            (23, 24),
            (23, 25), (25, 27), (27, 29), (29, 31),
            (24, 26), (26, 28), (28, 30), (30, 32),
        ]

    @staticmethod
    def _build_adjacency(edges, num_nodes):
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i, j in edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
        A += np.eye(num_nodes, dtype=np.float32)
        # Normalize
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
        return torch.tensor(A_norm, dtype=torch.float32)


class StGcnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.2):
        super().__init__()
        self.A = A
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T, V]
        x_gcn = torch.einsum("nctv,vw->nctw", x, self.A.to(x.device))
        x_gcn = self.gcn(x_gcn)
        x = self.tcn(x_gcn) + self.residual(x)
        return self.relu(x)


class PitchStGcn(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.graph = Graph()
        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(33 * 3)

        self.st_gcn_layers = nn.ModuleList(
            [
                StGcnBlock(3, 32, A, stride=1, dropout=dropout),
                StGcnBlock(32, 32, A, stride=1, dropout=dropout),
                StGcnBlock(32, 32, A, stride=1, dropout=dropout),
                StGcnBlock(32, 64, A, stride=2, dropout=dropout),
                StGcnBlock(64, 64, A, stride=1, dropout=dropout),
                StGcnBlock(64, 128, A, stride=2, dropout=dropout),
            ]
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B, T, 99] -> [B, T, 33, 3] -> [B, 3, T, 33]
        b, t, _ = x.shape
        x = x.view(b, t, 33, 3).permute(0, 3, 1, 2).contiguous()
        x = x.view(b, 3 * 33, t)
        x = self.data_bn(x)
        x = x.view(b, 3, 33, t).permute(0, 1, 3, 2).contiguous()

        for layer in self.st_gcn_layers:
            x = layer(x)

        # Masked global average pooling over time and joints
        _, c, t_out, v = x.shape
        mask = torch.arange(t_out, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.dtype).view(b, 1, t_out, 1)
        x = x * mask
        denom = lengths.to(x.dtype).clamp(min=1).view(b, 1, 1, 1) * v
        x = x.sum(dim=(2, 3)) / denom.squeeze(-1).squeeze(-1)
        return self.fc(x)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float):
    model.train()
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

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits.float(), y)
        if not torch.isfinite(loss):
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
        preds = logits.argmax(dim=1)
        for cls in range(num_classes):
            mask = y == cls
            total[cls] += mask.sum().item()
            correct[cls] += (preds[mask] == y[mask]).sum().item()
    acc = {int(i): (correct[i] / total[i] if total[i] > 0 else 0.0) for i in range(num_classes)}
    return acc, total


@torch.no_grad()
def eval_confusion_matrix(model, loader, device, num_classes: int):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, lengths, y in loader:
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -5.0, 5.0)
        x = x.to(device=device, dtype=torch.float32)
        lengths = lengths.to(device)
        y = y.to(device)
        logits = model(x, lengths)
        preds = logits.argmax(dim=1)
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1
    return cm


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-GCN for PitchType")
    parser.add_argument("--labels-csv", default="data/pitch_labels.csv")
    parser.add_argument("--poses-dir", default="data/new_poses")
    parser.add_argument("--out-dir", default="baseball-z/outputs/pitch_st_gcn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-seq-len", type=int, default=400)
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
    print(f"Classes ({len(label_encoder.classes_)}): {label_encoder.classes_.tolist()}")

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

    torch.set_default_dtype(torch.float32)

    model = PitchStGcn(num_classes=len(label_encoder.classes_), dropout=cfg.dropout).to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_state = None
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

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
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

    # Evaluate best model for per-class accuracy and confusion matrix
    if best_state is not None:
        model.load_state_dict(best_state)
    per_class_acc, per_class_total = eval_per_class_accuracy(
        model, val_loader, device, num_classes=len(label_encoder.classes_)
    )
    print("Per-class val accuracy:")
    per_class_rows = []
    for idx, name in enumerate(label_encoder.classes_):
        acc = per_class_acc[idx]
        n = int(per_class_total[idx])
        print(f"  {name}: {acc:.3f} (n={n})")
        per_class_rows.append({"class": name, "accuracy": acc, "count": n})
    with open(out_dir / "per_class_val_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(per_class_rows, f, indent=2)

    cm = eval_confusion_matrix(model, val_loader, device, num_classes=len(label_encoder.classes_))
    with open(out_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, indent=2)

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

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Val)")
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45, ha="right")
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    print(f"Best val acc: {best_acc:.3f}")
    print(f"Saved model to {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
