#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from render_pose_videos import render_pose_video


def load_ids(csv_path: Path, id_column: str = "ID"):
    ids = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if id_column not in reader.fieldnames:
            raise ValueError(f"Column '{id_column}' not found in {csv_path}. Columns: {reader.fieldnames}")
        for row in reader:
            value = row.get(id_column, "").strip()
            if value:
                ids.append(value)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Batch render pose overlays for IDs in pitch_labels.csv.")
    parser.add_argument("--csv", type=Path, default=Path("data/pitch_labels.csv"))
    parser.add_argument("--id-column", type=str, default="ID")
    parser.add_argument("--video-dir", type=Path, default=Path("data/videos_clip"))
    parser.add_argument("--pose-dir", type=Path, default=Path("data/new_poses"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/pose_display"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ids = load_ids(args.csv, args.id_column)
    if args.limit:
        ids = ids[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    missing_video = 0
    missing_pose = 0
    skipped = 0

    for vid in ids:
        video_path = args.video_dir / f"{vid}.mp4"
        pose_path = args.pose_dir / f"{vid}.npy"
        out_path = args.out_dir / f"{vid}.mp4"

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        if not video_path.exists():
            missing_video += 1
            continue
        if not pose_path.exists():
            missing_pose += 1
            continue

        rendered = render_pose_video(video_path, pose_path, out_path)
        processed += 1
        print(f"{vid}: {rendered} frames -> {out_path}")

    print(
        "\nSummary:\n"
        f"  processed: {processed}\n"
        f"  skipped existing: {skipped}\n"
        f"  missing video: {missing_video}\n"
        f"  missing pose: {missing_pose}\n"
    )


if __name__ == "__main__":
    main()
