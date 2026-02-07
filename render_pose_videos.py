#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# MediaPipe Pose keypoint connections (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (17, 19), (18, 20),
]


def load_pose(pose_path: Path) -> np.ndarray:
    pose = np.load(pose_path)
    if pose.ndim != 2 or pose.shape[1] != 99:
        raise ValueError(f"Unexpected pose shape {pose.shape} in {pose_path}")
    return pose.reshape(-1, 33, 3)


def draw_pose(frame, pose_frame, point_color=(0, 255, 0), line_color=(255, 0, 0)):
    h, w = frame.shape[:2]

    # Draw connections
    for a, b in POSE_CONNECTIONS:
        xa, ya = pose_frame[a][:2]
        xb, yb = pose_frame[b][:2]
        xa = int(xa * w)
        ya = int(ya * h)
        xb = int(xb * w)
        yb = int(yb * h)
        if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
            cv2.line(frame, (xa, ya), (xb, yb), line_color, 2)

    # Draw points
    for x, y, _ in pose_frame:
        cx = int(x * w)
        cy = int(y * h)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(frame, (cx, cy), 4, point_color, -1)

    return frame


def render_pose_video(video_path: Path, pose_path: Path, out_path: Path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = load_pose(pose_path)
    pose_frames = len(pose)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = pose_frames

    if max_frames is None:
        max_frames = min(total_frames, pose_frames)
    else:
        max_frames = min(max_frames, total_frames, pose_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_pose(frame, pose[frame_idx])
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return frame_idx


def collect_pairs(video_dir: Path, pose_dir: Path):
    pairs = []
    for pose_path in sorted(pose_dir.glob("*.npy")):
        stem = pose_path.stem
        video_path = video_dir / f"{stem}.mp4"
        if video_path.exists():
            pairs.append((video_path, pose_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Render pose keypoints onto videos.")
    parser.add_argument("--video", type=Path, help="Single input video (.mp4)")
    parser.add_argument("--pose", type=Path, help="Single pose file (.npy)")
    parser.add_argument("--out", type=Path, help="Output video path (.mp4)")
    parser.add_argument("--video-dir", type=Path, default=Path("data/videos_clip"))
    parser.add_argument("--pose-dir", type=Path, default=Path("data/new_poses"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/pose_display"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")

    args = parser.parse_args()

    if args.video and args.pose:
        if not args.out:
            args.out = args.out_dir / f"{args.pose.stem}.mp4"
        rendered = render_pose_video(args.video, args.pose, args.out)
        print(f"Rendered {rendered} frames to {args.out}")
        return

    pairs = collect_pairs(args.video_dir, args.pose_dir)
    if args.limit:
        pairs = pairs[: args.limit]

    if not pairs:
        raise SystemExit("No matching video/pose pairs found.")

    for video_path, pose_path in pairs:
        out_path = args.out_dir / f"{pose_path.stem}.mp4"
        rendered = render_pose_video(video_path, pose_path, out_path)
        print(f"{pose_path.stem}: {rendered} frames -> {out_path}")


if __name__ == "__main__":
    main()
