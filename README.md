# Baseball Pitch-Type Classification (Pose-Based)

## 1. What Can This Repo Do For You
This repo provides an end-to-end pipeline to build pose-based pitch-type classifiers from baseball video:
- Crawl and download MLB video clips.
- Extract pitcher pose keypoints from video using YOLO + MediaPipe.
- Render pose overlays for visual QA.
- Train multiple sequence models (LSTM, CNN+LSTM+Attention, ST-GCN).
- Run grid search to compare hyperparameters.

It’s set up so you can go from raw video links to trained models and reproducible results.

## 2. My Research Results
Below are the validation accuracies recorded in `outputs/*/metadata.json` and the grid search summary in `outputs/grid_search_cnn_lstm_atten/grid_results.csv`.

**Single-run model comparisons (best validation accuracy):**
| Model | Best Val Acc | Notes |
| --- | --- | --- |
| LSTM (`pitch_lstm.py`) | **～0.40** | Baseline sequence model |
| CNN+LSTM+Attention (`pitch_cnn_lstm_atten.py`) | **～0.64** | Stronger temporal + attention model |
| ST-GCN (`pitch_st_gcn.py`) | **～0.40** | Spatial-temporal graph model |

**Grid search (CNN+LSTM+Attention):**
Best configuration in the current grid:
- `max_seq_len=330`, `hidden_size=192`, `lr=0.0003`, `dropout=0.3`, `epochs=60`
- Best validation accuracy: **0.6409**

Top 5 grid runs (by `best_val_acc`):
| run_name | max_seq_len | hidden_size | lr | dropout | epochs | best_val_acc |
| --- | --- | --- | --- | --- | --- | --- |
| len330_hid192_uni_lr0.0003_drop0.3 | 330 | 192 | 0.0003 | 0.3 | 60 | 0.6409 |
| len300_hid192_uni_lr0.0003_drop0.3 | 300 | 192 | 0.0003 | 0.3 | 60 | 0.6332 |
| len240_hid192_uni_lr0.0003_drop0.2 | 240 | 192 | 0.0003 | 0.2 | 60 | 0.6255 |
| len300_hid192_uni_lr0.0003_drop0.2 | 300 | 192 | 0.0003 | 0.2 | 60 | 0.6178 |
| len330_hid192_uni_lr0.0003_drop0.2 | 330 | 192 | 0.0003 | 0.2 | 60 | 0.6178 |

## 3. Prepare Your Own Data
This project expects three core data artifacts:
- `data/statcast_data.csv`: source table of video links (used by `crawler.py`).
- `data/videos_clip/`: short clips per pitch (mp4).
- `data/new_poses/`: pose keypoints per clip (npy).

If you are preparing your own dataset:
1. Collect high-quality videos with a clear, unobstructed pitcher view. Consistent camera angle helps.
2. Create a CSV with a `VideoLink` column (or update `crawler.py` to match your schema).
3. Label each clip in `data/pitch_labels.csv` with an `ID` that matches the video filename.

Higher-quality and more diverse clips usually lead to better accuracy.

## 4. Quick Start
### 4.1 Environment setup
```bash
pyenv activate <your-env-name>
pip install -r requirements.txt
```

### 4.2 Crawl and download video clips
Uses `data/statcast_data.csv` to pull video links and save 5s clips.
```bash
python crawler.py
```
Output folders:
- `data/videos_tiny/` (raw downloads)
- `data/videos_clip/` (clipped 5s videos)

### 4.3 Extract poses (keypoints) from video
`keypoint_extraction_model.py` provides the pose extraction function used across the pipeline. You can call it from a script or notebook to generate `.npy` pose sequences.

Typical output folder:
- `data/new_poses/`

Minimal example:
```bash
python - <<'PY'
from keypoint_extraction_model import extract_pose_sequence
from pathlib import Path

video = Path("data/videos_clip/<video_id>.mp4")
out = Path("data/new_poses/<video_id>.npy")
pose = extract_pose_sequence(video, output_npy=out)
print("saved:", out)
PY
```

### 4.4 Render pose overlays (QA)
Single video + pose:
```bash
python render_pose_videos.py --video <path/to/video.mp4> --pose <path/to/pose.npy> --out <path/to/out.mp4>
```
Batch render by pitch labels:
```bash
python batch_render_pose_videos.py --csv data/pitch_labels.csv --video-dir data/videos_clip --pose-dir data/new_poses --out-dir data/pose_display
```

### 4.5 Train models
LSTM:
```bash
python pitch_lstm.py --labels-csv data/pitch_labels.csv --poses-dir data/new_poses --out-dir outputs/pitch_lstm
```
CNN+LSTM+Attention:
```bash
python pitch_cnn_lstm_atten.py --labels-csv data/pitch_labels.csv --poses-dir data/new_poses --out-dir outputs/pitch_cnn_lstm_atten
```
ST-GCN:
```bash
python pitch_st_gcn.py --labels-csv data/pitch_labels.csv --poses-dir data/new_poses --out-dir outputs/pitch_st_gcn
```
Note: the scripts default to `baseball-z/outputs/...`, so explicitly set `--out-dir` if you want outputs to live under this repo’s `outputs/` folder.

### 4.6 Grid search
```bash
python grid_search_cnn_lstm_atten.py --out-dir outputs/grid_search_cnn_lstm_atten
```
Note: the grid search script invokes `baseball-z/pitch_cnn_lstm_atten.py`, so run it from the repo root.
