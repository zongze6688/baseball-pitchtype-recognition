import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from ultralytics import YOLO
from tqdm import tqdm
import importlib

HAS_MP_SOLUTIONS = hasattr(mp, "solutions")

# initialize mediapipe
if HAS_MP_SOLUTIONS:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
else:
    mp_pose = None
    mp_drawing = None
    mp_drawing_styles = None
    _mp_tasks = importlib.import_module("mediapipe.tasks.python")
    _mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
    _mp_image = importlib.import_module("mediapipe.tasks.python.vision.core.image")
    PoseLandmarker = _mp_vision.PoseLandmarker
    PoseLandmarkerOptions = _mp_vision.PoseLandmarkerOptions
    RunningMode = _mp_vision.RunningMode
    BaseOptions = _mp_tasks.BaseOptions
    Image = _mp_image.Image
    ImageFormat = _mp_image.ImageFormat

yolo_model = YOLO("./models/yolov8n.pt")
PERSON_CLASS_ID = 0  # COCO 数据集中 person 的类别 ID

def extract_pose_sequence(input_path, output_npy=None):
    """
    extract keypoint information from the video
    Args:
        input_path: input video path
        output_npy: save as npy file
    Returns:
        pose_sequence: np.array, shape=(seq_len, 99) --- 33 keypoints * (x, y, z)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("cannot open video")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_sequence = []

    if HAS_MP_SOLUTIONS:
        # Legacy Solutions API (Python <= 3.11)
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, 
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO
                results = yolo_model(frame, verbose=False)
                detections = results[0].boxes.data.cpu().numpy()

                person_boxes = [d for d in detections if int(d[5]) == PERSON_CLASS_ID]
                if not person_boxes:
                    print("no person detected, skip this frame")
                    continue

                selected_box = min(person_boxes, key=lambda box: abs((box[3] + box[1]) / 2 - height))

                x1, y1, x2, y2, conf, cls = selected_box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                person_img = frame[y1:y2, x1:x2]

                image_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                result = pose.process(image_rgb)

                if result.pose_landmarks:
                    landmarks = []
                    for lm in result.pose_landmarks.landmark:

                        orig_x = x1 + lm.x * (x2 - x1)
                        orig_y = y1 + lm.y * (y2 - y1)

                        landmarks.extend([
                            round(orig_x / width, 6),
                            round(orig_y / height, 6),
                            round(lm.z, 6)
                        ])

                    pose_sequence.append(landmarks)
    else:
        # MediaPipe Tasks API (Python 3.12+)
        model_path = os.environ.get(
            "MP_POSE_MODEL",
            './models/pose_landmarker_lite.task',
            #os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "MediaPipe Tasks model not found. "
                "Set MP_POSE_MODEL to a .task file (e.g., pose_landmarker_lite.task) "
                f"or place it at {model_path}."
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = yolo_model(frame, verbose=False)
                detections = results[0].boxes.data.cpu().numpy()

                person_boxes = [d for d in detections if int(d[5]) == PERSON_CLASS_ID]
                if not person_boxes:
                    print("no person detected, skip this frame")
                    frame_idx += 1
                    continue

                selected_box = min(person_boxes, key=lambda box: abs((box[3] + box[1]) / 2 - height))

                x1, y1, x2, y2, conf, cls = selected_box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                person_img = frame[y1:y2, x1:x2]

                image_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                if timestamp_ms == 0:
                    timestamp_ms = int(frame_idx * (1000.0 / fps))

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    landmarks = []
                    for lm in result.pose_landmarks[0]:
                        orig_x = x1 + lm.x * (x2 - x1)
                        orig_y = y1 + lm.y * (y2 - y1)

                        landmarks.extend([
                            round(orig_x / width, 6),
                            round(orig_y / height, 6),
                            round(lm.z, 6)
                        ])

                    pose_sequence.append(landmarks)

                frame_idx += 1

    cap.release()
    print(f"extracted {len(pose_sequence)} frames of keypoint info")

    pose_sequence = np.array(pose_sequence, dtype=np.float32)  # shape: (seq_len, 99)

    if output_npy:
        np.save(output_npy, pose_sequence)
        print(f"saved as : {output_npy}")

    return pose_sequence

if __name__ == "__main__":
    df = pd.read_csv("./data/pitch_labels.csv")
    os.makedirs("./data/new_poses", exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        play_id = row["ID"]
        video_path = f"./data/videos_clip/{play_id}.mp4"
        output_path = f"./data/new_poses/{play_id}.npy"
        print(f'extracting pose for {video_path}')
        if os.path.exists(video_path):
            extract_pose_sequence(video_path, output_path)
        else:
            print(f'video {video_path} does not exist!')

  