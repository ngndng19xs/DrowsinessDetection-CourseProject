# -*- coding: utf-8 -*-
"""
MODULE: TRÍCH XUẤT ĐẶC TRƯNG TỪ VIDEO THỰC TẾ
===============================================

Mục đích:
    - Đọc các video .mp4 trong data/train/{Normal,Drowsy,Distracted}
      và data/test/{Normal,Drowsy,Distracted}
    - Với mỗi video: lấy 15 frame liên tiếp ở giữa video
    - Trích xuất 4 features (EAR, MAR, Pitch, Yaw) qua 15 frames → vector 60 chiều
    - Lưu ra file CSV: data/train_features.csv và data/test_features.csv

Cấu trúc CSV:
    EAR_f0 ... EAR_f14 | MAR_f0 ... MAR_f14 | Pitch_f0 ... Pitch_f14 | Yaw_f0 ... Yaw_f14 | label
    (60 feature columns + 1 label column)

Label mapping:
    0 → Normal
    1 → Drowsy
    2 → Distracted

Cách chạy:
    python src/models/extract_features_from_videos.py

Lưu ý:
    Tương thích với MediaPipe >= 0.10.x (Tasks API)
    Cần file model: models/face_landmarker.task (tự động tải về nếu chưa có)
"""

import os
import sys
import csv
import urllib.request
import warnings
import numpy as np
import cv2

# Thêm thư mục gốc dự án vào PYTHONPATH để import utils
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Cấu hình encoding cho Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[CANH BAO] MediaPipe chua duoc cai dat. Chay: pip install mediapipe")
    sys.exit(1)

from src.utils.geometry import calculate_ear, calculate_mar
from src.utils.pose import get_head_pose

# ══════════════════════════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════════════════════════
DATA_DIR   = os.path.join(ROOT_DIR, "data")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
TEST_DIR   = os.path.join(DATA_DIR, "test")
TRAIN_CSV  = os.path.join(DATA_DIR, "train_features.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_features.csv")

# Đường dẫn file model MediaPipe FaceLandmarker
MODEL_DIR  = os.path.join(ROOT_DIR, "src", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

NUM_FRAMES = 15          # Số frame liên tiếp dùng làm 1 mẫu

LABEL_MAP = {
    "Normal":     0,
    "Drowsy":     1,
    "Distracted": 2,
}

# Các index MediaPipe FaceMesh cho EAR trái, EAR phải, MAR
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX     = [61, 40, 37, 0, 267, 270, 291, 321]


# ══════════════════════════════════════════════════════════════════
#  TẢI MODEL NẾU CHƯA CÓ
# ══════════════════════════════════════════════════════════════════
def ensure_model():
    """Tải file model face_landmarker.task nếu chưa tồn tại."""
    if os.path.exists(MODEL_PATH):
        print(f"  [OK] Model da ton tai: {MODEL_PATH}")
        return True

    print(f"  [TAI] Dang tai model tu: {MODEL_URL}")
    print(f"        Luu vao: {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb  = downloaded / 1024 / 1024
                print(f"\r        {pct:3d}% ({mb:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, show_progress)
        print(f"\n  [OK] Tai model thanh cong!")
        return True
    except Exception as e:
        print(f"\n  [LOI] Khong the tai model: {e}")
        print("        Hay tai thu cong tu:")
        print(f"        {MODEL_URL}")
        print(f"        Va luu vao: {MODEL_PATH}")
        return False


# ══════════════════════════════════════════════════════════════════
#  TẠO FACE LANDMARKER (Tasks API cho MediaPipe >= 0.10)
# ══════════════════════════════════════════════════════════════════
def create_face_landmarker():
    """Khởi tạo FaceLandmarker dùng MediaPipe Tasks API."""
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ══════════════════════════════════════════════════════════════════
#  TRÍCH XUẤT FEATURES TỪ MỘT VIDEO
# ══════════════════════════════════════════════════════════════════
def extract_features_from_video(video_path: str, landmarker) -> list[float] | None:
    """
    Trích xuất 60 features từ một file video.

    Chiến lược lấy frame:
        - Bỏ 10% đầu và 10% cuối (thường là cảnh chuyển tiếp)
        - Lấy NUM_FRAMES frame cách đều nhau từ phần giữa

    Trả về:
        list 60 float: [EAR×15, MAR×15, Pitch×15, Yaw×15]
        None nếu không phát hiện đủ khuôn mặt
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < NUM_FRAMES:
        cap.release()
        return None

    # Xác định vùng lấy frame (loại bỏ 10% đầu - cuối)
    start_f = max(0, int(total_frames * 0.10))
    end_f   = min(total_frames - 1, int(total_frames * 0.90))
    usable  = end_f - start_f

    if usable < NUM_FRAMES:
        # Fallback: lấy toàn bộ
        start_f, end_f, usable = 0, total_frames - 1, total_frames

    # Tính vị trí NUM_FRAMES frame cách đều nhau
    frame_indices = np.linspace(start_f, end_f, NUM_FRAMES, dtype=int)

    ears, mars, pitches, yaws = [], [], [], []

    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tạo MediaPipe Image từ numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Phát hiện landmarks
        result = landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            continue

        lms = result.face_landmarks[0]  # list of NormalizedLandmark

        # EAR (trung bình 2 mắt)
        left_eye  = [lms[i] for i in LEFT_EYE_IDX]
        right_eye = [lms[i] for i in RIGHT_EYE_IDX]
        ear_l = calculate_ear(left_eye,  w, h)
        ear_r = calculate_ear(right_eye, w, h)
        ear   = (ear_l + ear_r) / 2.0

        # MAR
        mouth = [lms[i] for i in MOUTH_IDX]
        mar   = calculate_mar(mouth, w, h)

        # Head Pose
        pitch, yaw, _ = get_head_pose(lms, w, h)

        ears.append(ear)
        mars.append(mar)
        pitches.append(pitch)
        yaws.append(yaw)

    cap.release()

    # Kiểm tra đủ 15 frames có khuôn mặt
    if len(ears) < NUM_FRAMES:
        # Padding bằng giá trị trung bình nếu thiếu (do mất face detection)
        if len(ears) == 0:
            return None
        while len(ears) < NUM_FRAMES:
            ears.append(np.mean(ears))
            mars.append(np.mean(mars))
            pitches.append(np.mean(pitches))
            yaws.append(np.mean(yaws))

    # Ghép thành vector 60
    feature_vector = ears[:NUM_FRAMES] + mars[:NUM_FRAMES] + pitches[:NUM_FRAMES] + yaws[:NUM_FRAMES]
    return feature_vector


# ══════════════════════════════════════════════════════════════════
#  XỬ LÝ TOÀN BỘ MỘT TẬP (TRAIN HOẶC TEST)
# ══════════════════════════════════════════════════════════════════
def process_split(split_dir: str, output_csv: str, split_name: str) -> int:
    """
    Xử lý toàn bộ một tập (train hoặc test):
        - Duyệt qua 3 thư mục con: Normal, Drowsy, Distracted
        - Trích xuất features từng video
        - Lưu ra CSV

    Trả về số mẫu thành công.
    """
    print(f"\n{'='*60}")
    print(f"  XU LY TAP: {split_name.upper()}  ({split_dir})")
    print(f"{'='*60}")

    rows      = []
    total_ok  = 0
    total_err = 0

    # Tạo header
    feature_names = (
        [f"EAR_f{i}"   for i in range(NUM_FRAMES)] +
        [f"MAR_f{i}"   for i in range(NUM_FRAMES)] +
        [f"Pitch_f{i}" for i in range(NUM_FRAMES)] +
        [f"Yaw_f{i}"   for i in range(NUM_FRAMES)] +
        ["label"]
    )

    # Tạo landmarker một lần dùng cho cả tập
    landmarker = create_face_landmarker()

    try:
        for class_name, label_id in LABEL_MAP.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"  [!!] Khong tim thay thu muc: {class_dir}")
                continue

            video_files = sorted([
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov"))
            ])
            print(f"\n  [{class_name}] (label={label_id}) - {len(video_files)} video")

            for i, vfile in enumerate(video_files, 1):
                vpath = os.path.join(class_dir, vfile)
                feats = extract_features_from_video(vpath, landmarker)

                if feats is not None:
                    rows.append(feats + [label_id])
                    total_ok += 1
                    if i % 20 == 0 or i == len(video_files):
                        print(f"    [{i}/{len(video_files)}] OK: {total_ok} mau")
                else:
                    total_err += 1
                    if total_err <= 5:
                        print(f"    [SKIP] Khong trich xuat duoc: {vfile}")
    finally:
        landmarker.close()

    # Ghi CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        writer.writerows(rows)

    print(f"\n  >> Luu CSV: {output_csv}")
    print(f"  >> Tong mau thanh cong: {total_ok} | Bo qua: {total_err}")
    return total_ok


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  TRICH XUAT DAC TRUNG TU VIDEO THUC TE")
    print("  (EAR, MAR, Pitch, Yaw) x 15 frames -> 60 features")
    print("  Su dung MediaPipe Tasks API (>= 0.10.x)")
    print("=" * 60)
    print(f"  Thu muc du lieu: {DATA_DIR}")
    print(f"  So frame / mau : {NUM_FRAMES}")

    # Đảm bảo có model file
    if not ensure_model():
        sys.exit(1)

    # Xử lý tập Train
    n_train = process_split(TRAIN_DIR, TRAIN_CSV, "train")

    # Xử lý tập Test
    n_test = process_split(TEST_DIR, TEST_CSV, "test")

    print(f"\n{'='*60}")
    print(f"  HOAN THANH TRICH XUAT DAC TRUNG")
    print(f"  Train: {n_train} mau  ->  {TRAIN_CSV}")
    print(f"  Test : {n_test} mau   ->  {TEST_CSV}")
    print(f"{'='*60}")
    print("\nBuoc tiep theo:")
    print("  python src/models/train_rf.py")
    print("  python src/models/train_svm.py")
    print("  python src/models/train_xgboost.py")


if __name__ == "__main__":
    main()
