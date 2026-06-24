# -*- coding: utf-8 -*-
"""
MODULE: ĐÁNH GIÁ TỪNG GIAI ĐOẠN (PIPELINE) CHO MÔ HÌNH SVM
==========================================================

Mục đích:
    Tập lệnh này thực hiện đánh giá chi tiết hệ thống nhận diện buồn ngủ 
    theo 3 giai đoạn chính:
    1. Giai đoạn 1: Trích xuất đặc trưng (Tỉ lệ nhận diện khuôn mặt & mắt)
    2. Giai đoạn 2: Phân loại trạng thái (Sử dụng mô hình SVM)
    3. Giai đoạn 3: Hiệu năng tổng thể (Tốc độ xử lý FPS và Latency)

Cách chạy:
    python src/models/evaluate_svm_pipeline.py
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# Đảm bảo import được các module trong src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# Đường dẫn
MODELS_DIR = os.path.join(ROOT_DIR, "src", "models")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
TEST_CSV = os.path.join(ROOT_DIR, "data", "test_features.csv")
TEST_VIDEO_DIR = os.path.join(ROOT_DIR, "data", "test")
MODEL_TASK_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

CLASS_NAMES = ["Normal", "Drowsy", "Distracted"]

warnings.filterwarnings("ignore")


def evaluate_stage_1_feature_extraction(test_dir: str, sample_limit: int = 50):
    """
    Giai đoạn 1: Đánh giá tỉ lệ phát hiện khuôn mặt và trích xuất điểm chuẩn.
    Thử nghiệm trên một số lượng video mẫu trong tập test.
    """
    print("\n" + "="*60)
    print(" GIAI ĐOẠN 1: ĐÁNH GIÁ TRÍCH XUẤT ĐẶC TRƯNG (MediaPipe)")
    print("="*60)
    
    if not MP_AVAILABLE or not os.path.exists(MODEL_TASK_PATH):
        print("  [BỎ QUA] Không tìm thấy MediaPipe hoặc file face_landmarker.task.")
        return

    if not os.path.exists(test_dir):
        print(f"  [BỎ QUA] Không tìm thấy thư mục video test: {test_dir}")
        return

    # Khởi tạo landmarker
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    total_frames = 0
    detected_frames = 0
    video_count = 0

    print(f"  Đang quét tối đa {sample_limit} video từ: {test_dir}")
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        videos = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi'))]
        for v in videos:
            if video_count >= sample_limit:
                break
                
            vpath = os.path.join(class_dir, v)
            cap = cv2.VideoCapture(vpath)
            
            # Chỉ lấy tối đa 30 frame mỗi video để đánh giá nhanh
            frames_to_check = 30
            f_count = 0
            while cap.isOpened() and f_count < frames_to_check:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                result = landmarker.detect(mp_image)
                if result.face_landmarks and len(result.face_landmarks) > 0:
                    detected_frames += 1
                
                f_count += 1
                
            cap.release()
            video_count += 1

    landmarker.close()

    if total_frames > 0:
        detection_rate = (detected_frames / total_frames) * 100
        print(f"  - Số lượng video đã test    : {video_count}")
        print(f"  - Tổng số frames đã test    : {total_frames}")
        print(f"  - Số frames tìm thấy mặt    : {detected_frames}")
        print(f"  -> TỈ LỆ NHẬN DIỆN THÀNH CÔNG: {detection_rate:.2f}%")
    else:
        print("  - Không đọc được frame nào từ tập test.")


def evaluate_stage_2_svm_classification(csv_path: str, model_path: str):
    """
    Giai đoạn 2: Đánh giá mô hình SVM dựa trên dữ liệu features đã trích xuất.
    """
    print("\n" + "="*60)
    print(" GIAI ĐOẠN 2: ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI SVM")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print(f"  [LỖI] Không tìm thấy file CSV: {csv_path}")
        return None, None, None

    if not os.path.exists(model_path):
        print(f"  [LỖI] Không tìm thấy file mô hình SVM: {model_path}")
        return None, None, None

    # Load dữ liệu
    df = pd.read_csv(csv_path)
    X_test = df.drop(columns=["label"]).values.astype(np.float32)
    y_test = df["label"].values.astype(int)

    # Load model
    with open(model_path, "rb") as f:
        svm_model = pickle.load(f)

    # Dự đoán
    t0 = time.perf_counter()
    y_pred = svm_model.predict(X_test)
    inference_time = time.perf_counter() - t0

    # Tính toán chỉ số
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Kết quả phân loại trên {len(X_test)} mẫu test:")
    print(f"  - Accuracy (Độ chính xác) : {acc * 100:.2f}%")
    print(f"  - Precision (Độ chuẩn xác): {prec * 100:.2f}%")
    print(f"  - Recall (Độ bao phủ)     : {rec * 100:.2f}%")
    print(f"  - F1-Score                : {f1 * 100:.2f}%")
    
    print("\n  Ma trận nhầm lẫn (Confusion Matrix):")
    print("  " + "-"*35)
    print("               |  Dự đoán")
    print("  Thực tế      |  Nor   Dro   Dis")
    print("  " + "-"*35)
    for i, cls in enumerate(["Nor", "Dro", "Dis"]):
        row = cm[i]
        print(f"  {cls:<12} | {row[0]:>4}  {row[1]:>4}  {row[2]:>4}")
    
    print("\n  Chi tiết từng lớp (Classification Report):")
    print("  " + "-"*55)
    print(f"  {'Trạng thái':<15} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("  " + "-"*55)
    
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:<15} | {prec_per_class[i]*100:>8.2f}% | {rec_per_class[i]*100:>8.2f}% | {f1_per_class[i]*100:>8.2f}%")
    print("  " + "-"*55)

    return svm_model, X_test, inference_time


def evaluate_stage_3_end_to_end_performance(svm_model, X_test, inference_time_total):
    """
    Giai đoạn 3: Đánh giá hiệu năng và độ trễ của hệ thống (Latency / FPS)
    """
    print("\n" + "="*60)
    print(" GIAI ĐOẠN 3: ĐÁNH GIÁ HIỆU NĂNG TỔNG THỂ")
    print("="*60)

    if svm_model is None or X_test is None:
        print("  [BỎ QUA] Không có dữ liệu từ Giai đoạn 2 để tính toán.")
        return

    n_samples = len(X_test)
    
    # Thời gian xử lý của mô hình SVM (milli-giây trên 1 mẫu)
    svm_latency_ms = (inference_time_total / n_samples) * 1000
    
    # Ước lượng tổng thời gian xử lý 1 frame (Face Detection + SVM)
    # MediaPipe FaceLandmarker thường mất khoảng 15-25ms trên CPU hiện đại
    estimated_mediapipe_ms = 20.0 
    
    # Lưu ý: Mô hình của ta dùng 15 frames làm 1 mẫu (sequence), 
    # nhưng inference SVM thì tính trên toàn bộ mảng đã trích xuất.
    total_pipeline_latency_ms = estimated_mediapipe_ms + svm_latency_ms
    estimated_fps = 1000.0 / total_pipeline_latency_ms if total_pipeline_latency_ms > 0 else 0

    print(f"  - Thời gian trích xuất đặc trưng (ước tính): ~{estimated_mediapipe_ms:.2f} ms/frame")
    print(f"  - Thời gian dự đoán mô hình SVM            :  {svm_latency_ms:.4f} ms/chuỗi(15 frames)")
    print(f"  - Tổng độ trễ Pipeline (Latency)           : ~{total_pipeline_latency_ms:.2f} ms")
    print(f"  - Tốc độ khung hình xử lý (FPS) lý thuyết  : ~{estimated_fps:.0f} FPS")
    
    print("\n  KẾT LUẬN HIỆU NĂNG:")
    if estimated_fps >= 15:
        print("  -> Hệ thống ĐẠT chuẩn Real-time (> 15 FPS), phù hợp để cảnh báo trực tiếp.")
    else:
        print("  -> Hệ thống có thể bị trễ (lag), cần tối ưu lại cấu hình phần cứng hoặc mô hình.")
    print("="*60 + "\n")


def main():
    print("\n BẮT ĐẦU ĐÁNH GIÁ PIPELINE CHO MÔ HÌNH SVM ".center(60, "*"))
    
    # Chạy Giai đoạn 1 (Tuỳ chọn: Nếu máy chạy lâu có thể comment lại)
    evaluate_stage_1_feature_extraction(TEST_VIDEO_DIR, sample_limit=147)
    
    # Chạy Giai đoạn 2
    svm_model, X_test, inference_time = evaluate_stage_2_svm_classification(TEST_CSV, SVM_MODEL_PATH)
    
    # Chạy Giai đoạn 3
    evaluate_stage_3_end_to_end_performance(svm_model, X_test, inference_time)


if __name__ == "__main__":
    main()
