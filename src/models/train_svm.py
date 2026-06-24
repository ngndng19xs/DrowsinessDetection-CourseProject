# -*- coding: utf-8 -*-
"""
MODULE: HUẤN LUYỆN SVM TỪ DỮ LIỆU THỰC TẾ
============================================

Đọc file data/train_features.csv (được tạo bởi extract_features_from_videos.py)
và huấn luyện Pipeline (StandardScaler + SVM), sau đó lưu ra svm_model.pkl

Cách chạy:
    python src/models/train_svm.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Cấu hình encoding Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Đường dẫn ────────────────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_CSV  = os.path.join(ROOT_DIR, "data", "train_features.csv")
MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

CLASS_NAMES = ["Normal", "Drowsy", "Distracted"]


def load_data(csv_path: str) -> tuple:
    """Đọc CSV và tách X (features) và y (label)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Khong tim thay file CSV: {csv_path}\n"
            "Hay chay extract_features_from_videos.py truoc!"
        )
    df = pd.read_csv(csv_path)
    X  = df.drop(columns=["label"]).values.astype(np.float32)
    y  = df["label"].values.astype(int)
    return X, y


def train_svm():
    print("=" * 60)
    print("  HUAN LUYEN SVM - DROWSINESS DETECTION")
    print("=" * 60)

    # ── 1. Đọc dữ liệu ────────────────────────────────────────────
    print(f"\n[1/4] Doc du lieu tu: {TRAIN_CSV}")
    X_train, y_train = load_data(TRAIN_CSV)
    print(f"       Kich thuoc tap train: {X_train.shape[0]} mau x {X_train.shape[1]} features")

    unique, counts = np.unique(y_train, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"       Lop {lbl} ({CLASS_NAMES[lbl]}): {cnt} mau")

    # ── 2. Xây dựng Pipeline ──────────────────────────────────────
    print("\n[2/4] Xay dung Pipeline (StandardScaler → SVC kernel Linear)...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="linear",
            C=0.1,
            probability=True,
            random_state=42,
            class_weight="balanced",    # Cân bằng class không đều
        )),
    ])
    print(f"       SVC: kernel=linear, C=0.1, class_weight=balanced")

    # ── 3. Huấn luyện ─────────────────────────────────────────────
    print("\n[3/4] Dang huan luyen SVM (co the mat vai phut)...")
    svm_pipeline.fit(X_train, y_train)

    # Đánh giá trên tập train
    y_pred_train = svm_pipeline.predict(X_train)
    acc_train    = accuracy_score(y_train, y_pred_train)
    print(f"       Accuracy tren tap train: {acc_train*100:.2f}%")

    print("\n       Classification Report (Train):")
    print(classification_report(y_train, y_pred_train, target_names=CLASS_NAMES, digits=4))

    # ── 4. Lưu mô hình ────────────────────────────────────────────
    print(f"\n[4/4] Luu Pipeline (Scaler + SVM) tai: {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(svm_pipeline, f)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"       Kich thuoc file: {size_kb:.1f} KB")
    print("=" * 60)
    print("  HOAN THANH! Pipeline SVM da san sang su dung.")
    print("=" * 60)


if __name__ == "__main__":
    train_svm()
