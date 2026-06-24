# -*- coding: utf-8 -*-
"""
MODULE: HUẤN LUYỆN RANDOM FOREST TỪ DỮ LIỆU THỰC TẾ
=====================================================

Đọc file data/train_features.csv (được tạo bởi extract_features_from_videos.py)
và huấn luyện mô hình Random Forest, sau đó lưu ra rf_model.pkl

Cách chạy:
    python src/models/train_rf.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

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


def train_rf():
    print("=" * 60)
    print("  HUAN LUYEN RANDOM FOREST - DROWSINESS DETECTION")
    print("=" * 60)

    # ── 1. Đọc dữ liệu ────────────────────────────────────────────
    print(f"\n[1/4] Doc du lieu tu: {TRAIN_CSV}")
    X_train, y_train = load_data(TRAIN_CSV)
    print(f"       Kich thuoc tap train: {X_train.shape[0]} mau x {X_train.shape[1]} features")

    unique, counts = np.unique(y_train, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"       Lop {lbl} ({CLASS_NAMES[lbl]}): {cnt} mau")

    # ── 2. Khởi tạo mô hình ───────────────────────────────────────
    print("\n[2/4] Khoi tao mo hinh Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,       # Số cây quyết định
        max_depth=3,            # Độ sâu tối đa (Giảm từ 12 -> 3 để chống Overfitting)
        min_samples_split=10,   # Số mẫu tối thiểu để tách nút
        min_samples_leaf=2,     # Số mẫu tối thiểu tại lá
        class_weight="balanced",# Cân bằng dữ liệu không đều
        random_state=42,
        n_jobs=-1,              # Dùng toàn bộ CPU cores
    )
    print(f"       n_estimators={model.n_estimators}, max_depth={model.max_depth}")

    # ── 3. Huấn luyện ─────────────────────────────────────────────
    print("\n[3/4] Dang huan luyen Random Forest...")
    model.fit(X_train, y_train)

    # Đánh giá trên tập train
    y_pred_train = model.predict(X_train)
    acc_train    = accuracy_score(y_train, y_pred_train)
    print(f"       Accuracy tren tap train: {acc_train*100:.2f}%")

    print("\n       Classification Report (Train):")
    print(classification_report(y_train, y_pred_train, target_names=CLASS_NAMES, digits=4))

    # ── Feature importance top 10 ──────────────────────────────────
    print("       Top 10 Features quan trong nhat:")
    feature_names = (
        [f"EAR_f{i}"   for i in range(15)] +
        [f"MAR_f{i}"   for i in range(15)] +
        [f"Pitch_f{i}" for i in range(15)] +
        [f"Yaw_f{i}"   for i in range(15)]
    )
    importances = model.feature_importances_
    top10_idx   = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        print(f"       {rank:2d}. {feature_names[idx]:<12s}: {importances[idx]:.4f}")

    # ── 4. Lưu mô hình ────────────────────────────────────────────
    print(f"\n[4/4] Luu mo hinh tai: {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"       Kich thuoc file: {size_kb:.1f} KB")
    print("=" * 60)
    print("  HOAN THANH! Mo hinh RF da san sang su dung.")
    print("=" * 60)


if __name__ == "__main__":
    train_rf()
