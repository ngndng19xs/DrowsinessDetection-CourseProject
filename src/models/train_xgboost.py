# -*- coding: utf-8 -*-
"""
MODULE: HUẤN LUYỆN XGBOOST TỪ DỮ LIỆU THỰC TẾ
================================================

Đọc file data/train_features.csv (được tạo bởi extract_features_from_videos.py)
và huấn luyện mô hình XGBoost, sau đó lưu ra xgboost_model.pkl

Cách chạy:
    python src/models/train_xgboost.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Cấu hình encoding Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from xgboost import XGBClassifier
except ImportError:
    print("[LOI] XGBoost chua duoc cai dat. Chay: pip install xgboost")
    sys.exit(1)

# ── Đường dẫn ────────────────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_CSV  = os.path.join(ROOT_DIR, "data", "train_features.csv")
MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

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


def train_xgboost():
    print("=" * 60)
    print("  HUAN LUYEN XGBOOST - DROWSINESS DETECTION")
    print("=" * 60)

    # ── 1. Đọc dữ liệu ────────────────────────────────────────────
    print(f"\n[1/5] Doc du lieu tu: {TRAIN_CSV}")
    X_all, y_all = load_data(TRAIN_CSV)
    print(f"       Kich thuoc tap train: {X_all.shape[0]} mau x {X_all.shape[1]} features")

    unique, counts = np.unique(y_all, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"       Lop {lbl} ({CLASS_NAMES[lbl]}): {cnt} mau")

    # ── 2. Tách train / validation nội bộ (để Early Stopping) ─────
    print("\n[2/5] Tach tap validation noi bo (10%) de Early Stopping...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.10, random_state=42,
        stratify=y_all if len(np.unique(y_all)) > 1 else None
    )
    print(f"       Train: {X_train.shape[0]} mau | Validation: {X_val.shape[0]} mau")

    # ── 3. Khởi tạo mô hình ───────────────────────────────────────
    print("\n[3/5] Khoi tao mo hinh XGBoost (da lop - softmax)...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=2,          # Giảm từ 6 -> 2
        learning_rate=0.01,   # Giảm từ 0.1 -> 0.01
        reg_lambda=1,         # Thêm L2 regularization
        subsample=0.7,
        colsample_bytree=0.7,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    print(f"       n_estimators={xgb_model.n_estimators}, max_depth={xgb_model.max_depth}, lr={xgb_model.learning_rate}")

    # ── 4. Huấn luyện ─────────────────────────────────────────────
    print("\n[4/5] Dang huan luyen XGBoost...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Đánh giá trên toàn tập train
    y_pred_train = xgb_model.predict(X_all)
    acc_train    = accuracy_score(y_all, y_pred_train)
    print(f"       Accuracy tren toan tap train: {acc_train*100:.2f}%")

    print("\n       Classification Report (Train):")
    print(classification_report(y_all, y_pred_train, target_names=CLASS_NAMES, digits=4))

    # ── Feature importance top 10 ──────────────────────────────────
    print("       Top 10 Features quan trong nhat:")
    feature_names = (
        [f"EAR_f{i}"   for i in range(15)] +
        [f"MAR_f{i}"   for i in range(15)] +
        [f"Pitch_f{i}" for i in range(15)] +
        [f"Yaw_f{i}"   for i in range(15)]
    )
    importances = xgb_model.feature_importances_
    top10_idx   = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        print(f"       {rank:2d}. {feature_names[idx]:<12s}: {importances[idx]:.4f}")

    # ── 5. Lưu mô hình ────────────────────────────────────────────
    print(f"\n[5/5] Luu mo hinh XGBoost tai: {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"       Kich thuoc file: {size_kb:.1f} KB")
    print("=" * 60)
    print("  HOAN THANH! Mo hinh XGBoost da san sang su dung.")
    print("=" * 60)


if __name__ == "__main__":
    train_xgboost()
