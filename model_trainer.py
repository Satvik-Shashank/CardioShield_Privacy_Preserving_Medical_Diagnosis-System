"""
model_trainer.py
────────────────
Trains the Logistic Regression baseline model on the UCI Heart Disease dataset,
evaluates performance using Stratified K-Fold cross-validation and holdout metrics,
benchmarks genuine TenSEAL CKKS homomorphic inference against plaintext predictions,
and exports all artifacts including `metrics.json`.
"""

import json
import os
import pickle
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

COLUMNS = FEATURE_COLS + ["target"]


def load_uci_data(csv_path: str = "heart.csv") -> pd.DataFrame:
    """Load UCI Cleveland Heart Disease dataset from local file, remote URL, or synthetic fallback."""
    if os.path.exists(csv_path):
        print(f"[data] Loading from local file: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        if "target" not in df.columns and "num" in df.columns:
            df.rename(columns={"num": "target"}, inplace=True)
        return df

    try:
        print(f"[data] Fetching dataset from UCI Repository: {UCI_URL}")
        df = pd.read_csv(UCI_URL, names=COLUMNS, na_values="?")
        print(f"[data] Successfully downloaded {len(df)} records from UCI.")
        return df
    except Exception as e:
        print(f"[data] Network fetch failed ({e}) — generating standard synthetic dataset.")
        return _make_synthetic_data()


def _make_synthetic_data(n: int = 303, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos

    def sample(mean, std, low, high, size):
        return np.clip(rng.normal(mean, std, size), low, high)

    pos = {
        "age": sample(57, 8, 30, 80, n_pos),
        "sex": rng.choice([0, 1], n_pos, p=[0.3, 0.7]),
        "cp": rng.choice([0, 1, 2, 3], n_pos, p=[0.5, 0.2, 0.2, 0.1]),
        "trestbps": sample(134, 18, 90, 200, n_pos),
        "chol": sample(251, 48, 130, 564, n_pos),
        "fbs": rng.choice([0, 1], n_pos, p=[0.85, 0.15]),
        "restecg": rng.choice([0, 1, 2], n_pos, p=[0.5, 0.4, 0.1]),
        "thalach": sample(139, 23, 70, 200, n_pos),
        "exang": rng.choice([0, 1], n_pos, p=[0.35, 0.65]),
        "oldpeak": sample(1.6, 1.4, 0, 6.2, n_pos),
        "slope": rng.choice([0, 1, 2], n_pos, p=[0.1, 0.4, 0.5]),
        "ca": rng.choice([0, 1, 2, 3], n_pos, p=[0.3, 0.3, 0.25, 0.15]),
        "thal": rng.choice([1, 2, 3], n_pos, p=[0.05, 0.3, 0.65]),
        "target": np.ones(n_pos, int),
    }
    neg = {
        "age": sample(52, 9, 30, 80, n_neg),
        "sex": rng.choice([0, 1], n_neg, p=[0.55, 0.45]),
        "cp": rng.choice([0, 1, 2, 3], n_neg, p=[0.1, 0.2, 0.4, 0.3]),
        "trestbps": sample(129, 17, 90, 200, n_neg),
        "chol": sample(243, 46, 130, 564, n_neg),
        "fbs": rng.choice([0, 1], n_neg, p=[0.85, 0.15]),
        "restecg": rng.choice([0, 1, 2], n_neg, p=[0.7, 0.25, 0.05]),
        "thalach": sample(158, 19, 70, 200, n_neg),
        "exang": rng.choice([0, 1], n_neg, p=[0.8, 0.2]),
        "oldpeak": sample(0.6, 0.9, 0, 6.2, n_neg),
        "slope": rng.choice([0, 1, 2], n_neg, p=[0.05, 0.6, 0.35]),
        "ca": rng.choice([0, 1, 2, 3], n_neg, p=[0.6, 0.25, 0.1, 0.05]),
        "thal": rng.choice([1, 2, 3], n_neg, p=[0.05, 0.6, 0.35]),
        "target": np.zeros(n_neg, int),
    }
    df = pd.DataFrame({k: np.concatenate([pos[k], neg[k]]) for k in pos})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def preprocess(df: pd.DataFrame):
    """Clean missing values, binarize target, and apply StandardScaler."""
    df = df.copy()
    
    # Coerce all columns to numeric
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0)
        df["target"] = (df["target"] > 0).astype(int)

    X = df[FEATURE_COLS].astype(float).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, X_train


def train_and_evaluate(csv_path: str = "heart.csv"):
    """Train the model, evaluate metrics, benchmark HE, and export artefacts."""
    print("\n" + "=" * 55)
    print("      CARDIOSHIELD MODEL TRAINING & VALIDATION")
    print("=" * 55)

    df = load_uci_data(csv_path)
    print(f"[data] Dataset size: {len(df)} rows, {len(FEATURE_COLS)} features")

    X_train_s, X_test_s, y_train, y_test, scaler, X_train_raw = preprocess(df)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_s, y_train)

    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="accuracy")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Holdout Test Set Evaluation
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"[cv]   5-Fold CV Accuracy: {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%")
    print(f"[test] Hold-out Accuracy:  {acc * 100:.2f}%")
    print(f"[test] Precision:          {prec * 100:.2f}%")
    print(f"[test] Recall:             {rec * 100:.2f}%")
    print(f"[test] F1 Score:           {f1 * 100:.2f}%")
    print(f"[test] ROC-AUC:            {auc * 100:.2f}%")

    # Polynomial sigmoid error over [-4, 4]
    z_eval = np.linspace(-4, 4, 10000)
    true_sig = 1.0 / (1.0 + np.exp(-z_eval))
    approx_sig = np.clip(0.5 + 0.2159 * z_eval - 0.0093 * (z_eval**3), 0.0, 1.0)
    poly_linf_err = float(np.max(np.abs(approx_sig - true_sig)))

    # Real TenSEAL HE Benchmark on Hold-out Test Samples
    print("\n[he] Running genuine TenSEAL CKKS benchmark on test set...")
    import tenseal as ts
    from he_engine import create_context, encrypt_patient_data, homomorphic_predict

    ctx = create_context()
    w = clf.coef_[0]
    b = float(clf.intercept_[0])
    enc_w = ts.ckks_vector(ctx, w.tolist())
    enc_b = ts.ckks_vector(ctx, [b])

    he_matches = 0
    he_latencies = []
    he_deltas = []

    for i in range(len(X_test_s)):
        x_sample = X_test_s[i]
        t0 = time.perf_counter()
        enc_x = encrypt_patient_data(ctx, x_sample)
        enc_pred = homomorphic_predict(enc_x, enc_w, enc_b)
        t_he = time.perf_counter() - t0

        dec_prob = float(enc_pred.decrypt()[0])
        dec_prob = max(0.0, min(1.0, dec_prob))
        he_pred_class = int(dec_prob >= 0.5)

        he_latencies.append(t_he)
        he_deltas.append(abs(dec_prob - y_prob[i]))

        if he_pred_class == y_pred[i]:
            he_matches += 1

    he_match_rate = float((he_matches / len(X_test_s)) * 100)
    avg_he_lat_ms = float(np.mean(he_latencies) * 1000)
    avg_he_delta = float(np.mean(he_deltas))

    print(f"[he] Plaintext <-> HE Class Match Rate: {he_match_rate:.1f}%")
    print(f"[he] Average HE Inference Latency:   {avg_he_lat_ms:.1f} ms")
    print(f"[he] Average Probability Delta:       {avg_he_delta:.4f}")

    # Feature weights dictionary
    feature_importance = {
        name: float(weight) for name, weight in zip(FEATURE_COLS, w)
    }

    metrics_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_type": "Logistic Regression (L-BFGS, Balanced)",
        "sample_count": len(df),
        "test_sample_count": len(X_test_s),
        "accuracy": round(acc * 100, 2),
        "cv_accuracy_mean": round(cv_mean * 100, 2),
        "cv_accuracy_std": round(cv_std * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2),
        "confusion_matrix": cm,
        "poly_sigmoid_max_error": round(poly_linf_err, 5),
        "he_match_rate": round(he_match_rate, 2),
        "avg_he_latency_ms": round(avg_he_lat_ms, 2),
        "avg_he_prob_delta": round(avg_he_delta, 5),
        "ckks_parameters": {
            "poly_modulus_degree": 8192,
            "security_level": "128-bit RLWE",
            "scale": "2^20",
            "sigmoid_polynomial_degree": 3,
            "multiplicative_depth": 2,
        },
        "feature_importance": feature_importance,
    }

    # Save to artifacts/ directory
    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(scaler.transform(X_train_raw[:100]), f)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n[save] Successfully exported artifacts and metrics.json!")
    print("=" * 55 + "\n")
    return metrics_data


if __name__ == "__main__":
    train_and_evaluate()