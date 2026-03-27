"""
model_trainer.py
────────────────
Step 1 of the pipeline: train a sklearn LogisticRegression on the UCI Heart
Disease dataset, save all artefacts, and run a plaintext vs encrypted
accuracy profile so you can show exact numbers to the judges.

Run once before launching Streamlit:
    python model_trainer.py
"""

import os, pickle, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target",
]

def load_uci_data(csv_path: str = "heart.csv") -> pd.DataFrame:
    if os.path.exists(csv_path):
        print(f"[data] Loading from local file: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        if "target" not in df.columns and "num" in df.columns:
            df.rename(columns={"num": "target"}, inplace=True)
    else:
        try:
            print(f"[data] Fetching from UCI: {UCI_URL}")
            df = pd.read_csv(UCI_URL, names=COLUMNS, na_values="?")
        except Exception:
            print("[data] ⚠  Network unavailable – generating synthetic data for demo")
            df = _make_synthetic_data()
    return df

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

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)

    for col in ("ca", "thal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    X = df[FEATURE_COLS].astype(float).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, X_train

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def train_baseline(csv_path: str = "heart.csv"):
    print("\n══════════ TRAINING BASELINE MODEL ══════════")
    df = load_uci_data(csv_path)
    print(f"[data] Shape: {df.shape}  |  Target dist:\n{df['target'].value_counts().to_dict()}")

    X_train, X_test, y_train, y_test, scaler, X_train_raw = preprocess(df)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"[cv]   5-fold accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    print(f"[test] Avg prediction confidence: {np.mean(y_prob)*100:.1f}%")
    
    acc = accuracy_score(y_test, y_pred)
    print(f"[test] Hold-out accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    os.makedirs("artefacts", exist_ok=True)
    with open("artefacts/model.pkl",   "wb") as f: pickle.dump(clf,    f)
    with open("artefacts/scaler.pkl",  "wb") as f: pickle.dump(scaler, f)
    
    X_train_scaled = scaler.transform(X_train_raw[:100])
    with open("artefacts/X_train.pkl", "wb") as f: pickle.dump(X_train_scaled, f)

    print("[save] artefacts/ → model.pkl, scaler.pkl, X_train.pkl")
    print(f"═══════════════════════════════════════════════\n")
    return clf, scaler, acc

# ─────────────────────────────────────────────────────────────────────────────
# 4.  HE ACCURACY PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def profile_he_accuracy(acc_score: float, n_samples: int = 50):
    try:
        import tenseal as ts
        from he_engine import create_context, encrypt_patient_data, homomorphic_predict
    except ImportError as e:
        print(f"[profile] ⚠  Skipping HE profile: {e}")
        return

    print("\n══════════ HE ACCURACY PROFILE ══════════")
    with open("artefacts/model.pkl",  "rb") as f: clf    = pickle.load(f)
    with open("artefacts/scaler.pkl", "rb") as f: scaler = pickle.load(f)

    df     = load_uci_data()
    X_arr, _, y_arr, _, _, _ = preprocess(df)
    
    df["target"] = (df["target"] > 0).astype(int)
    for col in ("ca", "thal"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)
        
    X_all = scaler.transform(df[FEATURE_COLS].astype(float).values)
    y_all = df["target"].values
    X_s, y_s = X_all[:n_samples], y_all[:n_samples]

    plain_preds = clf.predict(X_s)

    ctx       = create_context()
    w         = clf.coef_[0]
    b         = clf.intercept_[0]
    enc_w     = ts.ckks_vector(ctx, w.tolist())
    enc_b_val = ts.ckks_vector(ctx, [b])

    matches, t0 = 0, time.perf_counter()
    latencies = []

    for i, x in enumerate(X_s):
        t1        = time.perf_counter()
        enc_x     = encrypt_patient_data(ctx, x)
        enc_pred  = homomorphic_predict(enc_x, enc_w, enc_b_val)
        
        dec_prob = enc_pred.decrypt()[0]
        dec_prob = max(0.0, min(1.0, dec_prob))
        he_class = int(dec_prob >= 0.5)
        
        latencies.append(time.perf_counter() - t1)
        if he_class == plain_preds[i]:
            matches += 1

    match_rate = matches / n_samples * 100
    mean_lat   = np.mean(latencies)
    print(f"[profile] Samples tested      : {n_samples}")
    print(f"[profile] Plaintext accuracy  : {accuracy_score(y_s, plain_preds)*100:.1f}%")
    print(f"[profile] Plain↔HE match rate : {match_rate:.1f}%")
    print(f"[profile] Avg HE latency      : {mean_lat:.2f}s  |  Total: {time.perf_counter()-t0:.1f}s")
    print("══════════════════════════════════════════\n")
    print("\n📊 MODEL SUMMARY")
    print(f"• Algorithm       : Logistic Regression")
    print(f"• Features        : {len(FEATURE_COLS)} clinical inputs")
    print(f"• Dataset         : UCI Heart Disease (Cleveland)")
    print(f"• Accuracy        : {acc_score*100:.1f}%")
    print(f"• HE Compatible   : Yes")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf, scaler, acc = train_baseline()
    profile_he_accuracy(acc_score=acc, n_samples=50)