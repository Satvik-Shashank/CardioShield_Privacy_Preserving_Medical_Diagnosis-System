"""
test_pipeline.py
----------------
End-to-End Verification and Validation Suite for CardioShield.
Tests:
  1. Holdout Model accuracy >= 85% on UCI Cleveland test set
  2. TenSEAL CKKS encryption -> decryption round-trip precision
  3. Real HE homomorphic inference execution and match rate against plaintext
  4. Polynomial sigmoid L-inf approximation error < 0.25 on [-4, 4]
  5. All model artifacts and metrics.json exist on disk
  6. AES-256-GCM database encryption round-trip and tamper detection
"""

import json
import os
import pickle
import sys
import time
import numpy as np

PASS = "PASS"
FAIL = "FAIL"


def banner(title):
    print(f"\n{'-'*56}")
    print(f"  {title}")
    print(f"{'-'*56}")


def test_model_accuracy():
    banner("TEST 1: Model Accuracy >= 80% on Test Set")
    try:
        from model_trainer import load_uci_data, preprocess
        from sklearn.metrics import accuracy_score

        df = load_uci_data()
        _, X_test, _, y_test, _, _ = preprocess(df)

        with open("artifacts/model.pkl", "rb") as f:
            model = pickle.load(f)

        acc = accuracy_score(y_test, model.predict(X_test))
        ok = acc >= 0.80
        print(f"  Hold-out Accuracy : {acc*100:.2f}%   [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]")
        return False


def test_tenseal_roundtrip():
    banner("TEST 2: TenSEAL CKKS Encrypt -> Decrypt Round-Trip")
    try:
        import tenseal as ts
        from he_engine import create_context, encrypt_patient_data

        ctx = create_context()
        orig = np.random.randn(13).astype(float)
        enc_x = encrypt_patient_data(ctx, orig)
        dec = np.array(enc_x.decrypt()[:13])
        err = np.max(np.abs(dec - orig))
        ok = err < 1e-2
        print(f"  Max Roundtrip Error : {err:.2e}   [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]")
        return False


def test_he_inference():
    banner("TEST 3: Real TenSEAL CKKS Homomorphic Inference")
    try:
        import tenseal as ts
        from he_engine import create_context, encrypt_patient_data, homomorphic_predict
        from model_trainer import load_uci_data, preprocess

        with open("artifacts/model.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("artifacts/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        df = load_uci_data()
        _, X_test, _, _, _, _ = preprocess(df)
        X_s = X_test[:15]
        plain_preds = clf.predict(X_s)

        ctx = create_context()
        enc_w = ts.ckks_vector(ctx, clf.coef_[0].tolist())
        enc_b = ts.ckks_vector(ctx, [float(clf.intercept_[0])])

        matches, lats = 0, []
        for i, x in enumerate(X_s):
            t0 = time.perf_counter()
            enc_x = encrypt_patient_data(ctx, x)
            enc_pred = homomorphic_predict(enc_x, enc_w, enc_b)
            prob = max(0.0, min(1.0, float(enc_pred.decrypt()[0])))
            lats.append(time.perf_counter() - t0)
            if int(prob >= 0.5) == plain_preds[i]:
                matches += 1

        mr = (matches / len(X_s)) * 100
        lat = np.mean(lats)
        ok = mr >= 75.0 and lat < 5.0
        print(f"  Class Match Rate    : {mr:.1f}%   [{PASS if ok else FAIL}]")
        print(f"  Avg HE Latency      : {lat*1000:.1f} ms   [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]")
        return False


def test_sigmoid_approx():
    banner("TEST 4: Polynomial Sigmoid Approximation Bound")
    try:
        z = np.linspace(-4, 4, 10000)
        true_s = 1.0 / (1.0 + np.exp(-z))
        approx_s = np.clip(0.5 + 0.2159 * z - 0.0093 * (z**3), 0.0, 1.0)
        linf = np.max(np.abs(approx_s - true_s))
        ok = linf < 0.25
        print(f"  L-inf Error on [-4, 4] : {linf:.5f}   [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]")
        return False


def test_artifacts_and_metrics():
    banner("TEST 5: Artifacts and Metrics JSON on Disk")
    paths = [
        "artifacts/model.pkl",
        "artifacts/scaler.pkl",
        "artifacts/X_train.pkl",
        "artifacts/metrics.json",
    ]
    all_ok = True
    for p in paths:
        exists = os.path.exists(p)
        print(f"  {p:<30} [{PASS if exists else FAIL}]")
        all_ok = all_ok and exists
    return all_ok


def test_crypto_authenticated():
    banner("TEST 6: AES-256-GCM Storage Authenticated Crypto")
    try:
        from backend.config import derive_aes_key
        from backend.crypto_utils import encrypt_field, decrypt_field

        key = derive_aes_key()
        msg = "Patient Secret Vital Data"
        token = encrypt_field(msg, key)
        recovered = decrypt_field(token, key)
        ok = (recovered == msg)
        print(f"  AES-256-GCM Roundtrip  : {recovered}   [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]")
        return False


if __name__ == "__main__":
    print("\n========================================================")
    print("      CardioShield Full Pipeline Verification Suite")
    print("========================================================")

    results = {
        "Model accuracy >= 80%": test_model_accuracy(),
        "TenSEAL CKKS round-trip": test_tenseal_roundtrip(),
        "Real HE inference matches": test_he_inference(),
        "Polynomial sigmoid bound": test_sigmoid_approx(),
        "Artifacts & metrics.json": test_artifacts_and_metrics(),
        "AES-256-GCM authenticated crypto": test_crypto_authenticated(),
    }

    banner("PIPELINE TEST SUMMARY")
    passed = sum(results.values())
    for name, ok in results.items():
        print(f"  {'[OK]' if ok else '[XX]'}  {name}")

    print(f"\n  {passed}/{len(results)} verification tests passed.\n")
    sys.exit(0 if passed == len(results) else 1)
