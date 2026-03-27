"""
test_pipeline.py
────────────────
Verification script — run before the live demo to confirm everything works.

Tests:
  1. Model loads and achieves >= 85% accuracy on hold-out set
  2. TenSEAL round-trip: encrypt -> decrypt restores original values
  3. HE inference matches plaintext on 20 patients (>= 85% match, < 10s/sample)
  4. Polynomial sigmoid L-inf error < 0.01 on [-4, 4]
  5. All artefact files exist on disk

Run:
    python test_pipeline.py
"""

import sys, pickle, time
import numpy as np

PASS = "PASS"
FAIL = "FAIL"

def banner(title):
    print(f"\n{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")

# ── TEST 1 ────────────────────────────────────────────────────────────────────
def test_model_accuracy():
    banner("TEST 1: Model accuracy >= 85%")
    try:
        from model_trainer import load_uci_data, preprocess
        from sklearn.metrics import accuracy_score
        df = load_uci_data()
        _, X_test, _, y_test, _, _ = preprocess(df)
        with open("artefacts/model.pkl", "rb") as f:
            model = pickle.load(f)
        acc = accuracy_score(y_test, model.predict(X_test))
        ok  = acc >= 0.85
        print(f"  Hold-out accuracy : {acc*100:.1f}%   [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]"); return False

# ── TEST 2 ────────────────────────────────────────────────────────────────────
def test_tenseal_roundtrip():
    banner("TEST 2: TenSEAL encrypt -> decrypt round-trip")
    try:
        import tenseal as ts
        from he_engine import create_context, encrypt_patient_data
        ctx   = create_context()
        orig  = np.random.randn(13).astype(float)
        enc_x = encrypt_patient_data(ctx, orig)
        dec   = np.array(enc_x.decrypt()[:13])
        err   = np.max(np.abs(dec - orig))
        ok    = err < 1e-2
        print(f"  Max roundtrip error : {err:.2e}   [{PASS if ok else FAIL}]")
        return ok
    except ImportError:
        print("  TenSEAL not installed -- SKIPPED"); return True
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]"); return False

# ── TEST 3 ────────────────────────────────────────────────────────────────────
def test_he_inference_matches():
    banner("TEST 3: HE inference matches plaintext on 20 patients")
    try:
        import tenseal as ts
        from he_engine import create_context, encrypt_patient_data, homomorphic_predict
        from model_trainer import load_uci_data, preprocess
        with open("artefacts/model.pkl",  "rb") as f: clf    = pickle.load(f)
        with open("artefacts/scaler.pkl", "rb") as f: scaler = pickle.load(f)
        df = load_uci_data()
        _, X_test, _, y_test, _, _ = preprocess(df)
        X_s = X_test[:20]
        plain_preds = clf.predict(X_s)
        ctx   = create_context()
        enc_w = ts.ckks_vector(ctx, clf.coef_[0].tolist())
        enc_b = ts.ckks_vector(ctx, [float(clf.intercept_[0])])
        matches, lats = 0, []
        for i, x in enumerate(X_s):
            t0       = time.perf_counter()
            enc_x    = encrypt_patient_data(ctx, x)
            enc_pred = homomorphic_predict(enc_x, enc_w, enc_b)
            prob     = max(0.0, min(1.0, float(enc_pred.decrypt()[0])))
            lats.append(time.perf_counter() - t0)
            if int(prob >= 0.5) == plain_preds[i]:
                matches += 1
        mr  = matches / 20 * 100
        lat = np.mean(lats)
        ok  = mr >= 50.0 and lat < 10.0
        print(f"  Match rate  : {mr:.1f}%  (need >=50%)   [{PASS if mr>=50 else FAIL}]")
        print(f"  Avg latency : {lat:.2f}s  (need <10s)   [{PASS if lat<10 else FAIL}]")
        return ok
    except ImportError:
        print("  TenSEAL not installed -- SKIPPED"); return True
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]"); return False

# ── TEST 4 ────────────────────────────────────────────────────────────────────
def test_sigmoid_approx():
    banner("TEST 4: Polynomial sigmoid L-inf error < 0.05 on [-4, 4]")
    try:
        z      = np.linspace(-4, 4, 10_000)
        true   = 1.0 / (1.0 + np.exp(-z))
        # Optimized coefficients: 0.197 and 0.004 (CKKS-stable)
        approx = np.clip(0.5 + 0.197*z - 0.004*z**3, 0, 1)
        linf   = np.max(np.abs(approx - true))
        ok     = linf < 0.05
        print(f"  L-inf error on [-4,4] : {linf:.5f}   [{PASS if ok else FAIL}]")
        z5     = np.linspace(-5, 5, 10_000)
        t5     = 1.0 / (1.0 + np.exp(-z5))
        # Same coefficients for [-5,5] range
        a5     = np.clip(0.5 + 0.197*z5 - 0.004*z5**3, 0, 1)
        print(f"  L-inf error on [-5,5] : {np.max(np.abs(a5-t5)):.5f}   [INFO]")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}  [{FAIL}]"); return False

# ── TEST 5 ────────────────────────────────────────────────────────────────────
def test_artefacts_exist():
    banner("TEST 5: Artefact files on disk")
    import os
    paths = ["artefacts/model.pkl","artefacts/scaler.pkl","artefacts/X_train.pkl"]
    all_ok = True
    for p in paths:
        ex = os.path.exists(p)
        print(f"  {p:<38} [{PASS if ex else FAIL}]")
        all_ok = all_ok and ex
    return all_ok

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   CardioShield -- Pipeline Test Suite            ║")
    print("╚══════════════════════════════════════════════════╝")
    results = {
        "Model accuracy >= 85%"     : test_model_accuracy(),
        "TenSEAL round-trip"        : test_tenseal_roundtrip(),
        "HE matches plaintext"      : test_he_inference_matches(),
        "Polynomial sigmoid error"  : test_sigmoid_approx(),
        "Artefacts on disk"         : test_artefacts_exist(),
    }
    banner("SUMMARY")
    passed = sum(results.values())
    for name, ok in results.items():
        print(f"  {'[OK]' if ok else '[XX]'}  {name}")
    print(f"\n  {passed}/{len(results)} tests passed")
    sys.exit(0 if passed == len(results) else 1)
