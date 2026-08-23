"""
he_engine.py
────────────
Strict, verified Homomorphic Encryption Engine using TenSEAL (CKKS Scheme).

The CKKS scheme (Cheon–Kim–Kim–Song, 2017) enables approximate arithmetic
on encrypted real-number vectors:

    enc(z) = enc(x) · w + enc(b)          (Calculated on ciphertext)
    prob   = sigmoid_approx(decrypt(z))   (Decrypted at caller/enclave boundary)

Polynomial approximation of sigmoid over [-4, 4]:
    σ(z) ≈ 0.5 + 0.2159z − 0.0093z³
"""

import os
import pickle
import time
from typing import Tuple
import numpy as np
import tenseal as ts

# ─────────────────────────────────────────────────────────────────────────────
# CKKS Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
POLY_MOD_DEGREE = 8192
COEFF_MOD_SIZES = [40, 21, 21, 21, 21, 21, 40]  # 128-bit RLWE security
SCALE = 2**20


def create_context() -> ts.Context:
    """
    Create and return a TenSEAL CKKS context with Galois and relinearization keys.
    """
    try:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=POLY_MOD_DEGREE,
            coeff_mod_bit_sizes=COEFF_MOD_SIZES,
        )
        ctx.generate_galois_keys()
        ctx.global_scale = SCALE
        return ctx
    except Exception as e:
        raise RuntimeError(f"Failed to initialize TenSEAL CKKS context: {e}") from e


def encrypt_patient_data(ctx: ts.Context, scaled_features: np.ndarray) -> ts.CKKSVector:
    """
    Encrypt a single patient's 13 StandardScaler-normalised features into a CKKSVector.
    """
    if ctx is None:
        raise ValueError("TenSEAL context is null/uninitialized.")
    if len(scaled_features) != 13:
        raise ValueError(f"Expected exactly 13 features, got {len(scaled_features)}")
    
    vec = scaled_features.astype(float).tolist()
    return ts.ckks_vector(ctx, vec)


def encrypt_batch(ctx: ts.Context, X_batch: np.ndarray) -> list:
    """
    Encrypt a batch of patient feature vectors.
    """
    return [encrypt_patient_data(ctx, x) for x in X_batch]


def sigmoid_approx(enc_z: ts.CKKSVector) -> ts.CKKSVector:
    """
    Degree-3 polynomial approximation of the sigmoid activation function:
        σ(z) ≈ 0.5 + 0.2159·z − 0.0093·z³
    Computed entirely over the ciphertext without decryption.
    """
    enc_z2 = enc_z * enc_z      # z²
    enc_z3 = enc_z2 * enc_z     # z³

    result = enc_z * 0.2159
    result = result - enc_z3 * 0.0093
    result = result + 0.5

    return result


def homomorphic_predict(
    enc_x: ts.CKKSVector,
    enc_w: ts.CKKSVector,
    enc_b: ts.CKKSVector,
) -> ts.CKKSVector:
    """
    Full Homomorphic Inference:
      1. enc_linear = enc_x.dot(enc_w)  (Ciphertext dot product)
      2. enc_linear = enc_linear + enc_b (Add encrypted bias)
      3. enc_prob = sigmoid_approx(enc_linear) (Polynomial approximation on ciphertext)
    """
    if enc_x is None or enc_w is None or enc_b is None:
        raise ValueError("Inputs to homomorphic_predict cannot be None.")

    enc_linear = enc_x.dot(enc_w)
    enc_linear = enc_linear + enc_b
    enc_prob = sigmoid_approx(enc_linear)

    return enc_prob


def load_and_encrypt_weights(ctx: ts.Context, model_path: str = "artifacts/model.pkl") -> Tuple[ts.CKKSVector, ts.CKKSVector, np.ndarray, float]:
    """
    Load trained weights and bias from model pickle and encrypt them with the CKKS context.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    w = clf.coef_[0]
    b = float(clf.intercept_[0])

    enc_w = ts.ckks_vector(ctx, w.tolist())
    enc_b = ts.ckks_vector(ctx, [b])

    return enc_w, enc_b, w, b


def verify_he_system() -> dict:
    """
    Perform a complete startup smoke test for the HE engine.
    Ensures context creation, encryption, ciphertext inference, and decryption work properly.
    """
    t0 = time.perf_counter()
    ctx = create_context()

    fake_x = np.array([0.5, -0.2, 0.1, -0.3, 0.3, 0.0, 0.2, -0.5, 0.0, 0.4, 0.4, 0.0, 0.3], dtype=float)
    fake_w = np.array([0.2, 0.4, 0.3, 0.1, 0.3, -0.1, 0.2, -0.5, 0.4, 0.5, 0.3, 0.6, 0.5], dtype=float)
    fake_b = 0.1

    enc_x = encrypt_patient_data(ctx, fake_x)
    enc_w = ts.ckks_vector(ctx, fake_w.tolist())
    enc_b = ts.ckks_vector(ctx, [fake_b])

    t_inf_start = time.perf_counter()
    enc_pred = homomorphic_predict(enc_x, enc_w, enc_b)
    t_inf = time.perf_counter() - t_inf_start

    dec_prob = float(enc_pred.decrypt()[0])
    dec_prob = max(0.0, min(1.0, dec_prob))

    # Plaintext reference
    z_plain = float(np.dot(fake_x, fake_w) + fake_b)
    p_plain = float(1.0 / (1.0 + np.exp(-z_plain)))

    abs_err = abs(dec_prob - p_plain)
    is_healthy = abs_err < 0.25 and ((dec_prob >= 0.5) == (p_plain >= 0.5))

    if not is_healthy:
        raise RuntimeError(f"HE verification error outside tolerance: delta={abs_err:.4f}")

    return {
        "status": "HEALTHY",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "inference_ms": round(t_inf * 1000, 2),
        "he_prob": round(dec_prob, 5),
        "plain_prob": round(p_plain, 5),
        "delta": round(abs_err, 5),
        "scheme": "CKKS",
        "poly_modulus_degree": POLY_MOD_DEGREE,
        "security_level": "128-bit RLWE",
    }


if __name__ == "__main__":
    import sys
    print("Running TenSEAL CKKS Self-Test...")
    res = verify_he_system()
    print("HE Self-Test Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    sys.exit(0 if res["status"] == "HEALTHY" else 1)
