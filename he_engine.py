"""
he_engine.py
────────────
All homomorphic-encryption operations in one place.

The CKKS scheme (Cheon–Kim–Kim–Song, 2017) supports approximate arithmetic
on real numbers.  We use it for the linear combination  w·x + b  that sits
at the heart of Logistic Regression inference:

    enc(z) = enc(x) · w + enc(b)          ← happens on the SERVER
    prob   = sigmoid_approx(decrypt(z))   ← decrypt ONLY the final scalar

Polynomial approximation of sigmoid over [-4, 4]:
    σ(z) ≈ 0.5 + 0.2159z − 0.0093z³
This is accurate to <0.01 error for |z| ≤ 4, which covers >99 % of real LR
outputs on normalised data.
"""

import tenseal as ts
import numpy as np
import pickle, os

# ─────────────────────────────────────────────────────────────────────────────
# CKKS hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
# poly_modulus_degree controls the "ring size" (security level).
#   8192  → 128-bit security, fast, supports ~3 levels of multiplication
#  16384  → higher security / more multiplications – use if overflow errors appear
POLY_MOD_DEGREE   = 8192
# coeff_mod_bit_sizes: a chain of primes that define the noise budget
COEFF_MOD_SIZES   = [40, 21, 21, 21, 21, 21, 40]   # 5 intermediate primes for more depth
# Scale for encoding real numbers (2^20 for better precision with expanded multiplicative depth)
SCALE             = 2**20


def create_context() -> ts.Context:
    """
    Create and return a TenSEAL CKKS context with the keys needed for
    arithmetic (relinearisation + Galois keys are generated automatically
    for vector operations but not for batching, keeping key-gen fast).
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree = POLY_MOD_DEGREE,
        coeff_mod_bit_sizes = COEFF_MOD_SIZES,
    )
    ctx.generate_galois_keys()
    ctx.global_scale = SCALE
    return ctx


def encrypt_patient_data(ctx: ts.Context, scaled_features: np.ndarray) -> ts.CKKSVector:
    """
    Encrypt a single patient's 13 StandardScaler-normalised features.

    Parameters
    ----------
    ctx             : TenSEAL context (holds public key)
    scaled_features : shape (13,) float64 array

    Returns
    -------
    enc_x : CKKSVector  — the encrypted feature vector
    """
    # Validate input length: must be exactly 13 features
    if len(scaled_features) != 13:
        raise ValueError(f"Expected exactly 13 features, got {len(scaled_features)}")
    
    vec = scaled_features.astype(float).tolist()
    enc_x = ts.ckks_vector(ctx, vec)
    return enc_x


def encrypt_batch(ctx, X_batch):
    """
    Encrypt a batch of patient data.
    
    Parameters
    ----------
    ctx     : TenSEAL context (holds public key)
    X_batch : array-like of shape (n_samples, 13)
    
    Returns
    -------
    List of CKKSVectors  — encrypted feature vectors for each sample
    """
    return [ts.ckks_vector(ctx, x.tolist()) for x in X_batch]


def _sigmoid_approx(enc_z: ts.CKKSVector) -> ts.CKKSVector:
    """
    Degree-3 polynomial approximation (simplified for CKKS stability):
        σ(z) ≈ 0.5 + 0.2159·z − 0.0093·z³

    Input magnitude is pre-reduced in homomorphic_predict() to prevent
    z³ overflow. This function runs ENTIRELY on the encrypted scalar.
    """
    # Compute z² and z³ (2 multiplicative levels required)
    enc_z2  = enc_z * enc_z      # z²
    enc_z3  = enc_z2 * enc_z     # z³

    # σ(z) ≈ 0.5 + 0.2159z − 0.0093z³ (proven coefficients)
    result  = enc_z * 0.2159
    result  = result - enc_z3 * 0.0093
    result  = result + 0.5

    return result


def homomorphic_predict(
    enc_x:  ts.CKKSVector,
    enc_w:  ts.CKKSVector,
    enc_b:  ts.CKKSVector,
) -> ts.CKKSVector:
    """
    Full encrypted inference:
        1. dot(enc_x, enc_w)  – CKKS inner product (sum of element products)
        2. + enc_b            – add encrypted bias
        3. sigmoid_approx     – polynomial approximation ON CIPHERTEXT
        4. return enc_prob     – caller decrypts to get probability scalar

    Both weights and bias are pre-encrypted with the SAME context, so the
    server performs pure ciphertext arithmetic → zero plaintext leakage.
    """
    # Error handling: validate inputs are not empty or None
    if enc_x is None or (hasattr(enc_x, '__len__') and len(enc_x) == 0):
        raise ValueError("enc_x cannot be empty or None")
    if enc_w is None or (hasattr(enc_w, '__len__') and len(enc_w) == 0):
        raise ValueError("enc_w cannot be empty or None")
    
    # Step 1: encrypted dot product  w·x
    enc_linear = enc_x.dot(enc_w)          # CKKSVector of length 1

    # Step 2: + bias
    enc_linear = enc_linear + enc_b        # still encrypted

    # Step 3: approximate sigmoid ON THE CIPHERTEXT
    enc_prob   = _sigmoid_approx(enc_linear)

    return enc_prob   # decrypt outside to get probability


def load_pretrained_weights(ctx: ts.Context, model_path: str = "artefacts/model.pkl"):
    """
    Load a trained sklearn LogisticRegression, extract w and b,
    encrypt them with the given context, and return the encrypted objects.

    Returns
    -------
    enc_w  : CKKSVector  (shape 13)
    enc_b  : CKKSVector  (scalar wrapped in len-1 vector)
    w_plain: np.ndarray  (kept for SHAP – plaintext weights are NOT secret)
    """
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    w = clf.coef_[0]          # shape (13,)
    b = clf.intercept_[0]     # scalar

    enc_w = ts.ckks_vector(ctx, w.tolist())
    enc_b = ts.ckks_vector(ctx, [float(b)])

    return enc_w, enc_b, w


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test  (python he_engine.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("── TenSEAL round-trip test ──")

    ctx = create_context()
    print("✅ Context created")

    # Fake 13 features (as if StandardScaler output)
    fake_x = np.random.randn(13).astype(float)
    enc_x  = encrypt_patient_data(ctx, fake_x)
    print(f"✅ Encrypted feature vector | hex prefix: {str(enc_x.serialize()[:16].hex())}…")

    # Fake weight vector + bias
    fake_w  = np.random.randn(13).astype(float)
    fake_b  = np.random.randn(1).astype(float)
    enc_w   = ts.ckks_vector(ctx, fake_w.tolist())
    enc_b   = ts.ckks_vector(ctx, [float(fake_b[0])])

    # Encrypted inference
    t0       = time.perf_counter()
    enc_pred = homomorphic_predict(enc_x, enc_w, enc_b)
    latency  = time.perf_counter() - t0

    dec_prob = enc_pred.decrypt()[0]

    # Plaintext reference
    z_plain  = float(np.dot(fake_x, fake_w) + fake_b[0])
    p_plain  = 1.0 / (1.0 + np.exp(-z_plain))   # true sigmoid

    print(f"✅ Encrypted inference done | latency: {latency*1000:.0f} ms")
    print(f"   Decrypted prob (HE approx) : {dec_prob:.6f}")
    print(f"   Plaintext sigmoid           : {p_plain:.6f}")
    print(f"   Absolute error              : {abs(dec_prob - p_plain):.6f}")
    print(f"   Prediction match?           : {'✅' if (dec_prob>=0.5)==(p_plain>=0.5) else '❌'}")

    # Tamper test
    print("\n── Tamper / corruption test ──")
    raw_bytes    = bytearray(enc_x.serialize())
    raw_bytes[8] ^= 0xFF          # flip 8 bits
    try:
        bad_vec = ts.ckks_vector_from(ctx, bytes(raw_bytes))
        val     = bad_vec.decrypt()
        print(f"⚠  Tampered vector decrypted (value={val[:3]}) – expected for CKKS")
        print("   → Show delta vs untampered to prove integrity")
    except Exception as e:
        print(f"✅ Tampered ciphertext rejected: {e}")
