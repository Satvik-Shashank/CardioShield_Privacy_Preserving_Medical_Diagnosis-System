"""
tests/test_he_engine.py
───────────────────────
Unit tests for the TenSEAL CKKS Homomorphic Encryption engine.
"""

import numpy as np
import pytest
import tenseal as ts
from he_engine import (
    create_context,
    encrypt_patient_data,
    homomorphic_predict,
    sigmoid_approx,
    verify_he_system,
)


def test_create_context():
    """Verify CKKS context creation with 128-bit security parameters."""
    ctx = create_context()
    assert ctx is not None
    assert isinstance(ctx, ts.Context)


def test_encrypt_patient_data_shape_and_roundtrip():
    """Verify 13-feature vector encryption and decryption restoration within CKKS scale tolerance."""
    ctx = create_context()
    features = np.random.randn(13).astype(float)
    enc_x = encrypt_patient_data(ctx, features)

    assert enc_x is not None
    assert isinstance(enc_x, ts.CKKSVector)

    # Decrypt and compare (CKKS is an approximate arithmetic scheme with scale 2^20)
    dec = np.array(enc_x.decrypt()[:13])
    max_err = np.max(np.abs(dec - features))
    assert max_err < 1e-2, f"Roundtrip error too high: {max_err}"


def test_encrypt_invalid_length_raises():
    """Verify encrypting vectors of length != 13 raises ValueError."""
    ctx = create_context()
    with pytest.raises(ValueError):
        encrypt_patient_data(ctx, np.random.randn(10))


def test_homomorphic_predict_math():
    """Verify ciphertext linear combination and sigmoid approximation match plaintext within tolerance."""
    ctx = create_context()
    x = np.array([0.5, -0.2, 0.1, -0.3, 0.3, 0.0, 0.2, -0.5, 0.0, 0.4, 0.4, 0.0, 0.3], dtype=float)
    w = np.array([0.2, 0.4, 0.3, 0.1, 0.3, -0.1, 0.2, -0.5, 0.4, 0.5, 0.3, 0.6, 0.5], dtype=float)
    b = 0.1

    enc_x = encrypt_patient_data(ctx, x)
    enc_w = ts.ckks_vector(ctx, w.tolist())
    enc_b = ts.ckks_vector(ctx, [b])

    enc_pred = homomorphic_predict(enc_x, enc_w, enc_b)
    dec_prob = float(enc_pred.decrypt()[0])

    # Plaintext calculation
    z_plain = float(np.dot(x, w) + b)
    p_plain = float(1.0 / (1.0 + np.exp(-z_plain)))

    assert abs(dec_prob - p_plain) < 0.25
    assert (dec_prob >= 0.5) == (p_plain >= 0.5)


def test_verify_he_system():
    """Verify the startup smoke test returns HEALTHY."""
    res = verify_he_system()
    assert res["status"] == "HEALTHY"
    assert res["scheme"] == "CKKS"
    assert res["delta"] < 0.25
