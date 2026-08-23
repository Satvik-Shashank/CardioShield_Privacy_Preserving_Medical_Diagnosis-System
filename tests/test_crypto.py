"""
tests/test_crypto.py
────────────────────
Unit tests for AES-256-GCM authenticated encryption in backend/crypto_utils.py.
"""

import base64
import os
import pytest
from backend.config import derive_aes_key
from backend.crypto_utils import (
    decrypt_field,
    decrypt_json_blob,
    decrypt_patient_record,
    encrypt_field,
    encrypt_json_blob,
    encrypt_patient_record,
)


@pytest.fixture
def key():
    return derive_aes_key()


def test_encrypt_decrypt_field(key):
    msg = "CardioShield Secret Patient 12345"
    token = encrypt_field(msg, key)
    assert token != msg
    decrypted = decrypt_field(token, key)
    assert decrypted == msg


def test_tamper_detection(key):
    msg = "Sensitive Clinical Result"
    token = encrypt_field(msg, key)

    # Tamper with the ciphertext bytes
    raw = bytearray(base64.b64decode(token))
    raw[-5] ^= 0xFF
    bad_token = base64.b64encode(raw).decode("ascii")

    with pytest.raises(Exception):
        decrypt_field(bad_token, key)


def test_encrypt_decrypt_patient_record(key):
    record = {
        "name": "Jane Doe",
        "age": 58,
        "chol": 240.5,
        "is_active": True,
    }
    enc_rec = encrypt_patient_record(record, key)
    assert enc_rec["name"] != record["name"]

    dec_rec = decrypt_patient_record(enc_rec, key)
    assert dec_rec["name"] == record["name"]
    assert dec_rec["age"] == record["age"]
    assert dec_rec["chol"] == record["chol"]
    assert dec_rec["is_active"] == record["is_active"]


def test_encrypt_decrypt_json_blob(key):
    blob = {
        "shap_values": {"age": 0.25, "chol": 0.81, "thalach": -0.42},
        "tags": ["critical", "cardiac-review"],
    }
    token = encrypt_json_blob(blob, key)
    dec_blob = decrypt_json_blob(token, key)
    assert dec_blob == blob
