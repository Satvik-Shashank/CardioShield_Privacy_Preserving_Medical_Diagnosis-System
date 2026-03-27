"""
crypto_utils.py
───────────────
AES-256-GCM encryption / decryption for patient data.

Every field is encrypted independently so that:
  • Each ciphertext has a unique random nonce (IV)
  • Tampering with any field is detected (GCM authentication tag)
  • Granular decryption is possible (e.g. decrypt only the name)

Wire format (base64-encoded):
    nonce (12 bytes) || ciphertext (variable) || tag (16 bytes)
"""

import base64
import json
from Crypto.Cipher import AES


# ── Primitive operations ─────────────────────────────────────────────────────

def encrypt_field(plaintext: str, key: bytes) -> str:
    """
    Encrypt a single string field with AES-256-GCM.

    Returns
    -------
    str : base64-encoded  nonce_len(1 byte) || nonce || ciphertext || tag(16 bytes)
    """
    cipher     = AES.new(key, AES.MODE_GCM)
    ct, tag    = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
    nonce      = cipher.nonce
    # prefix with nonce length (1 byte) so decrypt knows the split point
    raw        = bytes([len(nonce)]) + nonce + ct + tag
    return base64.b64encode(raw).decode("ascii")


def decrypt_field(token: str, key: bytes) -> str:
    """
    Decrypt and authenticate a base64-encoded AES-256-GCM token.

    Raises ValueError if authentication fails (tampered data).
    """
    raw       = base64.b64decode(token)
    nonce_len = raw[0]
    nonce     = raw[1 : 1 + nonce_len]
    tag       = raw[-16:]
    ct        = raw[1 + nonce_len : -16]
    cipher    = AES.new(key, AES.MODE_GCM, nonce=nonce)
    pt        = cipher.decrypt_and_verify(ct, tag)
    return pt.decode("utf-8")


# ── Record-level operations ──────────────────────────────────────────────────

def encrypt_patient_record(record: dict, key: bytes) -> dict:
    """
    Encrypt every value in a patient record dict.

    Numeric values are first converted to strings. The keys (field names)
    are NOT encrypted — only the values, because field names are not
    sensitive (they are the same 13 clinical feature names for every patient).

    Returns a new dict with the same keys but encrypted string values.
    """
    encrypted = {}
    for field_name, value in record.items():
        plaintext = json.dumps(value) if not isinstance(value, str) else value
        encrypted[field_name] = encrypt_field(plaintext, key)
    return encrypted


def decrypt_patient_record(encrypted_record: dict, key: bytes) -> dict:
    """
    Decrypt every value in an encrypted patient record dict.

    Attempts to JSON-parse decrypted values back to their original type
    (int, float, list, etc.).  Falls back to raw string if parsing fails.
    """
    decrypted = {}
    for field_name, token in encrypted_record.items():
        raw = decrypt_field(token, key)
        try:
            decrypted[field_name] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            decrypted[field_name] = raw
    return decrypted


def encrypt_json_blob(data: dict, key: bytes) -> str:
    """
    Serialise a dict to JSON and encrypt the entire blob as one ciphertext.
    Useful for storing complex nested structures (e.g. SHAP values).
    """
    plaintext = json.dumps(data)
    return encrypt_field(plaintext, key)


def decrypt_json_blob(token: str, key: bytes) -> dict:
    """
    Decrypt a JSON blob ciphertext back to a dict.
    """
    plaintext = decrypt_field(token, key)
    return json.loads(plaintext)
