"""
config.py
─────────
Configuration management for CardioShield backend.

The AES-256 master key is derived from an environment variable using PBKDF2
with a stored salt. This ensures:
  - The key is never hardcoded
  - The same passphrase always produces the same key (deterministic with salt)
  - Key derivation is computationally expensive (100k iterations) to resist brute-force
"""

import os
import hashlib
import secrets
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH     = BACKEND_DIR / "cardioshield.db"
SALT_PATH   = BACKEND_DIR / ".salt"

# ── Artefacts (model, scaler, X_train) ───────────────────────────────────────
ARTEFACT_DIR = BASE_DIR / "artefacts"
MODEL_PATH   = ARTEFACT_DIR / "model.pkl"
SCALER_PATH  = ARTEFACT_DIR / "scaler.pkl"
XTRAIN_PATH  = ARTEFACT_DIR / "X_train.pkl"

# ── Feature metadata ────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ── Flask settings ───────────────────────────────────────────────────────────
DEBUG       = os.getenv("CARDIOSHIELD_DEBUG", "false").lower() == "true"
HOST        = os.getenv("CARDIOSHIELD_HOST", "0.0.0.0")
PORT        = int(os.getenv("CARDIOSHIELD_PORT", "5000"))
CORS_ORIGIN = os.getenv("CARDIOSHIELD_CORS_ORIGIN", "*")

# ── API Key Authentication ───────────────────────────────────────────────────
# If not set, auto-generate one and print it at startup (dev convenience)
_api_key_env = os.getenv("CARDIOSHIELD_API_KEY")
if _api_key_env:
    API_KEY = _api_key_env
else:
    API_KEY = secrets.token_urlsafe(32)
    # Will be printed at startup so the user knows it


def _get_or_create_salt() -> bytes:
    """
    Return a persistent 32-byte salt.  Created once, stored on disk so that
    the same passphrase always derives the same AES key across restarts.
    """
    if SALT_PATH.exists():
        return SALT_PATH.read_bytes()
    salt = secrets.token_bytes(32)
    SALT_PATH.write_bytes(salt)
    return salt


def derive_aes_key() -> bytes:
    """
    Derive a 256-bit AES key from the CARDIOSHIELD_SECRET_KEY env var
    using PBKDF2-HMAC-SHA256 with 100 000 iterations.

    If the env var is not set, a random key is generated (suitable for
    development / demo, but data will NOT survive restarts).
    """
    passphrase = os.getenv("CARDIOSHIELD_SECRET_KEY")

    if not passphrase:
        # Dev mode — generate a random key and warn
        import warnings
        warnings.warn(
            "CARDIOSHIELD_SECRET_KEY not set — using a random key. "
            "Encrypted data will NOT survive restarts!",
            RuntimeWarning,
            stacklevel=2,
        )
        # Still derive from a random passphrase so the rest of the flow is identical
        passphrase = secrets.token_hex(32)

    salt = _get_or_create_salt()
    key  = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations=100_000,
        dklen=32,
    )
    return key
