"""
test_backend.py
───────────────
Automated test suite for the CardioShield backend.

Tests cover:
  1. AES-256-GCM encryption/decryption round-trip
  2. Tamper detection (authenticated encryption)
  3. Database encrypted CRUD
  4. API endpoints (Flask test client)
  5. Verification that raw data NEVER appears in the DB

Run:
    python -m pytest backend/test_backend.py -v
"""

import json
import os
import sqlite3
import sys
import tempfile
import time

import pytest

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.config import derive_aes_key, FEATURE_NAMES
from backend.crypto_utils import (
    encrypt_field,
    decrypt_field,
    encrypt_patient_record,
    decrypt_patient_record,
    encrypt_json_blob,
    decrypt_json_blob,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def aes_key():
    """Derive a test AES key."""
    os.environ["CARDIOSHIELD_SECRET_KEY"] = "test-secret-key-for-pytest"
    return derive_aes_key()


@pytest.fixture
def sample_patient():
    """Sample patient data matching the API schema."""
    return {
        "patient_name":    "John Doe",
        "clinician_name":  "Dr. Smith",
        "assessment_date": "2025-01-15",
        "features": {
            "age": 54, "sex": 1, "cp": 0, "trestbps": 130,
            "chol": 246, "fbs": 0, "restecg": 1, "thalach": 150,
            "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
        },
    }


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Flask test client with a temp database and API key."""
    # Use temp directory for DB and salt
    monkeypatch.setattr("backend.config.DB_PATH",  tmp_path / "test.db")
    monkeypatch.setattr("backend.config.SALT_PATH", tmp_path / ".salt")
    monkeypatch.setattr("backend.database.DB_PATH", tmp_path / "test.db")
    os.environ["CARDIOSHIELD_SECRET_KEY"] = "test-secret-key-for-pytest"
    os.environ["CARDIOSHIELD_API_KEY"] = "test-api-key"
    monkeypatch.setattr("backend.config.API_KEY", "test-api-key")

    from backend.app import create_app
    app = create_app(testing=True)

    class AuthClient:
        """Wraps Flask test client to auto-include X-API-Key header."""
        def __init__(self, client):
            self._client = client

        def get(self, *args, **kwargs):
            kwargs.setdefault("headers", {})["X-API-Key"] = "test-api-key"
            return self._client.get(*args, **kwargs)

        def post(self, *args, **kwargs):
            kwargs.setdefault("headers", {})["X-API-Key"] = "test-api-key"
            return self._client.post(*args, **kwargs)

        def delete(self, *args, **kwargs):
            kwargs.setdefault("headers", {})["X-API-Key"] = "test-api-key"
            return self._client.delete(*args, **kwargs)

    with app.test_client() as client:
        yield AuthClient(client)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Crypto round-trip tests
# ═════════════════════════════════════════════════════════════════════════════

class TestCrypto:
    def test_encrypt_decrypt_string(self, aes_key):
        """Verify that encrypting and decrypting returns the original string."""
        original = "John Doe"
        token    = encrypt_field(original, aes_key)
        assert token != original  # Must be encrypted
        assert decrypt_field(token, aes_key) == original

    def test_encrypt_decrypt_number(self, aes_key):
        """Verify numeric values survive round-trip."""
        original = "54.321"
        token    = encrypt_field(original, aes_key)
        assert decrypt_field(token, aes_key) == original

    def test_encrypt_decrypt_empty_string(self, aes_key):
        """Edge case: empty string."""
        token = encrypt_field("", aes_key)
        assert decrypt_field(token, aes_key) == ""

    def test_encrypt_decrypt_unicode(self, aes_key):
        """Unicode characters must survive round-trip."""
        original = "Patiënt Müller — σ(z) ≈ 0.5"
        token    = encrypt_field(original, aes_key)
        assert decrypt_field(token, aes_key) == original

    def test_unique_nonces(self, aes_key):
        """Each encryption must produce a different ciphertext (unique nonce)."""
        tokens = [encrypt_field("same input", aes_key) for _ in range(10)]
        assert len(set(tokens)) == 10  # All different

    def test_tamper_detection(self, aes_key):
        """Modifying any byte of the ciphertext must raise ValueError."""
        import base64
        token   = encrypt_field("secret data", aes_key)
        raw     = bytearray(base64.b64decode(token))
        raw[15] ^= 0xFF  # Tamper with a ciphertext byte
        tampered = base64.b64encode(bytes(raw)).decode("ascii")
        with pytest.raises(Exception):  # ValueError or MAC check failed
            decrypt_field(tampered, aes_key)

    def test_wrong_key_fails(self, aes_key):
        """Decrypting with a different key must fail."""
        import hashlib
        wrong_key = hashlib.sha256(b"wrong-key").digest()
        token     = encrypt_field("secret data", aes_key)
        with pytest.raises(Exception):
            decrypt_field(token, wrong_key)


class TestRecordEncryption:
    def test_patient_record_round_trip(self, aes_key):
        """Full patient record encrypt/decrypt round-trip."""
        record = {
            "name":  "Jane Doe",
            "age":   42,
            "chol":  200.5,
            "items": [1, 2, 3],
        }
        encrypted = encrypt_patient_record(record, aes_key)

        # All values must be encrypted strings
        for key, val in encrypted.items():
            assert isinstance(val, str)
            assert val != str(record[key])

        decrypted = decrypt_patient_record(encrypted, aes_key)
        assert decrypted["name"] == "Jane Doe"
        assert decrypted["age"] == 42
        assert decrypted["chol"] == 200.5
        assert decrypted["items"] == [1, 2, 3]

    def test_json_blob_round_trip(self, aes_key):
        """Test encrypt/decrypt of complex JSON blob."""
        data = {
            "shap_values": {"age": 0.123, "chol": -0.456},
            "metadata":    {"timestamp": "2025-01-15T12:00:00"},
        }
        token     = encrypt_json_blob(data, aes_key)
        recovered = decrypt_json_blob(token, aes_key)
        assert recovered == data


# ═════════════════════════════════════════════════════════════════════════════
# 2. API endpoint tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAPI:
    def test_health(self, app_client):
        """Health endpoint returns 200."""
        resp = app_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["encryption"] == "AES-256-GCM"

    def test_create_patient(self, app_client, sample_patient):
        """POST /api/patients creates a patient and returns prediction."""
        resp = app_client.post(
            "/api/patients",
            data=json.dumps(sample_patient),
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert "patient_id" in data
        assert "prediction" in data
        assert data["prediction"]["risk_class"] in [
            "HIGH RISK", "MODERATE RISK", "LOW RISK"
        ]
        assert 0 <= data["prediction"]["risk_score"] <= 1

    def test_create_patient_missing_features(self, app_client):
        """Should return 400 when features are missing."""
        resp = app_client.post(
            "/api/patients",
            data=json.dumps({"patient_name": "X", "clinician_name": "Y", "features": {"age": 50}}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "Missing features" in resp.get_json()["error"]

    def test_list_patients(self, app_client, sample_patient):
        """GET /api/patients returns all stored patients."""
        # Create two patients
        app_client.post("/api/patients", data=json.dumps(sample_patient), content_type="application/json")
        app_client.post("/api/patients", data=json.dumps(sample_patient), content_type="application/json")

        resp = app_client.get("/api/patients")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 2
        # Verify decrypted names
        assert data["patients"][0]["name"] == "John Doe"

    def test_get_patient(self, app_client, sample_patient):
        """GET /api/patients/<id> returns decrypted patient data."""
        create_resp = app_client.post(
            "/api/patients",
            data=json.dumps(sample_patient),
            content_type="application/json",
        )
        pid = create_resp.get_json()["patient_id"]

        resp = app_client.get(f"/api/patients/{pid}")
        assert resp.status_code == 200
        patient = resp.get_json()["patient"]
        assert patient["name"] == "John Doe"
        assert patient["features"]["age"] == 54
        assert patient["features"]["chol"] == 246

    def test_get_patient_encrypted(self, app_client, sample_patient):
        """GET /api/patients/<id>?encrypted=true returns raw ciphertext."""
        create_resp = app_client.post(
            "/api/patients",
            data=json.dumps(sample_patient),
            content_type="application/json",
        )
        pid = create_resp.get_json()["patient_id"]

        resp = app_client.get(f"/api/patients/{pid}?encrypted=true")
        assert resp.status_code == 200
        patient = resp.get_json()["patient"]
        # Encrypted columns should be present and NOT readable
        assert "enc_name" in patient
        assert patient["enc_name"] != "John Doe"

    def test_get_prediction(self, app_client, sample_patient):
        """GET /api/patients/<id>/prediction returns decrypted prediction."""
        create_resp = app_client.post(
            "/api/patients",
            data=json.dumps(sample_patient),
            content_type="application/json",
        )
        pid = create_resp.get_json()["patient_id"]

        resp = app_client.get(f"/api/patients/{pid}/prediction")
        assert resp.status_code == 200
        prediction = resp.get_json()["prediction"]
        assert "risk_score" in prediction
        assert "risk_class" in prediction
        assert "shap_values" in prediction

    def test_delete_patient(self, app_client, sample_patient):
        """DELETE /api/patients/<id> removes patient and predictions."""
        create_resp = app_client.post(
            "/api/patients",
            data=json.dumps(sample_patient),
            content_type="application/json",
        )
        pid = create_resp.get_json()["patient_id"]

        # Delete
        resp = app_client.delete(f"/api/patients/{pid}")
        assert resp.status_code == 200

        # Verify gone
        resp = app_client.get(f"/api/patients/{pid}")
        assert resp.status_code == 404

    def test_predict_only(self, app_client, sample_patient):
        """POST /api/predict returns prediction without storing."""
        resp = app_client.post(
            "/api/predict",
            data=json.dumps({"features": sample_patient["features"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["stored"] is False
        assert "prediction" in data

    def test_patient_not_found(self, app_client):
        """GET non-existent patient returns 404."""
        resp = app_client.get("/api/patients/9999")
        assert resp.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# 3. Database privacy verification
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabasePrivacy:
    def test_raw_data_never_in_db(self, app_client, sample_patient, tmp_path):
        """
        CRITICAL TEST: After storing a patient, inspect the raw SQLite file
        and verify that none of the plaintext values appear anywhere.
        """
        app_client.post(
            "/api/patients",
            data=json.dumps(sample_patient),
            content_type="application/json",
        )

        # Read the raw database file as bytes
        db_path = tmp_path / "test.db"
        raw_bytes = db_path.read_bytes()
        raw_text  = raw_bytes.decode("latin-1")  # safe for binary

        # These plaintext values must NEVER appear in the DB
        forbidden = [
            "John Doe",
            "Dr. Smith",
            "2025-01-15",
            # Check that numeric feature values don't appear as strings
        ]
        for text in forbidden:
            assert text not in raw_text, (
                f"PRIVACY VIOLATION: '{text}' found in raw database!"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
