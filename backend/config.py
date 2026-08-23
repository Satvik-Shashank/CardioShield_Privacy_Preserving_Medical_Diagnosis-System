"""
backend/config.py
─────────────────
Configuration and cryptographic key management for the CardioShield backend.
"""

import hashlib
import os
import secrets
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = BACKEND_DIR / "cardioshield.db"
SALT_PATH = BACKEND_DIR / ".salt"

# Model Artifacts
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
XTRAIN_PATH = ARTIFACT_DIR / "X_train.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

# Feature Names and Clinical Labels
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Biological Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar (>120 mg/dl)",
    "restecg": "Resting ECG Result",
    "thalach": "Maximum Heart Rate (bpm)",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (oldpeak)",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels (0-3)",
    "thal": "Thalassemia Type",
}

SHAP_ADVICE = {
    "age": (
        "Advanced age increases cardiovascular risk.",
        "Maintain routine annual cardiology evaluations, control blood pressure, and sustain daily physical activity.",
    ),
    "sex": (
        "Biological sex affects baseline risk profile.",
        "Males face statistically higher early onset risk. Monitor baseline lipid panels and arterial health.",
    ),
    "cp": (
        "Chest pain characteristics indicate cardiac stress.",
        "Typical angina symptoms warrant prompt formal cardiological diagnostic evaluation and stress testing.",
    ),
    "trestbps": (
        "Elevated resting blood pressure increases cardiac workload.",
        "Target BP < 120/80 mmHg through dietary sodium reduction, exercise, stress mitigation, and antihypertensive therapy if indicated.",
    ),
    "chol": (
        "Elevated serum cholesterol promotes atherosclerotic plaque.",
        "Target total cholesterol < 200 mg/dl and LDL < 100 mg/dl through dietary changes or statin therapy.",
    ),
    "fbs": (
        "Elevated fasting blood sugar indicates metabolic dysfunction.",
        "Maintain strict glycaemic control, routine HbA1c monitoring every 3 to 6 months, and active lifestyle.",
    ),
    "restecg": (
        "Resting ECG abnormalities suggest conduction or myocardial issues.",
        "Follow up with resting echocardiogram or 24-hour Holter monitoring to evaluate electrical stability.",
    ),
    "thalach": (
        "Sub-optimal peak exercise heart rate reflects lower cardiac reserve.",
        "Consider structured aerobic conditioning and supervised exercise testing under physician guidance.",
    ),
    "exang": (
        "Exercise-induced angina signals myocardial ischaemia under exertion.",
        "Avoid unmonitored strenuous exertion; arrange coronary evaluation (e.g. CT coronary angiography).",
    ),
    "oldpeak": (
        "ST-segment depression indicates reversible myocardial ischaemia.",
        "Significant ST depression warrants prompt clinical review and comprehensive ischaemia workup.",
    ),
    "slope": (
        "Flat or downsloping ST segment is a recognized ischaemic marker.",
        "Correlate with stress imaging or angiography to assess multi-vessel disease.",
    ),
    "ca": (
        "Fluoroscopic vessel calcification or blockage increases risk.",
        "Specialist cardiology consultation (PCI/CABG evaluation) is advised for significant vessel involvement.",
    ),
    "thal": (
        "Thalassemia / myocardial perfusion defects indicate tissue compromise.",
        "Differentiate reversible vs fixed perfusion defects with nuclear imaging or cardiac MRI.",
    ),
}

# Settings
class Settings(BaseSettings):
    app_name: str = "CardioShield Backend"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = ["*"]
    api_key: str = os.getenv("CARDIOSHIELD_API_KEY", "cardioshield-secret-dev-key")
    secret_key: str = os.getenv("CARDIOSHIELD_SECRET_KEY", "cardioshield-default-encryption-passphrase")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()


def _get_or_create_salt() -> bytes:
    """Return persistent 32-byte salt for key derivation."""
    if SALT_PATH.exists():
        return SALT_PATH.read_bytes()
    salt = secrets.token_bytes(32)
    SALT_PATH.write_bytes(salt)
    return salt


def derive_aes_key() -> bytes:
    """Derive 256-bit AES key using PBKDF2-HMAC-SHA256 with 100k iterations."""
    passphrase = settings.secret_key
    salt = _get_or_create_salt()
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations=100_000,
        dklen=32,
    )
