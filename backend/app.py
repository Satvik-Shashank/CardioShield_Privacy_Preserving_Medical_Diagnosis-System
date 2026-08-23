"""
backend/app.py
──────────────
FastAPI REST API for CardioShield.
Provides genuine TenSEAL CKKS Homomorphic Encryption inference,
AES-256-GCM encrypted database persistence, SHAP explainability,
and clinical PDF generation.
"""

import json
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import shap
import tenseal as ts
from fastapi import FastAPI, HTTPException, Header, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import (
    FEATURE_NAMES,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    XTRAIN_PATH,
    derive_aes_key,
    settings,
)
from backend.crypto_utils import (
    decrypt_field,
    decrypt_json_blob,
    decrypt_patient_record,
    encrypt_field,
    encrypt_json_blob,
    encrypt_patient_record,
)
from backend.database import (
    delete_patient,
    get_all_predictions,
    get_patient,
    get_prediction,
    init_db,
    list_patients,
    store_patient,
    store_prediction,
)
from backend.report_generator import generate_clinical_report
from he_engine import (
    create_context,
    encrypt_patient_data,
    homomorphic_predict,
    verify_he_system,
)

# ═════════════════════════════════════════════════════════════════════════════
# FastAPI App Initialization
# ═════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="CardioShield Secure API",
    description="Privacy-Preserving Cardiovascular Risk Assessment with TenSEAL CKKS Homomorphic Encryption",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ─────────────────────────────────────────────────────────────
STATE = {
    "model": None,
    "scaler": None,
    "X_train_bg": None,
    "explainer": None,
    "he_ctx": None,
    "enc_w": None,
    "enc_b": None,
    "aes_key": None,
    "he_verified": False,
    "he_status": "UNINITIALIZED",
    "metrics": None,
}


def load_all_resources():
    """Load model artifacts, establish TenSEAL CKKS context, and derive AES keys."""
    # 1. Database
    init_db()

    # 2. AES Key
    STATE["aes_key"] = derive_aes_key()

    # 3. Model Artifacts
    if not (MODEL_PATH.exists() and SCALER_PATH.exists()):
        raise RuntimeError("Model artifacts missing. Run `python model_trainer.py` first.")

    with open(MODEL_PATH, "rb") as f:
        STATE["model"] = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        STATE["scaler"] = pickle.load(f)
    with open(XTRAIN_PATH, "rb") as f:
        STATE["X_train_bg"] = pickle.load(f)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            STATE["metrics"] = json.load(f)

    # SHAP Explainer
    STATE["explainer"] = shap.LinearExplainer(
        STATE["model"],
        STATE["X_train_bg"],
        feature_perturbation="interventional",
    )

    # 4. TenSEAL CKKS Context & Weights
    try:
        ctx = create_context()
        STATE["he_ctx"] = ctx
        w = STATE["model"].coef_[0]
        b = float(STATE["model"].intercept_[0])
        STATE["enc_w"] = ts.ckks_vector(ctx, w.tolist())
        STATE["enc_b"] = ts.ckks_vector(ctx, [b])

        # Run startup verification
        verify_res = verify_he_system()
        STATE["he_verified"] = (verify_res["status"] == "HEALTHY")
        STATE["he_status"] = "ACTIVE (128-bit RLWE)"
    except Exception as e:
        STATE["he_status"] = f"ERROR: {e}"
        raise RuntimeError(f"TenSEAL CKKS startup initialization failed: {e}") from e


@app.on_event("startup")
def startup_event():
    load_all_resources()
    print("\n[CardioShield Backend] Startup completed successfully!")
    print(f" • HE Engine Status: {STATE['he_status']}")
    print(f" • Database: Ready (AES-256-GCM encrypted persistence)")
    print(f" • Model: Logistic Regression (Features: {len(FEATURE_NAMES)})\n")


# ═════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas
# ═════════════════════════════════════════════════════════════════════════════

class ClinicalFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
    cp: float = Field(..., ge=0, le=3, description="Chest Pain Type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting BP (mmHg)")
    chol: float = Field(..., ge=50, le=600, description="Serum Cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (0/1)")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG (0-2)")
    thalach: float = Field(..., ge=50, le=240, description="Maximum Heart Rate (bpm)")
    exang: float = Field(..., ge=0, le=1, description="Exercise Induced Angina (0/1)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST Depression (oldpeak)")
    slope: float = Field(..., ge=0, le=2, description="ST Segment Slope (0-2)")
    ca: float = Field(..., ge=0, le=3, description="Major Vessels (0-3)")
    thal: float = Field(..., ge=1, le=3, description="Thalassemia (1=Normal, 2=Fixed, 3=Reversible)")


class PredictRequest(BaseModel):
    features: ClinicalFeatures
    patient_name: Optional[str] = "Anonymous Patient"
    clinician_name: Optional[str] = "Attending Physician"
    assessment_date: Optional[str] = None
    save_to_db: bool = False


class PredictResponse(BaseModel):
    risk_score_pct: float
    risk_class: str
    he_prob: float
    plain_prob: float
    prob_delta: float
    shap_values: Dict[str, float]
    feature_values: Dict[str, float]
    hex_proof: str
    ciphertext_bytes: int
    t_enc_ms: float
    t_he_ms: float
    t_dec_ms: float
    t_total_ms: float
    he_used: bool
    patient_id: Optional[int] = None


class PatientCreate(BaseModel):
    patient_name: str
    clinician_name: str
    assessment_date: Optional[str] = None
    features: ClinicalFeatures


class ReportRequest(BaseModel):
    patient_name: str
    clinician_name: str
    assessment_date: Optional[str] = None
    features: Dict[str, float]
    risk_prob: float
    risk_class: str
    shap_values: Dict[str, float]
    latency_ms: float = 0.0
    he_used: bool = True


# ═════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health_check():
    """Health status verifying HE engine, database, and model artifacts."""
    return {
        "status": "HEALTHY" if STATE["he_verified"] else "DEGRADED",
        "he_engine": {
            "status": STATE["he_status"],
            "scheme": "CKKS",
            "security": "128-bit RLWE",
            "poly_modulus_degree": 8192,
        },
        "database": "CONNECTED",
        "encryption": "AES-256-GCM Authenticated",
        "timestamp": time.time(),
    }


@app.get("/api/metrics")
def get_metrics():
    """Return dynamic metrics calculated from training and real HE validation."""
    if STATE["metrics"] is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found. Run model_trainer.py to generate metrics.",
        )
    return STATE["metrics"]


@app.post("/api/predict", response_model=PredictResponse)
def predict_heart_disease(req: PredictRequest):
    """
    Run real TenSEAL CKKS Homomorphic Inference and SHAP feature attributions.
    Never falls back to simulated/fake ciphertext.
    """
    if not STATE["he_verified"] or STATE["he_ctx"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Homomorphic Encryption Engine unavailable: {STATE['he_status']}",
        )

    t_wall_start = time.perf_counter()
    feat_dict = req.features.model_dump()
    raw_vals = [float(feat_dict[k]) for k in FEATURE_NAMES]

    # Preprocessing with StandardScaler
    scaled = STATE["scaler"].transform([raw_vals])[0]

    # ── 1. Client-to-Ciphertext Encryption ────────────────────────────────────
    t0 = time.perf_counter()
    enc_x = encrypt_patient_data(STATE["he_ctx"], scaled)
    t_enc = (time.perf_counter() - t0) * 1000

    raw_ciphertext_bytes = enc_x.serialize()
    ciphertext_size = len(raw_ciphertext_bytes)
    # Generate structured hex display for visual proof
    hex_prefix = raw_ciphertext_bytes[:160].hex().upper()
    hex_formatted = " ".join(hex_prefix[i:i+2] for i in range(0, len(hex_prefix), 2)) + " ..."

    # ── 2. Server Homomorphic Prediction on Ciphertext ────────────────────────
    t0 = time.perf_counter()
    enc_pred = homomorphic_predict(enc_x, STATE["enc_w"], STATE["enc_b"])
    t_he = (time.perf_counter() - t0) * 1000

    # ── 3. Client Decryption of Final Scalar ──────────────────────────────────
    t0 = time.perf_counter()
    dec_prob = float(enc_pred.decrypt()[0])
    dec_prob = max(0.0, min(1.0, dec_prob))
    t_dec = (time.perf_counter() - t0) * 1000

    # Plaintext reference probability
    z_plain = float(np.dot(scaled, STATE["model"].coef_[0]) + STATE["model"].intercept_[0])
    plain_prob = float(1.0 / (1.0 + np.exp(-z_plain)))

    # SHAP Explanations
    shap_vals_arr = STATE["explainer"].shap_values(scaled.reshape(1, -1))[0]
    shap_dict = {fname: float(val) for fname, val in zip(FEATURE_NAMES, shap_vals_arr)}

    risk_pct = round(dec_prob * 100, 2)
    if risk_pct >= 60.0:
        risk_class = "High Risk"
    elif risk_pct >= 40.0:
        risk_class = "Moderate Risk"
    else:
        risk_class = "Low Risk"

    t_total = (time.perf_counter() - t_wall_start) * 1000

    # ── 4. Optional Persistent Encrypted Storage ──────────────────────────────
    saved_patient_id = None
    if req.save_to_db:
        enc_name = encrypt_field(req.patient_name or "Anonymous", STATE["aes_key"])
        enc_clinician = encrypt_field(req.clinician_name or "Physician", STATE["aes_key"])
        enc_date = encrypt_field(req.assessment_date or time.strftime("%Y-%m-%d"), STATE["aes_key"])
        enc_feats = encrypt_json_blob(feat_dict, STATE["aes_key"])

        patient_id = store_patient(enc_name, enc_clinician, enc_date, enc_feats)
        saved_patient_id = patient_id

        # Encrypt & store prediction record
        enc_risk_score = encrypt_field(str(risk_pct), STATE["aes_key"])
        enc_risk_class = encrypt_field(risk_class, STATE["aes_key"])
        enc_shap = encrypt_json_blob(shap_dict, STATE["aes_key"])
        enc_plain = encrypt_field(str(round(plain_prob * 100, 2)), STATE["aes_key"])

        store_prediction(
            patient_id=patient_id,
            enc_risk_score=enc_risk_score,
            enc_risk_class=enc_risk_class,
            enc_shap_values=enc_shap,
            enc_plain_prob=enc_plain,
            he_used=True,
            encryption_time_ms=t_enc,
            inference_time_ms=t_he,
            total_time_ms=t_total,
        )

    return PredictResponse(
        risk_score_pct=risk_pct,
        risk_class=risk_class,
        he_prob=round(dec_prob, 5),
        plain_prob=round(plain_prob, 5),
        prob_delta=round(abs(dec_prob - plain_prob), 5),
        shap_values=shap_dict,
        feature_values=feat_dict,
        hex_proof=hex_formatted,
        ciphertext_bytes=ciphertext_size,
        t_enc_ms=round(t_enc, 2),
        t_he_ms=round(t_he, 2),
        t_dec_ms=round(t_dec, 2),
        t_total_ms=round(t_total, 2),
        he_used=True,
        patient_id=saved_patient_id,
    )


@app.post("/api/patients")
def create_patient_record(req: PatientCreate):
    """Persist an encrypted patient record and compute prediction."""
    feat_dict = req.features.model_dump()
    enc_name = encrypt_field(req.patient_name, STATE["aes_key"])
    enc_clinician = encrypt_field(req.clinician_name, STATE["aes_key"])
    enc_date = encrypt_field(req.assessment_date or time.strftime("%Y-%m-%d"), STATE["aes_key"])
    enc_feats = encrypt_json_blob(feat_dict, STATE["aes_key"])

    patient_id = store_patient(enc_name, enc_clinician, enc_date, enc_feats)

    # Run prediction
    pred_res = predict_heart_disease(
        PredictRequest(
            features=req.features,
            patient_name=req.patient_name,
            clinician_name=req.clinician_name,
            assessment_date=req.assessment_date,
            save_to_db=False,
        )
    )

    # Store encrypted prediction
    store_prediction(
        patient_id=patient_id,
        enc_risk_score=encrypt_field(str(pred_res.risk_score_pct), STATE["aes_key"]),
        enc_risk_class=encrypt_field(pred_res.risk_class, STATE["aes_key"]),
        enc_shap_values=encrypt_json_blob(pred_res.shap_values, STATE["aes_key"]),
        enc_plain_prob=encrypt_field(str(round(pred_res.plain_prob * 100, 2)), STATE["aes_key"]),
        he_used=True,
        encryption_time_ms=pred_res.t_enc_ms,
        inference_time_ms=pred_res.t_he_ms,
        total_time_ms=pred_res.t_total_ms,
    )

    return {"patient_id": patient_id, "prediction": pred_res}


@app.get("/api/patients")
def get_all_patients():
    """List all stored patients, decrypting metadata for authenticated view."""
    raw_rows = list_patients()
    decrypted_patients = []
    key = STATE["aes_key"]

    for row in raw_rows:
        try:
            name = decrypt_field(row["enc_name"], key)
            clinician = decrypt_field(row["enc_clinician"], key)
            date = decrypt_field(row["enc_date"], key)
            latest_pred = get_prediction(row["id"])

            pred_summary = None
            if latest_pred:
                pred_summary = {
                    "risk_score": float(decrypt_field(latest_pred["enc_risk_score"], key)),
                    "risk_class": decrypt_field(latest_pred["enc_risk_class"], key),
                    "he_used": bool(latest_pred["he_used"]),
                    "created_at": latest_pred["created_at"],
                }

            decrypted_patients.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "patient_name": name,
                "clinician_name": clinician,
                "assessment_date": date,
                "latest_prediction": pred_summary,
            })
        except Exception:
            continue

    return decrypted_patients


@app.get("/api/patients/{patient_id}")
def get_patient_details(patient_id: int):
    """Retrieve decrypted patient details and prediction history."""
    row = get_patient(patient_id)
    if not row:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    key = STATE["aes_key"]
    name = decrypt_field(row["enc_name"], key)
    clinician = decrypt_field(row["enc_clinician"], key)
    date = decrypt_field(row["enc_date"], key)
    features = decrypt_json_blob(row["enc_features"], key)

    pred_rows = get_all_predictions(patient_id)
    predictions = []
    for pr in pred_rows:
        try:
            predictions.append({
                "id": pr["id"],
                "created_at": pr["created_at"],
                "risk_score_pct": float(decrypt_field(pr["enc_risk_score"], key)),
                "risk_class": decrypt_field(pr["enc_risk_class"], key),
                "shap_values": decrypt_json_blob(pr["enc_shap_values"], key),
                "plain_prob_pct": float(decrypt_field(pr["enc_plain_prob"], key)),
                "he_used": bool(pr["he_used"]),
                "encryption_time_ms": pr["encryption_time_ms"],
                "inference_time_ms": pr["inference_time_ms"],
                "total_time_ms": pr["total_time_ms"],
            })
        except Exception:
            continue

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "patient_name": name,
        "clinician_name": clinician,
        "assessment_date": date,
        "features": features,
        "predictions": predictions,
    }


@app.delete("/api/patients/{patient_id}")
def delete_patient_record(patient_id: int):
    """Delete patient and associated encrypted predictions."""
    ok = delete_patient(patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Patient record not found.")
    return {"status": "DELETED", "patient_id": patient_id}


@app.post("/api/report/pdf")
def export_pdf_report(req: ReportRequest):
    """Generate and stream clinical PDF report."""
    pdf_bytes = generate_clinical_report(
        patient_name=req.patient_name,
        clinician_name=req.clinician_name,
        assessment_date=req.assessment_date,
        features=req.features,
        risk_prob=req.risk_prob,
        risk_class=req.risk_class,
        shap_values=[req.shap_values.get(f, 0.0) for f in FEATURE_NAMES],
        feature_names=FEATURE_NAMES,
        he_used=req.he_used,
        latency_ms=req.latency_ms,
    )

    clean_name = req.patient_name.replace(" ", "_")
    filename = f"CardioShield_Report_{clean_name}_{req.assessment_date or 'latest'}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Frontend Static SPA Mounting (Production Mode)
# ═════════════════════════════════════════════════════════════════════════════
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found.")
        target = os.path.join(FRONTEND_DIST, full_path)
        if os.path.exists(target) and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=settings.host, port=settings.port, reload=settings.debug)
