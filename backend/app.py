"""
app.py  –  CardioShield Backend API
────────────────────────────────────
Flask REST API where ALL patient data is encrypted (AES-256-GCM) before
touching the database.  Raw data exists ONLY in memory during a request
and is NEVER persisted.

Endpoints:
    GET  /api/health                    – health check
    POST /api/patients                  – encrypt & store patient + run prediction
    GET  /api/patients                  – list all patients (encrypted metadata + IDs)
    GET  /api/patients/<id>             – retrieve & decrypt a single patient
    GET  /api/patients/<id>/prediction  – retrieve decrypted prediction
    DELETE /api/patients/<id>           – delete patient (cascade)
    POST /api/predict                   – run prediction only (no storage)

Run:
    python -m backend.app
"""

import json
import os
import pickle
import sys
import time
import traceback

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Ensure project root is on sys.path so he_engine can be imported ──────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functools import wraps

from backend.config import (
    API_KEY,
    ARTEFACT_DIR,
    CORS_ORIGIN,
    DB_PATH,
    DEBUG,
    FEATURE_NAMES,
    HOST,
    MODEL_PATH,
    PORT,
    SCALER_PATH,
    XTRAIN_PATH,
    derive_aes_key,
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


# ═════════════════════════════════════════════════════════════════════════════
# App factory
# ═════════════════════════════════════════════════════════════════════════════

def create_app(testing: bool = False) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["TESTING"] = testing
    CORS(app, origins=CORS_ORIGIN)

    # ── API Key Authentication ───────────────────────────────────────────
    def require_api_key(f):
        """Decorator: reject requests without a valid X-API-Key header."""
        @wraps(f)
        def decorated(*args, **kwargs):
            key = request.headers.get("X-API-Key", "")
            if key != API_KEY:
                return jsonify({
                    "error": "Unauthorized — provide a valid X-API-Key header."
                }), 401
            return f(*args, **kwargs)
        return decorated

    # ── Derive encryption key ────────────────────────────────────────────
    aes_key = derive_aes_key()

    # ── Init database ────────────────────────────────────────────────────
    init_db()

    # ── Load ML artefacts ────────────────────────────────────────────────
    model = scaler = X_train_bg = None

    def _load_artefacts():
        nonlocal model, scaler, X_train_bg
        if model is not None:
            return True
        try:
            with open(str(MODEL_PATH),  "rb") as f:
                model = pickle.load(f)
            with open(str(SCALER_PATH), "rb") as f:
                scaler = pickle.load(f)
            with open(str(XTRAIN_PATH), "rb") as f:
                X_train_bg = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

    # ── HE context (lazy) ────────────────────────────────────────────────
    _he_ctx = {"ctx": None}

    def _get_he_context():
        if _he_ctx["ctx"] is None:
            try:
                from he_engine import create_context
                _he_ctx["ctx"] = create_context()
            except Exception:
                _he_ctx["ctx"] = None
        return _he_ctx["ctx"]

    # ── Prediction helpers ───────────────────────────────────────────────
    def _run_prediction(raw_values: list[float]):
        """
        Run the full prediction pipeline:
          1. Scale features
          2. Try HE inference (TenSEAL CKKS)
          3. Fallback to plaintext if HE unavailable
          4. Compute SHAP values
          5. Return all results

        Raw data is ONLY in memory here — caller encrypts before storage.
        """
        if not _load_artefacts():
            raise RuntimeError(
                "Model artefacts not found. Run `python model_trainer.py` first."
            )

        scaled = scaler.transform([raw_values])[0]
        t_wall = time.perf_counter()
        he_ok  = False
        t_enc = t_he = 0.0
        he_prob = 0.0

        try:
            import tenseal as ts
            from he_engine import create_context, encrypt_patient_data, homomorphic_predict

            ctx = _get_he_context()
            if ctx is None:
                raise RuntimeError("CKKS context failed")

            t0    = time.perf_counter()
            enc_x = encrypt_patient_data(ctx, scaled)
            t_enc = time.perf_counter() - t0

            w, b  = model.coef_[0], model.intercept_[0]
            enc_w = ts.ckks_vector(ctx, w.tolist())
            enc_b = ts.ckks_vector(ctx, [float(b)])

            t0       = time.perf_counter()
            enc_pred = homomorphic_predict(enc_x, enc_w, enc_b)
            t_he     = time.perf_counter() - t0

            he_prob = float(enc_pred.decrypt()[0])
            he_prob = max(0.0, min(1.0, he_prob))
            he_ok   = True

        except Exception:
            z       = float(np.dot(scaled, model.coef_[0]) + model.intercept_[0])
            he_prob = float(1.0 / (1.0 + np.exp(-z)))

        # SHAP values
        shap_vals = np.zeros(13).tolist()
        try:
            import shap
            explainer = shap.LinearExplainer(
                model, X_train_bg, feature_perturbation="interventional"
            )
            shap_vals = explainer.shap_values(scaled.reshape(1, -1))[0].tolist()
        except Exception:
            pass

        # Plaintext reference probability
        z_plain    = float(np.dot(scaled, model.coef_[0]) + model.intercept_[0])
        plain_prob = float(1.0 / (1.0 + np.exp(-z_plain)))
        t_total    = time.perf_counter() - t_wall

        risk_pct = he_prob * 100
        if risk_pct >= 60:
            risk_class = "HIGH RISK"
        elif risk_pct >= 40:
            risk_class = "MODERATE RISK"
        else:
            risk_class = "LOW RISK"

        return {
            "risk_score":         he_prob,
            "risk_percentage":    round(risk_pct, 2),
            "risk_class":         risk_class,
            "plain_prob":         plain_prob,
            "shap_values":        dict(zip(FEATURE_NAMES, shap_vals)),
            "he_used":            he_ok,
            "encryption_time_ms": round(t_enc * 1000, 2),
            "inference_time_ms":  round(t_he * 1000, 2),
            "total_time_ms":      round(t_total * 1000, 2),
        }

    # ═════════════════════════════════════════════════════════════════════
    # ROUTES
    # ═════════════════════════════════════════════════════════════════════

    # ── Health check ─────────────────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health():
        artefacts_ok = _load_artefacts()
        return jsonify({
            "status":       "healthy",
            "artefacts":    artefacts_ok,
            "database":     DB_PATH.exists(),
            "encryption":   "AES-256-GCM",
            "he_available": _get_he_context() is not None,
        })

    # ── POST /api/patients  —  encrypt + store + predict ─────────────────
    @app.route("/api/patients", methods=["POST"])
    @require_api_key
    def create_patient():
        """
        Accept raw patient data, encrypt ALL fields, store encrypted-only,
        run prediction, encrypt results, store, return decrypted prediction.

        Expected JSON body:
        {
            "patient_name":   "John Doe",
            "clinician_name": "Dr. Smith",
            "assessment_date": "2025-01-15",
            "features": {
                "age": 54, "sex": 1, "cp": 0, "trestbps": 130,
                "chol": 246, "fbs": 0, "restecg": 1, "thalach": 150,
                "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
            }
        }
        """
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Request body required"}), 400

            # Validate required fields
            required = ["patient_name", "clinician_name", "features"]
            missing  = [f for f in required if f not in data]
            if missing:
                return jsonify({"error": f"Missing fields: {missing}"}), 400

            features = data["features"]
            missing_feats = [f for f in FEATURE_NAMES if f not in features]
            if missing_feats:
                return jsonify({
                    "error": f"Missing features: {missing_feats}"
                }), 400

            # Build raw values list in canonical order
            raw_values = [float(features[f]) for f in FEATURE_NAMES]

            # ─── ENCRYPT patient metadata ────────────────────────────────
            enc_name      = encrypt_field(data["patient_name"], aes_key)
            enc_clinician = encrypt_field(data["clinician_name"], aes_key)
            enc_date      = encrypt_field(
                data.get("assessment_date", time.strftime("%Y-%m-%d")),
                aes_key,
            )

            # Encrypt raw features as a JSON blob
            features_record = {
                "raw_values":    raw_values,
                "feature_names": FEATURE_NAMES,
            }
            enc_features = encrypt_json_blob(features_record, aes_key)

            # ─── STORE encrypted patient (raw data NEVER touches disk) ───
            patient_id = store_patient(
                enc_name, enc_clinician, enc_date, enc_features
            )

            # ─── RUN PREDICTION ──────────────────────────────────────────
            prediction = _run_prediction(raw_values)

            # ─── ENCRYPT prediction results ──────────────────────────────
            enc_risk_score  = encrypt_field(str(prediction["risk_score"]), aes_key)
            enc_risk_class  = encrypt_field(prediction["risk_class"], aes_key)
            enc_shap        = encrypt_json_blob(prediction["shap_values"], aes_key)
            enc_plain_prob  = encrypt_field(str(prediction["plain_prob"]), aes_key)

            # ─── STORE encrypted prediction ──────────────────────────────
            pred_id = store_prediction(
                patient_id      = patient_id,
                enc_risk_score  = enc_risk_score,
                enc_risk_class  = enc_risk_class,
                enc_shap_values = enc_shap,
                enc_plain_prob  = enc_plain_prob,
                he_used         = prediction["he_used"],
                encryption_time_ms = prediction["encryption_time_ms"],
                inference_time_ms  = prediction["inference_time_ms"],
                total_time_ms      = prediction["total_time_ms"],
            )

            # ─── RETURN decrypted results to client ──────────────────────
            return jsonify({
                "patient_id":    patient_id,
                "prediction_id": pred_id,
                "prediction":    prediction,
                "message":       "Patient data encrypted and stored. "
                                 "Raw data was NEVER written to disk.",
            }), 201

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # ── GET /api/patients  —  list all ───────────────────────────────────
    @app.route("/api/patients", methods=["GET"])
    @require_api_key
    def list_all_patients():
        """
        Return list of all patients.  By default returns decrypted names
        and dates for display. Pass ?encrypted=true to see raw ciphertext.
        """
        try:
            rows      = list_patients()
            encrypted = request.args.get("encrypted", "false").lower() == "true"

            result = []
            for row in rows:
                if encrypted:
                    result.append(row)
                else:
                    result.append({
                        "id":         row["id"],
                        "created_at": row["created_at"],
                        "name":       decrypt_field(row["enc_name"], aes_key),
                        "clinician":  decrypt_field(row["enc_clinician"], aes_key),
                        "date":       decrypt_field(row["enc_date"], aes_key),
                    })

            return jsonify({"patients": result, "count": len(result)})

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # ── GET /api/patients/<id> ───────────────────────────────────────────
    @app.route("/api/patients/<int:patient_id>", methods=["GET"])
    @require_api_key
    def get_single_patient(patient_id: int):
        """Retrieve and decrypt a single patient record."""
        try:
            row = get_patient(patient_id)
            if not row:
                return jsonify({"error": "Patient not found"}), 404

            encrypted = request.args.get("encrypted", "false").lower() == "true"

            if encrypted:
                return jsonify({"patient": row})

            # Decrypt all fields
            features_data = decrypt_json_blob(row["enc_features"], aes_key)

            patient = {
                "id":         row["id"],
                "created_at": row["created_at"],
                "name":       decrypt_field(row["enc_name"], aes_key),
                "clinician":  decrypt_field(row["enc_clinician"], aes_key),
                "date":       decrypt_field(row["enc_date"], aes_key),
                "features":   dict(zip(
                    features_data["feature_names"],
                    features_data["raw_values"],
                )),
            }
            return jsonify({"patient": patient})

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # ── GET /api/patients/<id>/prediction ────────────────────────────────
    @app.route("/api/patients/<int:patient_id>/prediction", methods=["GET"])
    @require_api_key
    def get_patient_prediction(patient_id: int):
        """Retrieve and decrypt the latest prediction for a patient."""
        try:
            row = get_prediction(patient_id)
            if not row:
                return jsonify({"error": "Prediction not found"}), 404

            encrypted = request.args.get("encrypted", "false").lower() == "true"

            if encrypted:
                return jsonify({"prediction": row})

            prediction = {
                "id":                  row["id"],
                "patient_id":          row["patient_id"],
                "created_at":          row["created_at"],
                "risk_score":          float(decrypt_field(row["enc_risk_score"], aes_key)),
                "risk_class":          decrypt_field(row["enc_risk_class"], aes_key),
                "shap_values":         decrypt_json_blob(row["enc_shap_values"], aes_key),
                "plain_prob":          float(decrypt_field(row["enc_plain_prob"], aes_key)),
                "he_used":             bool(row["he_used"]),
                "encryption_time_ms":  row["encryption_time_ms"],
                "inference_time_ms":   row["inference_time_ms"],
                "total_time_ms":       row["total_time_ms"],
            }
            return jsonify({"prediction": prediction})

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # ── DELETE /api/patients/<id> ────────────────────────────────────────
    @app.route("/api/patients/<int:patient_id>", methods=["DELETE"])
    @require_api_key
    def remove_patient(patient_id: int):
        """Delete a patient and all associated predictions."""
        try:
            deleted = delete_patient(patient_id)
            if not deleted:
                return jsonify({"error": "Patient not found"}), 404
            return jsonify({
                "message":    f"Patient {patient_id} and all predictions deleted.",
                "patient_id": patient_id,
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # ── POST /api/predict  —  predict without storage ────────────────────
    @app.route("/api/predict", methods=["POST"])
    @require_api_key
    def predict_only():
        """
        Run prediction on raw features WITHOUT storing anything.
        Useful for one-off predictions.
        """
        try:
            data = request.get_json(force=True)
            if not data or "features" not in data:
                return jsonify({"error": "features dict required"}), 400

            features = data["features"]
            missing  = [f for f in FEATURE_NAMES if f not in features]
            if missing:
                return jsonify({"error": f"Missing features: {missing}"}), 400

            raw_values = [float(features[f]) for f in FEATURE_NAMES]
            prediction = _run_prediction(raw_values)

            return jsonify({
                "prediction": prediction,
                "stored":     False,
                "message":    "Prediction computed in memory only. "
                              "Nothing was stored to disk.",
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # ── GET /api/patients/<id>/history ───────────────────────────────────
    @app.route("/api/patients/<int:patient_id>/history", methods=["GET"])
    @require_api_key
    def get_prediction_history(patient_id: int):
        """Return all predictions for a patient, newest first."""
        try:
            rows = get_all_predictions(patient_id)
            if not rows:
                return jsonify({"error": "No predictions found"}), 404

            predictions = []
            for row in rows:
                predictions.append({
                    "id":                  row["id"],
                    "created_at":          row["created_at"],
                    "risk_score":          float(decrypt_field(row["enc_risk_score"], aes_key)),
                    "risk_class":          decrypt_field(row["enc_risk_class"], aes_key),
                    "he_used":             bool(row["he_used"]),
                    "encryption_time_ms":  row["encryption_time_ms"],
                    "inference_time_ms":   row["inference_time_ms"],
                    "total_time_ms":       row["total_time_ms"],
                })

            return jsonify({
                "patient_id":  patient_id,
                "predictions": predictions,
                "count":       len(predictions),
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import backend.config as _cfg
    print("╔══════════════════════════════════════════════════════╗")
    print("║   CardioShield — Privacy-Preserving Backend API     ║")
    print("║   All patient data encrypted with AES-256-GCM       ║")
    print("║   Raw data NEVER touches the database               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  Database : {DB_PATH}")
    print(f"  Server   : http://{HOST}:{PORT}")
    print(f"  API Key  : {API_KEY}")
    print(f"  Debug    : {DEBUG}")
    if not os.getenv("CARDIOSHIELD_API_KEY"):
        print()
        print("  ⚠  API key auto-generated (set CARDIOSHIELD_API_KEY env var for production)")
    print()

    app = create_app()
    app.run(host=HOST, port=PORT, debug=DEBUG)
