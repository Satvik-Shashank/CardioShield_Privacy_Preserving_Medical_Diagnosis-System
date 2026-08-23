"""
tests/test_backend_api.py
─────────────────────────
Integration tests for the FastAPI backend endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app import app, load_all_resources

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def init_app_resources():
    """Ensure startup resources are loaded."""
    load_all_resources()


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["he_engine"]["scheme"] == "CKKS"


def test_metrics_endpoint():
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "accuracy" in data
    assert "roc_auc" in data
    assert "he_match_rate" in data
    assert data["accuracy"] > 70.0


def test_predict_endpoint_real_he():
    payload = {
        "patient_name": "Test Patient",
        "clinician_name": "Dr. Test",
        "assessment_date": "2026-08-23",
        "save_to_db": False,
        "features": {
            "age": 54,
            "sex": 1,
            "cp": 0,
            "trestbps": 130,
            "chol": 246,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        },
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "risk_score_pct" in data
    assert "risk_class" in data
    assert "shap_values" in data
    assert data["he_used"] is True
    assert len(data["hex_proof"]) > 20
    assert data["ciphertext_bytes"] > 0
    assert data["t_he_ms"] > 0


def test_patient_crud_lifecycle():
    # 1. Create Patient
    payload = {
        "patient_name": "Alice Wonderland",
        "clinician_name": "Dr. Cardiology",
        "assessment_date": "2026-08-23",
        "features": {
            "age": 62,
            "sex": 0,
            "cp": 2,
            "trestbps": 140,
            "chol": 260,
            "fbs": 1,
            "restecg": 0,
            "thalach": 135,
            "exang": 1,
            "oldpeak": 2.2,
            "slope": 2,
            "ca": 1,
            "thal": 3,
        },
    }
    create_res = client.post("/api/patients", json=payload)
    assert create_res.status_code == 200
    p_id = create_res.json()["patient_id"]
    assert p_id is not None

    # 2. List Patients
    list_res = client.get("/api/patients")
    assert list_res.status_code == 200
    patients = list_res.json()
    assert any(p["id"] == p_id for p in patients)

    # 3. Get Details
    det_res = client.get(f"/api/patients/{p_id}")
    assert det_res.status_code == 200
    p_det = det_res.json()
    assert p_det["patient_name"] == "Alice Wonderland"
    assert len(p_det["predictions"]) >= 1

    # 4. Delete Patient
    del_res = client.delete(f"/api/patients/{p_id}")
    assert del_res.status_code == 200

    # 5. Verify Deletion
    get_after_del = client.get(f"/api/patients/{p_id}")
    assert get_after_del.status_code == 404


def test_pdf_report_export():
    payload = {
        "patient_name": "John Smith",
        "clinician_name": "Dr. House",
        "assessment_date": "2026-08-23",
        "features": {
            "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240,
            "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
        },
        "risk_prob": 0.65,
        "risk_class": "High Risk",
        "shap_values": {"age": 0.12, "chol": 0.25, "cp": 0.40},
        "latency_ms": 145.2,
        "he_used": True,
    }
    res = client.post("/api/report/pdf", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF")
