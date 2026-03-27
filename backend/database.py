"""
database.py
───────────
SQLite database layer for CardioShield.

ALL stored values are AES-256-GCM ciphertext — the database NEVER contains
any raw patient data.  Schema columns are explicitly named `enc_*` to make
this unmistakable.
"""

import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from .config import DB_PATH


# ── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      REAL    NOT NULL,
    enc_name        TEXT    NOT NULL,
    enc_clinician   TEXT    NOT NULL,
    enc_date        TEXT    NOT NULL,
    enc_features    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    created_at          REAL    NOT NULL,
    enc_risk_score      TEXT    NOT NULL,
    enc_risk_class      TEXT    NOT NULL,
    enc_shap_values     TEXT    NOT NULL,
    enc_plain_prob      TEXT    NOT NULL,
    he_used             INTEGER NOT NULL DEFAULT 0,
    encryption_time_ms  REAL    NOT NULL DEFAULT 0,
    inference_time_ms   REAL    NOT NULL DEFAULT 0,
    total_time_ms       REAL    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_predictions_patient
    ON predictions(patient_id);
"""


# ── Connection helper ────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """
    Context manager for a SQLite connection with:
      - WAL journal mode (better concurrent read performance)
      - Foreign keys enabled
      - Row factory for dict-like access
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript(_SCHEMA)


# ── Patient CRUD ─────────────────────────────────────────────────────────────

def store_patient(
    enc_name: str,
    enc_clinician: str,
    enc_date: str,
    enc_features: str,
) -> int:
    """
    Insert an encrypted patient record.  Every argument is already
    AES-256-GCM ciphertext — this function never touches raw data.

    Returns the auto-generated patient ID.
    """
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO patients
               (created_at, enc_name, enc_clinician, enc_date, enc_features)
               VALUES (?, ?, ?, ?, ?)""",
            (time.time(), enc_name, enc_clinician, enc_date, enc_features),
        )
        return cur.lastrowid


def get_patient(patient_id: int) -> Optional[dict]:
    """Return a single patient row as a dict, or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None


def list_patients() -> list[dict]:
    """Return all patient rows (still encrypted), newest first."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, enc_name, enc_clinician, enc_date "
            "FROM patients ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_patient(patient_id: int) -> bool:
    """Delete a patient and cascade to their predictions.  Returns True if found."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        return cur.rowcount > 0


# ── Prediction CRUD ──────────────────────────────────────────────────────────

def store_prediction(
    patient_id: int,
    enc_risk_score: str,
    enc_risk_class: str,
    enc_shap_values: str,
    enc_plain_prob: str,
    he_used: bool,
    encryption_time_ms: float,
    inference_time_ms: float,
    total_time_ms: float,
) -> int:
    """Insert an encrypted prediction row.  Returns prediction ID."""
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (patient_id, created_at,
                enc_risk_score, enc_risk_class, enc_shap_values, enc_plain_prob,
                he_used, encryption_time_ms, inference_time_ms, total_time_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patient_id, time.time(),
                enc_risk_score, enc_risk_class, enc_shap_values, enc_plain_prob,
                int(he_used), encryption_time_ms, inference_time_ms, total_time_ms,
            ),
        )
        return cur.lastrowid


def get_prediction(patient_id: int) -> Optional[dict]:
    """Return the most recent prediction for a patient, or None."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM predictions
               WHERE patient_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (patient_id,),
        ).fetchone()
        return dict(row) if row else None


def get_all_predictions(patient_id: int) -> list[dict]:
    """Return all predictions for a patient, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM predictions
               WHERE patient_id = ?
               ORDER BY created_at DESC""",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]
