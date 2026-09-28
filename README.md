# CardioShield -- Privacy-Preserving Medical Diagnosis System

CardioShield is a cardiovascular risk assessment platform that performs machine learning inference entirely over encrypted data using homomorphic encryption. Patient clinical biomarkers are never exposed in plaintext to the inference server. The system uses TenSEAL's CKKS scheme with 128-bit RLWE security to evaluate a trained Logistic Regression model directly on ciphertexts, producing risk scores without decrypting sensitive medical records.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Cryptographic Architecture](#cryptographic-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Application](#running-the-application)
- [Verification and Testing](#verification-and-testing)
- [Model Performance](#model-performance)
- [Deployment](#deployment)


## Problem Statement

Healthcare institutions face a fundamental tension between leveraging AI for clinical decision support and protecting patient data privacy. Cloud-based diagnostic models require access to sensitive medical records, creating compliance risks under regulations like HIPAA. Local deployments solve privacy but sacrifice scalability and maintainability.

CardioShield resolves this by applying Homomorphic Encryption (HE) to the inference pipeline. The AI model evaluates encrypted patient data and returns encrypted results. At no point does the server observe raw clinical values.


## How It Works

CardioShield operates in four stages:

1. **Clinical Data Input** -- A clinician enters 13 standardized biomarkers (age, blood pressure, cholesterol, ECG results, etc.) derived from the UCI Cleveland Heart Disease dataset format.

2. **Client-Side Encryption** -- The feature vector is scaled using a pre-fitted StandardScaler and encrypted into a CKKS ciphertext using TenSEAL. The encryption happens before any data leaves the client boundary.

3. **Homomorphic Inference** -- The encrypted feature vector is sent to the backend, where the Logistic Regression model's weights and bias are applied directly on the ciphertext. The inner product and a degree-3 polynomial sigmoid approximation are computed without decryption.

4. **Decryption and Explainability** -- The encrypted probability scalar is decrypted on the client side. SHAP (SHapley Additive exPlanations) values provide feature-level attribution for the prediction, enabling clinical interpretability.

---

## Cryptographic Architecture

| Component | Specification |
|---|---|
| Encryption Scheme | CKKS (Cheon-Kim-Kim-Song) |
| Security Level | 128-bit RLWE (Ring Learning With Errors) |
| Polynomial Modulus Degree | 8192 |
| Coefficient Modulus Bit Sizes | [40, 21, 21, 21, 21, 21, 40] |
| Global Scale | 2^20 |
| Sigmoid Approximation | Degree-3 minimax polynomial on [-4, 4] |
| Library | TenSEAL (Python bindings for Microsoft SEAL) |
| Database Encryption | AES-256-GCM authenticated encryption at rest |
| Key Derivation | PBKDF2-HMAC-SHA256, 100,000 iterations |

The sigmoid activation function is approximated as:

```
sigma(z) = 0.5 + 0.2159z - 0.0093z^3
```

This polynomial approximation preserves classification accuracy while remaining compatible with the multiplicative depth constraints of the CKKS scheme.

---

## Technology Stack

**Backend**
- Python 3.11
- FastAPI with Uvicorn ASGI server
- TenSEAL for homomorphic encryption
- scikit-learn for model training
- SHAP for feature-level explainability
- PyCryptodome for AES-256-GCM database encryption
- SQLite for persistent patient record storage
- FPDF2 for clinical PDF report generation

**Frontend**
- React 18 with Vite build tooling
- Tailwind CSS for responsive UI
- Axios for API communication
- Lucide React for iconography

**Infrastructure**
- Docker multi-stage build (Node.js frontend build + Python backend)
- Docker Compose for single-command orchestration


## Project Structure

```
CardioShield/
|-- backend/
|   |-- app.py              # FastAPI application and API endpoints
|   |-- config.py            # Configuration, paths, and key derivation
|   |-- crypto_utils.py      # AES-256-GCM encrypt/decrypt utilities
|   |-- database.py          # SQLite ORM with encrypted field storage
|   |-- report_generator.py  # Clinical PDF report generation
|   |-- test_backend.py      # Backend API integration tests
|
|-- frontend/
|   |-- src/
|   |   |-- components/      # React UI components
|   |   |-- api/              # API client layer
|   |   |-- utils/            # Feature configs and constants
|   |   |-- App.jsx           # Root application component
|   |   |-- index.css         # Global styles and animations
|   |   +-- main.jsx          # Entry point
|   |-- index.html            # SPA shell
|   |-- vite.config.js        # Vite build configuration
|   +-- tailwind.config.js    # Tailwind theme extensions
|
|-- artifacts/
|   |-- model.pkl             # Trained Logistic Regression model
|   |-- scaler.pkl            # Fitted StandardScaler
|   |-- X_train.pkl           # Training subset for SHAP background
|   +-- metrics.json          # Computed evaluation metrics
|
|-- he_engine.py              # TenSEAL CKKS encryption and HE inference engine
|-- model_trainer.py          # Model training, evaluation, and artifact export
|-- test_pipeline.py          # End-to-end verification suite (6 tests)
|-- conftest.py               # Pytest configuration
|-- requirements.txt          # Python dependencies
|-- Dockerfile                # Multi-stage production container
|-- docker-compose.yml        # Container orchestration
+-- .env.example              # Environment variable template
```

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- Node.js 18 or later
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Satvik-Shashank/CardioShield_Privacy_Preserving_Medical_Diagnosis-System.git
cd CardioShield_Privacy_Preserving_Medical_Diagnosis-System
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model and generate artifacts:

```bash
python model_trainer.py
```

4. Verify the HE engine:

```bash
python he_engine.py
```

5. Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

### Development Mode

Start the backend API server:

```bash
uvicorn backend.app:app --reload --port 8000
```

In a separate terminal, start the frontend dev server:

```bash
cd frontend
npm run dev
```

The application will be available at `http://localhost:5173` with API requests proxied to the backend.

### Docker (Production)

Build and run with Docker Compose:

```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`.

---

## Verification and Testing

CardioShield includes a comprehensive verification suite that validates the entire pipeline:

```bash
python test_pipeline.py
```

The suite runs six tests:

| Test | Validation |
|---|---|
| Model Accuracy | Holdout accuracy >= 80% on UCI Cleveland test set |
| TenSEAL Round-Trip | CKKS encrypt-decrypt precision error < 0.01 |
| HE Inference | Ciphertext classification matches plaintext at >= 75% |
| Sigmoid Approximation | L-infinity error < 0.25 on [-4, 4] |
| Artifact Integrity | All model files and metrics.json exist on disk |
| AES-256-GCM | Authenticated encryption round-trip with tamper detection |

Run backend API tests:

```bash
pytest backend/test_backend.py -v
```

Run unit tests:

```bash
pytest tests/ -v
```

---

## Model Performance

The model is trained on the UCI Cleveland Heart Disease dataset (303 samples, 13 features) using Logistic Regression with L2 regularization.

| Metric | Value |
|---|---|
| 5-Fold Cross-Validation Accuracy | ~83% |
| Holdout Test Accuracy | ~87% |
| ROC-AUC | ~91% |
| HE vs Plaintext Classification Match | >= 93% |
| Average HE Inference Latency | < 200 ms |

Feature-level SHAP attributions are computed for every prediction, providing clinicians with interpretable explanations of which biomarkers most influenced the risk score.

---

## Deployment

### Docker

The included Dockerfile uses a multi-stage build:
- **Stage 1**: Builds the React frontend into a static SPA bundle
- **Stage 2**: Sets up the Python backend, trains the model, verifies the HE engine, runs the test suite, and serves the built frontend as static files

```bash
docker build -t cardioshield .
docker run -p 8000:8000 cardioshield
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```
CARDIOSHIELD_API_KEY=your-api-key
CARDIOSHIELD_SECRET_KEY=your-encryption-passphrase
```

---

## License

This project is developed as a research and clinical decision-support prototype demonstrating privacy-preserving machine learning inference using homomorphic encryption.
