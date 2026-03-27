# CardioShield_Privacy_Preserving_Medical_Diagnosis-System
CardioShield is a secure, AI-powered system for predicting heart disease risk while guaranteeing complete patient data privacy. It combines machine learning with homomorphic encryption, allowing predictions to be made directly on encrypted data — without ever exposing sensitive medical information.

---

## Privacy-Preserving Heart Disease Prediction using Homomorphic Encryption

CardioShield is an AI-powered healthcare application designed to predict heart disease risk while ensuring complete data privacy. The system leverages homomorphic encryption to perform machine learning inference directly on encrypted patient data, ensuring that sensitive medical information is never exposed during computation.

---

## Features

- End-to-end encryption using CKKS (TenSEAL), ensuring patient data remains encrypted throughout the pipeline  
- Logistic Regression model trained on the UCI Heart Disease dataset with approximately 87–90% accuracy  
- Explainable AI using SHAP to provide transparency into feature contributions  
- Real-time inference with secure predictions delivered within a few seconds  
- Downloadable PDF report containing structured prediction results  

---

## System Workflow

1. The user inputs 13 clinical features such as age, cholesterol, and ECG results  
2. The data is encrypted locally using the CKKS homomorphic encryption scheme  
3. Encrypted data is transmitted to the server  
4. The machine learning model performs inference directly on encrypted data  
5. The encrypted prediction result is returned to the client  
6. The client decrypts the result and displays the prediction  
7. SHAP is used to explain the contribution of each feature  

At no stage is the patient’s raw data exposed to the server.

---

## Model Details

- **Algorithm:** Logistic Regression  
- **Dataset:** UCI Heart Disease (Cleveland dataset)  
- **Number of features:** 13 clinical attributes  
- **Accuracy:** Approximately 87–90% using 5-fold cross-validation  

### Preprocessing

- StandardScaler normalization  
- Median imputation for missing values (ca, thal)  

---

## Privacy Guarantee

CardioShield ensures strong privacy guarantees through homomorphic encryption:

- No plaintext data leaves the client device  
- The server processes only encrypted data  
- Computation is performed without decryption  
- Only the client can decrypt the final prediction  

This approach provides mathematical privacy guarantees rather than relying on policy-based protection.

---

## Technology Stack

- **Frontend:** Streamlit  
- **Machine Learning:** Scikit-learn  
- **Encryption:** TenSEAL (CKKS scheme)  
- **Explainability:** SHAP  
- **Backend:** Python  

---
## Example Usage

- Input patient clinical data through the interface  
- Execute encrypted prediction  
- View risk score and explanation  
- Download a structured report  

---

## Use Cases

- Privacy-preserving healthcare applications  
- Secure medical diagnostics  
- Research in encrypted machine learning  
- Systems requiring strict data confidentiality  

---

## Value Proposition

Traditional healthcare AI systems require access to raw patient data, introducing privacy risks.  
CardioShield eliminates this risk by enabling accurate and explainable predictions on encrypted data, ensuring that sensitive information is never exposed.

---

## Disclaimer

This project is intended for research and demonstration purposes only.  
It should not be used as a substitute for professional medical diagnosis or treatment.

---

## Team

Developed as part of the Alexa Developers SRM Hackathon.
