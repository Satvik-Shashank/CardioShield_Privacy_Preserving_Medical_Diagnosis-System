# CardioShield_Privacy_Preserving_Medical_Diagnosis-System
CardioShield is a secure, AI-powered system for predicting heart disease risk while guaranteeing complete patient data privacy. It combines machine learning with homomorphic encryption, allowing predictions to be made directly on encrypted data — without ever exposing sensitive medical information.

Deployed and live at : https://cardioshield.streamlit.app/

---

## Overview

CardioShield is a clinical decision support system designed to estimate cardiovascular risk using machine learning on structured patient data.

The system enables real-time risk prediction, provides explainable outputs, and presents results through a simple web interface. It is built as a foundation for privacy-preserving, interpretable AI systems in healthcare.

Cardiovascular diseases are among the leading causes of mortality globally, and early risk assessment is critical for prevention and intervention.

---

## Key Features

- AI-based cardiovascular risk prediction using clinical parameters
- Explainable AI using SHAP for feature-level interpretability
- Structured clinical input interface
- Real-time inference and probability scoring
- Web-based deployment using Streamlit

---

## System Architecture

```
User Input (Clinical Data)
        ↓
Data Preprocessing
        ↓
Machine Learning Model
        ↓
Prediction Output (Risk Score)
        ↓
Explainability Layer (SHAP)
        ↓
Visualization Interface (Streamlit)
```

---

## Machine Learning Pipeline

### Data Processing
- Feature scaling and normalization
- Handling missing values
- Structured tabular data preparation

### Model
- Logistic Regression (baseline model)
- Extendable to ensemble models

### Evaluation Metrics
- Accuracy
- ROC-AUC
- Precision and Recall

---

## Explainability

The system uses SHAP (SHapley Additive exPlanations) to:

- Identify feature contributions to predictions
- Provide transparency for decision-making
- Improve interpretability of model outputs

---

## Tech Stack

### Languages and Libraries
- Python
- NumPy
- Pandas
- Scikit-learn
- SHAP

### Visualization
- Matplotlib
- Seaborn
- Plotly (optional)

### Deployment
- Streamlit

---

## Installation and Setup

### Clone the repository
```bash
git clone https://github.com/your-username/cardioshield.git
cd cardioshield
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
streamlit run app.py
```

---

## Example Input

| Feature            | Example |
|-------------------|--------|
| Age               | 54     |
| Cholesterol       | 240    |
| Blood Pressure    | 140    |
| Max Heart Rate    | 150    |

---

## Output

- Risk probability (percentage)
- Risk category (low, moderate, high)
- Feature contribution analysis (SHAP)

---

## Use Cases

- Preventive cardiology screening
- Clinical decision support systems
- Remote patient monitoring tools
- Healthcare AI research and prototyping

---

## Disclaimer

This system is intended for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.

---

## Future Improvements

- Ensemble and advanced models (XGBoost, Random Forest)
- Confidence intervals for predictions
- Counterfactual explanations
- Patient history tracking and longitudinal analysis
- Backend API (FastAPI) with database integration
- Authentication and access control
- Homomorphic encryption for privacy-preserving inference

---

## Project Structure

```
cardioshield/
│── app.py
│── model/
│── utils/
│── data/
│── requirements.txt
│── README.md
```

---

## Author

Satvik Shashank Janga  
Computer Science Engineering Student  
Focus: AI, Machine Learning, and Systems

---

## License

This project is open-source and available under the MIT License.
