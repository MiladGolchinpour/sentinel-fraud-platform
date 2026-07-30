# Sentinel

> An end-to-end fraud detection platform for detecting fraudulent financial transactions using machine learning.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-Ready-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red) ![Docker](https://img.shields.io/badge/Docker-Supported-2496ED) [![Sentinel CI](https://github.com/MiladGolchinpour/sentinel-fraud-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/MiladGolchinpour/sentinel-fraud-platform/actions/workflows/ci.yml)

Sentinel is a production-oriented machine learning system for real-time fraud detection. The project includes data preprocessing, feature engineering, model training, SHAP-based explainability, a REST API built with FastAPI, an interactive Streamlit dashboard, automated testing, and Dockerized deployment.

## Features

- End-to-end machine learning pipeline
- Automated feature engineering
- XGBoost fraud detection model
- SHAP-based prediction explanations
- FastAPI REST API
- Interactive Streamlit dashboard
- Dockerized deployment
- Automated testing with Pytest

## Architecture

```text
Transaction
    │
    ▼
Validation → Feature Engineering → Preprocessing → XGBoost Model → Fraud Probability
                                                                  ├── SHAP Explainability
                                                                  └── FastAPI → Streamlit Dashboard
```

## Project Structure

```text
Sentinel/
├── dashboard/         # Streamlit dashboard
├── data/              # Dataset (ignored by Git)
├── models/            # Trained model and artifacts
├── src/               # Training, API and ML pipeline
├── tests/             # Unit tests
├── .github/           # GitHub Actions
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Clone the repository

```bash
git clone https://github.com/MiladGolchinpour/sentinel-fraud-platform.git
cd sentinel-fraud-platform
```

### Option 1 — Run manually

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Start the API:

```bash
uvicorn src.api:app --reload
```

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

API documentation: `http://localhost:8000/docs`

### Option 2 — Run with Docker

```bash
docker compose up --build
```

This starts both the FastAPI service and the Streamlit dashboard.

---

### Live Demo

Dashboard:
https://sentinel-dashboard-8586.onrender.com

API:
https://sentinel-api-ad1r.onrender.com/docs

---

### Dataset

This project uses the [PaySim financial transactions dataset](https://www.kaggle.com/datasets/mtalaltariq/paysim-data).

The dataset is **not included** in this repository due to its size.

For manual training, download the dataset and place it in:

```text
data/raw/paysim.csv
```

Then train the model:

```bash
python src/training.py
```

---

### Example Request

```json
{
  "type": "TRANSFER",
  "amount": 500000,
  "oldbalanceOrg": 500000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 500000,
  "step": 100
}
```

### Response

```json
{
  "prediction": "fraud",
  "fraud_probability": 0.96,
  "risk_level": "high",
  "top_reasons": [
    "Transfer transaction"
  ]
}
```

## Model Performance

| Metric | Score |
|--------|------:|
| Accuracy | 99.98% |
| Precision | 97.43% |
| Recall | 87.59% |
| F1 Score | 92.25% |
| ROC-AUC | 99.94% |
| PR-AUC | 97.90% |

The model was evaluated on a held-out test set using XGBoost with engineered features and threshold tuning. Feature engineering was carefully reviewed to avoid target leakage and ensure realistic performance.

## Dashboard

The Streamlit dashboard provides:

- Real-time fraud prediction
- SHAP-based prediction explanations
- Model performance metrics
- Interactive data exploration

<p align="center">
  <img src="docs/dashboard.png">
</p>

## Testing

Run the test suite with:

```bash
pytest
```

## Future Work

- Model monitoring and data drift detection
- Hyperparameter optimization
- Support for additional ML models

## License

MIT
