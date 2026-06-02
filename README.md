# Alternative Credit Scoring System — Cameroonian Vendors

A machine-learning credit scoring system that evaluates the creditworthiness of small and medium vendors in Cameroon using alternative data sources — mobile money transaction history, utility payment records, supplier credit history, and business financials — instead of traditional bank statements. The system trains three models (Logistic Regression, Random Forest, XGBoost) on a dataset of 5,100 synthetic but statistically grounded vendor records, selects the best performer, and exposes it through a Flask REST API backed by a professional web interface where bank officers can input vendor details and instantly receive a credit score (300–850), a lending decision (APPROVE / REVIEW / REJECT), a calibrated default probability, and a risk band.

---

## Model Performance

All three models were trained on an 80/20 stratified split (4,080 train / 1,020 test). The Logistic Regression model was selected as the production model and further calibrated using Platt scaling (5-fold cross-validation).

| Model | Accuracy | AUC | F1 Score |
|---|---|---|---|
| **Logistic Regression** ✓ | **0.9637** | **0.9933** | **0.9338** |
| XGBoost | 0.9569 | 0.9902 | 0.9214 |
| Random Forest | 0.9382 | 0.9851 | 0.8857 |

> ✓ Selected as production model — highest accuracy and AUC, calibrated with Platt scaling.

**Decision thresholds (calibrated default probability):**

| Probability | Decision | Risk Band |
|---|---|---|
| < 0.30 | APPROVE | Low |
| 0.30 – 0.59 | REVIEW | Medium |
| ≥ 0.60 | REJECT | High |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML | scikit-learn, XGBoost |
| Data | pandas, NumPy |
| API | Flask, flask-cors |
| Model persistence | joblib |
| Frontend | Vanilla HTML / CSS / JavaScript |

---

## Project Structure

```
Credit_score-Updated/
│
├── data/                          # ⚠ excluded from repo (see note below)
│   ├── alt_data_expanded_v2.csv   #   training dataset (5,100 vendors)
│   └── final_scored_dataset_v2.csv
│
├── model/                         # ⚠ excluded from repo
│   └── credit_scorer.pkl          #   model bundle — generate with export_model.py
│
├── notebooks/
│   ├── credit_scoring_v2.ipynb    # full training pipeline (EDA → modelling → evaluation)
│   └── credit_scoring.ipynb       # original exploration notebook
│
├── outputs/                       # EDA and evaluation charts (committed)
│   ├── eda_analysis.png
│   ├── confusion_matrices.png
│   ├── roc_comparison.png
│   ├── calibration.png
│   ├── fairness_audit.png
│   └── feature_importance.png
│
├── templates/
│   └── index.html                 # single-file frontend (HTML + CSS + JS)
│
├── app.py                         # Flask API
├── export_model.py                # training + model export script
├── requirements.txt               # pinned Python dependencies
├── .gitignore
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset and train the model

Open and run all cells in `notebooks/credit_scoring_v2.ipynb` to generate `data/alt_data_expanded_v2.csv`. Then export the trained model:

```bash
python export_model.py
```

Expected output:
```
Loading data ... 5,100 records loaded.
Training Logistic Regression ...
Calibrating with Platt scaling (5-fold CV) ...

Calibrated model — Test Accuracy: 0.9637  AUC: 0.9933  F1: 0.9338

Model bundle saved -> model/credit_scorer.pkl
```

### 3. Start the API server

```bash
python app.py
```

### 4. Open the frontend

```
http://localhost:5000
```

---

## API Reference

### `GET /health`

Returns the model load status. Use this to confirm the server is ready before sending scoring requests.

**Response — healthy (HTTP 200):**
```json
{
  "status": "ok",
  "model": "Logistic Regression (Calibrated)"
}
```

**Response — model missing or corrupted (HTTP 503):**
```json
{
  "status": "degraded",
  "model": null,
  "error": "Model file not found. Run python export_model.py to generate it."
}
```

---

### `POST /score`

Scores a single vendor and returns a credit assessment.

**Request** — `Content-Type: application/json`

```json
{
  "age": 35,
  "gender": "Male",
  "region": "Douala",
  "business_type": "Retail",
  "years_in_business": 5,
  "monthly_revenue_xaf": 150000,
  "monthly_expenses_xaf": 90000,
  "profit_margin": 0.40,
  "mobile_money_txn_count": 50,
  "avg_wallet_balance_xaf": 75000,
  "utility_payment_score": 1,
  "supplier_credit_history": 1,
  "credit_score": 620
}
```

| Field | Type | Accepted values |
|---|---|---|
| `age` | int | 21–65 |
| `gender` | string | `"Male"` \| `"Female"` |
| `region` | string | `"Douala"` \| `"Yaounde"` \| `"Bamenda"` \| `"Bafoussam"` \| `"Garoua"` |
| `business_type` | string | `"Retail"` \| `"Food & Beverage"` \| `"Services"` \| `"Agriculture"` \| `"Manufacturing"` |
| `years_in_business` | int | 0–25 |
| `monthly_revenue_xaf` | float | > 0 |
| `monthly_expenses_xaf` | float | ≥ 0 |
| `profit_margin` | float | 0.01–0.99 |
| `mobile_money_txn_count` | int | 10–80 |
| `avg_wallet_balance_xaf` | float | ≥ 0 |
| `utility_payment_score` | int | `0` = No history \| `1` = Has history |
| `supplier_credit_history` | int | `0` = No history \| `1` = Has history |
| `credit_score` | int | 300–850 (bureau score) |

**Response — success (HTTP 200):**
```json
{
  "status": "success",
  "credit_score": 761,
  "default_probability": 0.16,
  "decision": "APPROVE",
  "model_used": "Logistic Regression (Calibrated)",
  "risk_band": "Low"
}
```

**Response — validation error (HTTP 400):**
```json
{
  "status": "error",
  "message": "Missing fields: ['region', 'credit_score']"
}
```

---

### `POST /batch-score`

Scores up to **50 vendors** in a single request. Per-vendor errors are reported inline without failing the entire batch.

**Request** — JSON array of vendor objects (same fields as `/score`):

```json
[
  { "age": 35, "gender": "Male", "region": "Douala", ... },
  { "age": 28, "gender": "Female", "region": "Yaounde", ... }
]
```

**Response (HTTP 200):**
```json
{
  "status": "success",
  "summary": {
    "total": 2,
    "approved": 1,
    "review": 0,
    "rejected": 1,
    "errors": 0
  },
  "results": [
    { "index": 0, "status": "success", "credit_score": 761, "decision": "APPROVE", "default_probability": 0.16, "risk_band": "Low", "model_used": "Logistic Regression (Calibrated)" },
    { "index": 1, "status": "success", "credit_score": 303, "decision": "REJECT",  "default_probability": 0.99, "risk_band": "High", "model_used": "Logistic Regression (Calibrated)" }
  ]
}
```

Returns HTTP 400 if the body is not an array, is empty, or exceeds 50 items. Returns HTTP 503 if the model is not loaded.

---

## Dataset and model files

The training dataset (`data/`) and serialised model (`model/`) are excluded from this repository via `.gitignore` because CSV files are large and `.pkl` files should always be regenerated from source rather than trusted as binaries.

To reproduce the full pipeline from scratch:

1. Run all cells in `notebooks/credit_scoring_v2.ipynb` — this generates `data/alt_data_expanded_v2.csv`
2. Run `python export_model.py` — this trains, calibrates, and saves `model/credit_scorer.pkl`
3. Run `python app.py` — the API is now ready

---

## Fairness note

The model was audited for demographic bias across gender and region. The gender approval gap is within the accepted < 5% threshold. Regional default probability variation reflects genuine economic differences in the underlying data rather than model bias. See `outputs/fairness_audit.png` for the full breakdown.
