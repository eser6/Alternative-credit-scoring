import os
import numpy as np
import pandas as pd
import joblib
from functools import wraps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Model loading with graceful error handling ─────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'credit_scorer.pkl')

model = scaler = encoders = FEATURES = THRESHOLDS = None
MODEL_ERROR: str | None = None

try:
    if not os.path.exists(MODEL_PATH):
        MODEL_ERROR = (
            f"Model file not found at '{MODEL_PATH}'. "
            "Run  python export_model.py  to generate it."
        )
    else:
        _bundle = joblib.load(MODEL_PATH)
        _required = {'model', 'scaler', 'encoders', 'features', 'thresholds'}
        _missing = _required - set(_bundle.keys())
        if _missing:
            MODEL_ERROR = (
                f"Model bundle is corrupted — missing keys: {sorted(_missing)}. "
                "Re-run  python export_model.py  to rebuild it."
            )
        else:
            model      = _bundle['model']
            scaler     = _bundle['scaler']
            encoders   = _bundle['encoders']
            FEATURES   = _bundle['features']
            THRESHOLDS = _bundle['thresholds']
except Exception as exc:
    MODEL_ERROR = (
        f"Failed to load model ({type(exc).__name__}: {exc}). "
        "Re-run  python export_model.py  to rebuild it."
    )

MODEL_NAME    = "Logistic Regression (Calibrated)"
APPROVE_BELOW = (THRESHOLDS or {}).get('approve_below', 0.30)
REVIEW_BELOW  = (THRESHOLDS or {}).get('review_below',  0.60)

# ── Decorator: guard routes that need the model ────────────────────────────
def model_required(f):
    @wraps(f)
    def guarded(*args, **kwargs):
        if MODEL_ERROR:
            return jsonify({'status': 'error', 'message': MODEL_ERROR}), 503
        return f(*args, **kwargs)
    return guarded

# ── Feature engineering ────────────────────────────────────────────────────
def engineer_features(d: dict) -> dict:
    d = d.copy()
    d['expense_ratio']      = d['monthly_expenses_xaf'] / d['monthly_revenue_xaf']
    d['business_stability'] = int(d['years_in_business'] >= 3)
    d['log_revenue']        = float(np.log1p(d['monthly_revenue_xaf']))
    d['log_wallet']         = float(np.log1p(d['avg_wallet_balance_xaf']))
    d['composite_risk']     = d['expense_ratio'] * (1 - d['profit_margin'])
    d['payment_score']      = d['utility_payment_score'] * d['supplier_credit_history']
    return d

def encode_categoricals(d: dict) -> dict:
    d = d.copy()
    d['gender_enc'] = int(encoders['gender'].transform([d['gender']])[0])
    d['region_enc'] = int(encoders['region'].transform([d['region']])[0])
    d['btype_enc']  = int(encoders['business_type'].transform([d['business_type']])[0])
    return d

def credit_decision(prob: float) -> str:
    if prob < APPROVE_BELOW:  return 'APPROVE'
    if prob < REVIEW_BELOW:   return 'REVIEW'
    return 'REJECT'

def risk_band(prob: float) -> str:
    if prob < APPROVE_BELOW:  return 'Low'
    if prob < REVIEW_BELOW:   return 'Medium'
    return 'High'

def probability_to_score(prob: float) -> int:
    return int(np.clip(850 - (prob * 550), 300, 850))

NUMERIC_FIELDS = {
    'age':                    int,
    'years_in_business':      int,
    'monthly_revenue_xaf':    float,
    'monthly_expenses_xaf':   float,
    'profit_margin':          float,
    'mobile_money_txn_count': int,
    'avg_wallet_balance_xaf': float,
    'utility_payment_score':  int,
    'supplier_credit_history': int,
    'credit_score':           int,
}
STRING_FIELDS = ['gender', 'region', 'business_type']
ALL_FIELDS    = list(NUMERIC_FIELDS) + STRING_FIELDS

# ── Core scoring helper (used by both /score and /batch-score) ─────────────
def score_one(raw: dict) -> dict:
    missing = [f for f in ALL_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    d = dict(raw)
    for field, typ in NUMERIC_FIELDS.items():
        d[field] = typ(d[field])

    d = engineer_features(d)
    d = encode_categoricals(d)

    X    = pd.DataFrame([{f: d[f] for f in FEATURES}])
    X_sc = scaler.transform(X)
    prob = float(model.predict_proba(X_sc)[0, 1])

    return {
        'credit_score':        probability_to_score(prob),
        'default_probability': round(prob, 2),
        'decision':            credit_decision(prob),
        'model_used':          MODEL_NAME,
        'risk_band':           risk_band(prob),
    }

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    if MODEL_ERROR:
        return jsonify({'status': 'degraded', 'model': None, 'error': MODEL_ERROR}), 503
    return jsonify({'status': 'ok', 'model': MODEL_NAME})


@app.route('/score', methods=['POST'])
@model_required
def score():
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        result = score_one(data)
        return jsonify({'status': 'success', **result})
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/batch-score', methods=['POST'])
@model_required
def batch_score():
    """Score up to 50 vendors in one request.

    Body: JSON array of vendor objects (same fields as /score).

    Response:
        {
          "status": "success",
          "summary": { "total", "approved", "review", "rejected", "errors" },
          "results": [ { "index", "status", ...score fields } ]
        }
    """
    try:
        payload = request.get_json(force=True)

        if not isinstance(payload, list):
            return jsonify({
                'status': 'error',
                'message': 'Request body must be a JSON array of vendor objects.',
            }), 400

        if len(payload) == 0:
            return jsonify({'status': 'error', 'message': 'Array must not be empty.'}), 400

        if len(payload) > 50:
            return jsonify({
                'status': 'error',
                'message': f'Maximum 50 vendors per request. Received {len(payload)}.',
            }), 400

        results = []
        for i, vendor in enumerate(payload):
            try:
                res = score_one(vendor)
                results.append({'index': i, 'status': 'success', **res})
            except (ValueError, KeyError) as e:
                results.append({'index': i, 'status': 'error', 'message': str(e)})

        summary = {
            'total':    len(results),
            'approved': sum(1 for r in results if r.get('decision') == 'APPROVE'),
            'review':   sum(1 for r in results if r.get('decision') == 'REVIEW'),
            'rejected': sum(1 for r in results if r.get('decision') == 'REJECT'),
            'errors':   sum(1 for r in results if r.get('status')   == 'error'),
        }

        return jsonify({'status': 'success', 'summary': summary, 'results': results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    if MODEL_ERROR:
        print(f"\n  WARNING: {MODEL_ERROR}\n")
    app.run(debug=True, port=5000)
