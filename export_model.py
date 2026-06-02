"""
Export the best credit scoring model (Logistic Regression) as a joblib bundle.
Replicates the exact training pipeline from credit_scoring_v2.ipynb.
Output: model/credit_scorer.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data', 'alt_data_expanded_v2.csv')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_scorer.pkl')

FEATURES = [
    'credit_score', 'profit_margin', 'expense_ratio',
    'years_in_business', 'mobile_money_txn_count',
    'utility_payment_score', 'supplier_credit_history',
    'log_revenue', 'log_wallet', 'business_stability',
    'composite_risk', 'payment_score',
    'gender_enc', 'region_enc', 'btype_enc', 'age'
]

THRESHOLDS = {
    'approve_below': 0.30,
    'review_below':  0.60,
}


def engineer_features(df):
    df = df.copy()
    df['expense_ratio']      = df['monthly_expenses_xaf'] / df['monthly_revenue_xaf']
    df['business_stability'] = (df['years_in_business'] >= 3).astype(int)
    df['log_revenue']        = np.log1p(df['monthly_revenue_xaf'])
    df['log_wallet']         = np.log1p(df['avg_wallet_balance_xaf'])
    df['composite_risk']     = df['expense_ratio'] * (1 - df['profit_margin'])
    df['payment_score']      = df['utility_payment_score'] * df['supplier_credit_history']
    return df


def encode_categoricals(df):
    df = df.copy()
    encoders = {}
    for raw, enc in [('gender', 'gender_enc'), ('region', 'region_enc'), ('business_type', 'btype_enc')]:
        le = LabelEncoder()
        df[enc] = le.fit_transform(df[raw])
        encoders[raw] = le
    return df, encoders


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} records loaded.")

    df, encoders = encode_categoricals(engineer_features(df))

    X = df[FEATURES]
    y = df['loan_defaulted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("Training Logistic Regression ...")
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    lr_model.fit(X_train_sc, y_train)

    print("Calibrating with Platt scaling (5-fold CV) ...")
    cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    calib_model = CalibratedClassifierCV(lr_model, method='sigmoid', cv=5)
    calib_model.fit(X_train_sc, y_train)

    calib_prob = calib_model.predict_proba(X_test_sc)[:, 1]
    calib_pred = (calib_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, calib_pred)
    auc = roc_auc_score(y_test, calib_prob)
    f1  = f1_score(y_test, calib_pred)
    print(f"\nCalibrated model — Test Accuracy: {acc:.4f}  AUC: {auc:.4f}  F1: {f1:.4f}")

    bundle = {
        'model':      calib_model,
        'scaler':     scaler,
        'encoders':   encoders,
        'features':   FEATURES,
        'thresholds': THRESHOLDS,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel bundle saved -> {MODEL_PATH}")
    print("Bundle keys: model, scaler, encoders (gender/region/business_type), features, thresholds")


if __name__ == '__main__':
    main()
