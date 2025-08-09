import numpy as np
from sklearn.ensemble import IsolationForest

def detect_transaction_anomalies(df, contamination=0.005, random_state=42):
    # Use amount and hour as features for anomaly detection
    X = df[['amount','hour']].fillna(0).values
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    clf.fit(X)
    preds = clf.predict(X)
    df['anomaly_iforest'] = preds == -1
    return df
