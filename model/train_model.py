"""
MediTriage AI - Model Training Script
--------------------------------------
Trains a Random Forest classifier to predict patient triage urgency
(Emergency / Urgent / Routine) from vitals + symptom flags.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

RNG = np.random.default_rng(42)
N = 6000

def generate_patient():
    age = int(np.clip(RNG.normal(38, 20), 0, 95))
    heart_rate = int(np.clip(RNG.normal(85, 20), 40, 180))
    resp_rate = int(np.clip(RNG.normal(18, 5), 8, 45))
    spo2 = int(np.clip(RNG.normal(96, 4), 70, 100))
    temp_c = round(float(np.clip(RNG.normal(37.0, 1.0), 34.0, 41.5)), 1)
    systolic_bp = int(np.clip(RNG.normal(120, 20), 60, 200))
    chest_pain = RNG.choice([0, 1], p=[0.85, 0.15])
    breathing_difficulty = RNG.choice([0, 1], p=[0.85, 0.15])
    severe_bleeding = RNG.choice([0, 1], p=[0.95, 0.05])
    confusion = RNG.choice([0, 1], p=[0.92, 0.08])
    pregnant_complication = RNG.choice([0, 1], p=[0.96, 0.04])

    # --- Rule-based early warning score (clinically inspired, simplified) ---
    score = 0
    if heart_rate > 130 or heart_rate < 45: score += 3
    elif heart_rate > 110 or heart_rate < 55: score += 1

    if resp_rate > 30 or resp_rate < 9: score += 3
    elif resp_rate > 22: score += 1

    if spo2 < 88: score += 3
    elif spo2 < 93: score += 2
    elif spo2 < 96: score += 1

    if temp_c > 39.5 or temp_c < 35.0: score += 2
    elif temp_c > 38.3: score += 1

    if systolic_bp < 80 or systolic_bp > 190: score += 3
    elif systolic_bp < 95: score += 1

    if age > 70 or age < 2: score += 1

    score += chest_pain * 3
    score += breathing_difficulty * 3
    score += severe_bleeding * 4
    score += confusion * 3
    score += pregnant_complication * 3

    score += RNG.normal(0, 0.7)

    if score >= 7:
        label = "Emergency"
    elif score >= 3:
        label = "Urgent"
    else:
        label = "Routine"

    return {
        "age": age,
        "heart_rate": heart_rate,
        "resp_rate": resp_rate,
        "spo2": spo2,
        "temp_c": temp_c,
        "systolic_bp": systolic_bp,
        "chest_pain": chest_pain,
        "breathing_difficulty": breathing_difficulty,
        "severe_bleeding": severe_bleeding,
        "confusion": confusion,
        "pregnant_complication": pregnant_complication,
        "triage_level": label,
    }

def build_dataset(n=N):
    rows = [generate_patient() for _ in range(n)]
    return pd.DataFrame(rows)

def main():
    df = build_dataset()
    print("Class distribution:\n", df["triage_level"].value_counts())

    feature_cols = [
        "age", "heart_rate", "resp_rate", "spo2", "temp_c", "systolic_bp",
        "chest_pain", "breathing_difficulty", "severe_bleeding",
        "confusion", "pregnant_complication",
    ]
    X = df[feature_cols]
    y = df["triage_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Triage-specific weighting: a missed Emergency case is far costlier than
    # a false alarm, so we bias the classifier to favor Emergency recall.
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42,
        class_weight={"Emergency": 6, "Urgent": 2, "Routine": 1},
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    joblib.dump(clf, os.path.join(os.path.dirname(__file__), "triage_model.joblib"))
    df.to_csv(os.path.join(os.path.dirname(__file__), "synthetic_triage_data.csv"), index=False)
    print("\nSaved model to model/triage_model.joblib")

if __name__ == "__main__":
    main()
