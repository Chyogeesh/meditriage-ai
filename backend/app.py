"""
MediTriage AI — FastAPI backend
--------------------------------
Serves:
  GET  /              -> health check
  POST /predict        -> triage prediction + SHAP-based explanation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import shap
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "triage_model.joblib")

app = FastAPI(title="MediTriage AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(clf)

FEATURE_COLS = [
    "age", "heart_rate", "resp_rate", "spo2", "temp_c", "systolic_bp",
    "chest_pain", "breathing_difficulty", "severe_bleeding",
    "confusion", "pregnant_complication",
]

FEATURE_LABELS = {
    "age": "Age",
    "heart_rate": "Heart rate",
    "resp_rate": "Respiratory rate",
    "spo2": "Blood oxygen (SpO2)",
    "temp_c": "Temperature",
    "systolic_bp": "Systolic blood pressure",
    "chest_pain": "Chest pain",
    "breathing_difficulty": "Breathing difficulty",
    "severe_bleeding": "Severe bleeding",
    "confusion": "Confusion / altered consciousness",
    "pregnant_complication": "Pregnancy complication",
}


class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    heart_rate: int = Field(..., ge=20, le=250)
    resp_rate: int = Field(..., ge=4, le=60)
    spo2: int = Field(..., ge=50, le=100)
    temp_c: float = Field(..., ge=30.0, le=43.0)
    systolic_bp: int = Field(..., ge=40, le=260)
    chest_pain: bool = False
    breathing_difficulty: bool = False
    severe_bleeding: bool = False
    confusion: bool = False
    pregnant_complication: bool = False


@app.get("/")
def health():
    return {"status": "ok", "service": "MediTriage AI"}


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        row = {
            "age": patient.age,
            "heart_rate": patient.heart_rate,
            "resp_rate": patient.resp_rate,
            "spo2": patient.spo2,
            "temp_c": patient.temp_c,
            "systolic_bp": patient.systolic_bp,
            "chest_pain": int(patient.chest_pain),
            "breathing_difficulty": int(patient.breathing_difficulty),
            "severe_bleeding": int(patient.severe_bleeding),
            "confusion": int(patient.confusion),
            "pregnant_complication": int(patient.pregnant_complication),
        }
        X = pd.DataFrame([row])[FEATURE_COLS]

        proba = clf.predict_proba(X)[0]
        classes = clf.classes_
        pred_idx = int(np.argmax(proba))
        pred_label = classes[pred_idx]
        confidence = float(proba[pred_idx])

        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_idx][0]
        else:
            class_shap = shap_values[0, :, pred_idx]

        contributions = sorted(
            zip(FEATURE_COLS, class_shap),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:4]

        explanation = []
        for feat, val in contributions:
            direction = "increased" if val > 0 else "decreased"
            explanation.append({
                "feature": FEATURE_LABELS[feat],
                "value": row[feat],
                "impact": round(float(val), 3),
                "direction": direction,
            })

        return {
            "triage_level": pred_label,
            "confidence": round(confidence, 3),
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
            "top_factors": explanation,
            "recommendation": _recommendation(pred_label),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _recommendation(level: str) -> str:
    if level == "Emergency":
        return "Immediate attention required. Escalate to nearest physician/hospital now."
    if level == "Urgent":
        return "Should be seen soon. Monitor closely and prioritize over routine cases."
    return "Stable — manage with routine care and standard follow-up."
