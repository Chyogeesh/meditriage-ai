# MediTriage AI

**Idea2Impact 2026 · Theme: Crisis Management, HealthTech & Emergency Response**

An AI-powered triage assistant that helps community health workers in low-resource
and rural settings prioritize patients by urgency — instantly, explainably, and
without needing a doctor on-site.

---

## The Problem

In rural clinics and disaster-response camps, a single community health worker
(often with basic paramedical training, not a physician) may face a queue of
patients with no way to quickly judge who needs help *right now* versus who can
safely wait. Delays in recognizing a deteriorating patient (sepsis, respiratory
failure, internal bleeding, obstetric emergencies) are a leading cause of
preventable deaths in under-resourced healthcare settings — and connectivity to
call a doctor for advice is often unreliable or unavailable.

**Who is affected:** rural/community health workers, ASHA workers, disaster and
camp medical volunteers, and ultimately the patients whose survival depends on
being triaged correctly and fast.

**Why it matters:** early-warning-score-based triage is a proven clinical
practice in hospitals, but it's rarely available to frontline workers without
formal medical training or digital tools. A lightweight, explainable AI tool
can put that same decision-support in their hands on a basic phone or laptop.

## The Solution

MediTriage AI takes a patient's vitals (heart rate, respiration, SpO2,
temperature, blood pressure, age) plus key symptom flags (chest pain,
breathing difficulty, severe bleeding, confusion, pregnancy complications) and
returns:

1. **A triage level** — Emergency / Urgent / Routine
2. **A confidence score** and full probability breakdown
3. **A plain-language explanation** of the top factors driving that
   classification (powered by SHAP), so the health worker understands *why*,
   not just *what*
4. **A concrete recommended action**

### How AI is used (functional core, not a wrapper)

- A **Random Forest classifier** (scikit-learn) is trained on a clinically-informed
  synthetic dataset (`model/train_model.py`) built around real early-warning-score
  logic (heart rate, respiratory rate, SpO2, temperature, BP thresholds combined
  with critical symptom flags).
- The model is deliberately **weighted to favor recall on the Emergency class**
  (class_weight `{Emergency: 6, Urgent: 2, Routine: 1}`) — in real triage, a missed
  emergency is far more costly than a false alarm.
- **SHAP (TreeExplainer)** computes per-patient feature contributions at inference
  time, so every prediction ships with an explanation, not just a label.
- Runs entirely on a lightweight trained model — **no external LLM API calls
  required**, meaning it can run near-offline once deployed, which matters for
  low-connectivity clinics.

### Why this is effective

- Narrow, concrete problem (patient triage) solved deeply, not a generic
  "AI for healthcare" pitch.
- Explainability directly addresses trust — frontline workers are (rightly)
  wary of black-box tools; SHAP factors let them sanity-check the AI against
  what they're seeing in front of them.
- Recall-biased toward the highest-stakes class, matching real clinical
  priorities rather than optimizing for raw accuracy.

## Tech Stack

- **Backend:** Python, FastAPI, scikit-learn (RandomForestClassifier), SHAP, joblib
- **Frontend:** HTML/CSS/vanilla JS (fast to deploy, works on any static host)
- **Model:** trained offline, shipped as a `.joblib` artifact, loaded at API startup

## Project Structure

```
meditriage-ai/
├── backend/
│   ├── app.py               # FastAPI app: /predict endpoint with SHAP explanation
│   └── requirements.txt
├── model/
│   ├── train_model.py       # Synthetic data generation + model training
│   ├── triage_model.joblib  # Trained model artifact
│   └── synthetic_triage_data.csv
├── frontend/
│   └── index.html           # Single-file UI, calls the backend API
├── render.yaml               # One-click backend deploy config for Render
└── README.md
```

## Setup Instructions (local)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd meditriage-ai

# 2. (Optional) retrain the model — a trained model is already included
cd model
pip install -r ../backend/requirements.txt
python3 train_model.py
cd ..

# 3. Run the backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# 4. Open the frontend
# Open frontend/index.html directly in a browser, or serve it:
cd ../frontend
python3 -m http.server 5500
# then visit http://localhost:5500
# Set the "API base URL" field in the UI to http://localhost:8000
```

## Deployment (for submission)

**Backend (Render — free tier):**
1. Push this repo to GitHub (public).
2. On [render.com](https://render.com), New → Web Service → connect the repo.
3. Render will detect `render.yaml` and auto-configure the build/start commands.
4. Once live, copy the Render URL (e.g. `https://meditriage-ai-backend.onrender.com`).

**Frontend (Vercel / Netlify / GitHub Pages):**
1. Deploy the `frontend/` folder as a static site (drag-and-drop works on
   Netlify, or use `vercel --prod` from inside `frontend/`).
2. In the deployed UI, paste your Render backend URL into the "API base URL" field.

> Note: Render's free tier spins down after inactivity — the first request
> after idling can take ~30-50 seconds to wake up. Mention this in your demo
> video so judges aren't surprised by the delay, or ping the backend URL a
> minute before recording.

## Example Request

```bash
curl -X POST https://<your-backend-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 68, "heart_rate": 138, "resp_rate": 32, "spo2": 86, "temp_c": 39.8,
    "systolic_bp": 78, "chest_pain": true, "breathing_difficulty": true,
    "severe_bleeding": false, "confusion": true, "pregnant_complication": false
  }'
```

Response:
```json
{
  "triage_level": "Emergency",
  "confidence": 0.814,
  "probabilities": {"Emergency": 0.814, "Routine": 0.0, "Urgent": 0.186},
  "top_factors": [
    {"feature": "Breathing difficulty", "value": 1, "impact": 0.18, "direction": "increased"},
    {"feature": "Chest pain", "value": 1, "impact": 0.166, "direction": "increased"},
    {"feature": "Confusion / altered consciousness", "value": 1, "impact": 0.133, "direction": "increased"},
    {"feature": "Systolic blood pressure", "value": 78, "impact": 0.066, "direction": "increased"}
  ],
  "recommendation": "Immediate attention required. Escalate to nearest physician/hospital now."
}
```

## Model Performance

Trained on a 6,000-row clinically-informed synthetic dataset, 80/20 split:

| Class     | Precision | Recall | F1   |
|-----------|-----------|--------|------|
| Emergency | 0.82      | 0.59   | 0.68 |
| Urgent    | 0.79      | 0.87   | 0.82 |
| Routine   | 0.92      | 0.89   | 0.90 |

Overall accuracy: **86%**

## Future Work

- Swap synthetic training data for a real, de-identified clinical dataset (e.g. MIMIC-IV vitals)
- Add offline-first PWA support for true no-connectivity use
- Multi-language UI for regional health workers
- SMS/USSD interface for feature-phone access in the field

## Team

Individual submission — Idea2Impact 2026 Online Hackathon.
