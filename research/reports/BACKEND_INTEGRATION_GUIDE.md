# ANCHOR AI HEALTHCARE — BACKEND INTEGRATION & MODEL SERVING GUIDE
**Document ID:** `ANCHOR-BACKEND-INTEGRATION-GUIDE-v1`  
**Execution Timestamp:** 2026-11-15T01:34:10+05:30  
**Target Audience:** Backend Engineer & Frontend Engineer  
**Working Branch:** `mohit`  

---

## 1. Executive Summary

This integration contract bridges the gap between the verified Machine Learning model and the upcoming backend API service.

### Fundamental Scientific Rule:
The machine learning pipeline **does not** accept resting heart rate, sleep duration, or step count. These are simulated UI placeholders. The real model operates on **intake demographics**, **trailing past-use lapse counts**, and **ecological momentary assessment (EMA) self-reports**.

---

## 2. Real-World Field Mapping vs. Required Redesign

| Existing Frontend Field (`index.html`) | Proposed Action | Mapped Model Feature | Underlying Construct & Scale |
|:---|:---|:---|:---|
| **Resting Heart Rate (`#heartRateInput`)** | **Deprecated / Decoupled** | None (Not used in ML) | Kept as optional display-only wearable metric if desired, but NOT passed to risk model. |
| **Sleep Duration (`#sleepInput`)** | **Deprecated / Decoupled** | None (Not used in ML) | Kept as optional lifestyle log; NOT passed to risk model. |
| **Mood / Steadiness (`#moodRange`)** | **Keep & Calibrate** | `ema_3.p0.l24.rrecent_response` | Maps to psychological valence / distress (rescaled $1–10 \to 1–11$). |
| **Urge to Drink / Craving** | **NEW REQUIRED INPUT** | `ema_2.p0.l24.rrecent_response` | Alcohol craving intensity slider ($0–12$). |
| **Perceived Stress / Conflict** | **NEW REQUIRED INPUT** | `ema_6.p0.l24.rrecent_response` | Perceived stress / interpersonal friction ($1–11$). |
| **Risky Environment / People** | **NEW REQUIRED INPUT** | `ema_7.p0.l24.rrecent_response` | Environmental alcohol exposure / cues ($1–11$). |
| **Confidence to Stay Sober** | **NEW REQUIRED INPUT** | `ema_10.p0.l24.rrecent_response` | Self-efficacy / confidence to avoid drinking ($1–11$). |
| **Any Alcohol Consumed Today?** | **NEW REQUIRED INPUT** | Updates `p12`, `p24`, `p168` | Binary radio button (`Yes` / `No`) + drink count. |

---

## 3. The Cold-Start Limitation & Warm-up Timeline

A new patient registering on Anchor begins with **zero days of historical EMA data**.

```
Patient Onboarding Timeline & Model Readiness:

Day 0 (Registration)          Day 1–2 (Early Data)           Day 3+ (Full Warm-up)
├─────────────────────────────┼─────────────────────────────┼─────────────────────────────►
• Static Intake Only          • Early Check-ins             • Stabilized 24h Early Warning
• AUROC: 0.504 (Chance)       • Partial Imputation          • AUROC: 0.881 | AUPRC: 0.459
• UI Warning: "Calibrating"   • Baseline Establishing       • Active Risk Alerts Enabled
```

### Cold-Start Guardrails for Backend:
1. **Days 1 to 2 (Hours $0–48$):** The backend MUST return `status = "calibrating"`. Display a clear, empathetic prompt on the dashboard: *"Anchor is learning your personal baseline rhythm. Risk estimates will be active after 3 days of check-ins."*
2. **Days 3 onwards (Hours $\ge 72$):** The backend generates active risk scores using `anchor_lapse_model_v1.joblib`.

---

## 4. Standard Python Inference Interface

The serialized model is encapsulated behind a clean, deterministic Python function:

```python
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

# Load serialized pipeline artifact
MODEL_PATH = "research/models/anchor_lapse_model_v1.joblib"
pipeline = joblib.load(MODEL_PATH)

def predict_risk(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates user check-in payload, evaluates calibrated lapse risk,
    and returns prospective early-warning probability for the next week.
    """
    # 1. Validate required historical duration
    hours_active = input_payload.get("hours_since_enrollment", 0)
    if hours_active < 72:
        return {
            "status": "calibrating",
            "message": "Anchor requires at least 72 hours of baseline check-ins to activate predictive risk.",
            "risk_score": None,
            "horizon": "24h_lead_time"
        }

    # 2. Convert dictionary to single-row DataFrame matching training schema
    df_input = pd.DataFrame([input_payload["features"]])
    
    # 3. Predict calibrated probability
    prob = float(pipeline.predict_proba(df_input)[0, 1])
    risk_score_100 = int(round(prob * 100))
    
    # 4. Classify risk tier based on validated Youden's J threshold (0.08)
    if prob >= 0.25:
        tier = "HIGHER_SUPPORT"
        guidance = "Your readings suggest connecting with your support circle today."
    elif prob >= 0.08:
        tier = "WATCHFUL"
        guidance = "A few subtle signals are elevated. Take gentle steps to protect your rhythm."
    else:
        tier = "STEADY"
        guidance = "Your baseline signals remain steady and supportive."

    return {
        "status": "active",
        "predicted_probability": round(prob, 4),
        "risk_score": risk_score_100,
        "risk_tier": tier,
        "guidance_message": guidance,
        "prediction_horizon": "24_hours_ahead_of_next_7_days",
        "model_version": "anchor_v1.0.0_hgb_compact"
    }
```

---

## 5. API Endpoint Specifications

### `POST /api/v1/checkin`
Submits a daily check-in and updates the trailing database counts.

**Request Payload:**
```json
{
  "user_id": "usr_99812",
  "timestamp": "2026-11-15T20:00:00Z",
  "craving_level": 4,
  "stress_level": 6,
  "risky_environment": 3,
  "abstinence_confidence": 8,
  "alcohol_consumed_past_24h": 0
}
```

**Response Payload:**
```json
{
  "checkin_id": "chk_44021",
  "status": "saved",
  "risk_assessment": {
    "status": "active",
    "risk_score": 19,
    "risk_tier": "WATCHFUL",
    "guidance": "A few subtle signals are elevated. Take gentle steps to protect your rhythm."
  }
}
```
