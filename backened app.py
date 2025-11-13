from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

MODEL_PATH = "../ml/model.joblib"
app = FastAPI(title="Lifeline AI Backend")

class WearableSample(BaseModel):
    heart_rate: float = Field(..., example=72.5)
    hr_variability: float = Field(..., example=45.2)
    blood_oxygen: float = Field(..., example=97.5)
    steps: float = Field(..., example=12.0)

_model = None
def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run ml/train_model.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.post("/ingest")
def ingest_sample(sample: WearableSample):
    return {"status": "ok", "received": sample.dict()}

@app.post("/predict")
def predict_risk(sample: WearableSample):
    model = load_model()
    X = np.array([[float(sample.heart_rate),
                   float(sample.hr_variability),
                   float(sample.blood_oxygen),
                   float(sample.steps)]], dtype=float)
    proba = model.predict_proba(X)[0][1]
    return {"risk_score": float(proba)}

@app.get("/")
def root():
    return {"service": "Lifeline AI backend", "model_path": MODEL_PATH}
