import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os

MODEL_PATH = "model.joblib"

def create_synthetic_dataset(n_samples=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    heart_rate = rng.normal(75.0, 12.0, size=n_samples).astype(float)
    hr_variability = rng.normal(40.0, 15.0, size=n_samples).astype(float)
    blood_oxygen = rng.normal(97.0, 1.5, size=n_samples).astype(float)
    steps = rng.uniform(0.0, 100.0, size=n_samples).astype(float)
    risk_score = (0.03 * (heart_rate - 60.0)
                 + 0.04 * (40.0 - hr_variability)
                 + 0.05 * (97.0 - blood_oxygen)
                 + 0.0005 * steps)
    probs = 1 / (1 + np.exp(-risk_score))
    labels = (probs > 0.6).astype(int)
    df = pd.DataFrame({
        "heart_rate": heart_rate,
        "hr_variability": hr_variability,
        "blood_oxygen": blood_oxygen,
        "steps": steps,
        "label": labels
    })
    return df

def train_and_save(n_samples=5000):
    df = create_synthetic_dataset(n_samples)
    X = df[["heart_rate","hr_variability","blood_oxygen","steps"]].astype(float)
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestClassifier(n_estimators=200, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, (preds>0.5).astype(int))
    print(f"Trained model — AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
