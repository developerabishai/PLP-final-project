import streamlit as st
import pandas as pd
import requests
import os

API_URL = os.getenv("LIFELINE_API", "http://localhost:8000")

st.set_page_config(page_title="Lifeline AI Dashboard", layout="wide")
st.title("Lifeline AI — Dashboard 💙")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_sim = st.checkbox("Use ml/simulated_wearable.csv", value=True)

df = None
if uploaded:
    df = pd.read_csv(uploaded)
elif use_sim and os.path.exists("../ml/simulated_wearable.csv"):
    df = pd.read_csv("../ml/simulated_wearable.csv")

if df is not None:
    st.dataframe(df.tail(10))
    latest = df.iloc[-1].to_dict()
    st.subheader("Latest sample")
    st.json(latest)
    if st.button("Get risk prediction for latest sample"):
        payload = {k: float(latest[k]) for k in ["heart_rate","hr_variability","blood_oxygen","steps"]}
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=5.0)
            res.raise_for_status()
            j = res.json()
            st.metric("Predicted risk (0-1)", j.get("risk_score"))
            if j.get("risk_score",0.0)>0.6:
                st.warning("⚠️ Elevated risk detected. Consider seeking medical advice.")
            else:
                st.success("✅ Risk looks low for now.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
