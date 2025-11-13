# Lifeline AI - HealthTech Python Project

**Lifeline AI** is a next-generation HealthTech project that predicts potential health risks using wearable data and connects users with verified doctors. Built using Python, FastAPI, and Streamlit, it combines machine learning with an interactive dashboard for real-time risk assessment.

---

## 🚀 Features

- **AI Symptom Analyzer**: Processes wearable data to calculate risk scores.
- **Predictive Modeling**: Machine learning model trained on synthetic health datasets.
- **Real-time Dashboard**: Streamlit interface to visualize latest readings and predictions.
- **Doctor Connect**: Option to connect with medical professionals based on predicted risk.
- **Wearable Data Simulation**: Generate realistic sample data for testing.

---

## 📂 Project Structure

```
lifeline-ai/
├── backend/
│   └── app.py
├── ml/
│   ├── train_model.py
│   ├── simulate_wearable.py
│   └── model.joblib
├── dashboard/
│   └── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Setup Instructions

1. Clone the repository:
```
git clone https://github.com/YOUR_USERNAME/lifeline-ai.git
cd lifeline-ai
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Train the ML model:
```
python ml/train_model.py
```

5. Run the backend API:
```
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

6. Run the Streamlit dashboard:
```
streamlit run dashboard/streamlit_app.py
```

7. Optionally, simulate wearable data:
```
python ml/simulate_wearable.py
```

---

## 🛠 Tech Stack

- **Backend**: FastAPI
- **Frontend / Dashboard**: Streamlit
- **Machine Learning**: Python, scikit-learn, joblib
- **Deployment**: Docker compatible

---

## 👤 Author

**Abishai (bebs 😍)** — AI Developer & Innovator