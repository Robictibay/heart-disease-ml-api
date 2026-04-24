import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import HeartDiseaseInput

app = FastAPI(
    title="Heart Disease Prediction API",
    description="A FastAPI app that serves a scikit-learn heart disease prediction model.",
    version="1.0.0",
)

model = joblib.load("app/model.pkl")

@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running.",
        "docs": "Visit /docs to test the API."
    }

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    risk_label = "Heart disease risk detected" if int(prediction) == 1 else "No heart disease risk detected"

    return {
        "prediction": int(prediction),
        "risk_label": risk_label,
        "probability": round(float(probability), 4)
    }
