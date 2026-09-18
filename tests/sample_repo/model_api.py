from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictionPipeline
from src.auth.security import verify_jwt_token

app = FastAPI(title="Customer Churn Prediction API")
pipeline = PredictionPipeline(model_path="models/churn_model.pkl")

class UserInput(BaseModel):
    customer_id: str
    tenure_months: int
    monthly_charges: float
    total_charges: float

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "churn-prediction"}

@app.post("/predict")
def predict(data: UserInput, user=Depends(verify_jwt_token)):
    """
    Main prediction endpoint. Receives UserInput, invokes
    the PredictionPipeline, and returns the churn prediction probability.
    """
    prediction = pipeline.predict(data.dict())
    return {
        "customer_id": data.customer_id,
        "churn_prediction": prediction["is_churn"],
        "probability": prediction["probability"]
    }
