import numpy as np

class PredictionPipeline:
    """End-to-end inference pipeline for customer churn scoring."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        # Placeholder for pickled scikit-learn or XGBoost model
        return {"loaded": True}

    def preprocess(self, input_features: dict) -> np.ndarray:
        """Preprocesses input feature dictionary into scaled feature vector."""
        tenure = input_features.get("tenure_months", 0) / 72.0
        monthly = input_features.get("monthly_charges", 0.0) / 120.0
        total = input_features.get("total_charges", 0.0) / 8000.0
        return np.array([[tenure, monthly, total]])

    def predict(self, input_features: dict) -> dict:
        """
        Executes preprocessing and model prediction.
        Returns predicted label and churn probability.
        """
        features = self.preprocess(input_features)
        # Mock calculation
        prob = float(np.mean(features))
        is_churn = bool(prob > 0.5)
        return {
            "is_churn": is_churn,
            "probability": round(prob, 4)
        }
