import datetime
from fastapi import FastAPI
import pickle
import numpy as np
import os
import tensorflow

# Handle imports for both local and container environments

try:
    from .setup_and_validation import DonationPredictionRequest
except ImportError:
    from setup_and_validation import DonationPredictionRequest

# Check if running in container, adjust paths accordingly
if os.path.exists('/app/models'):
    # Running in container
    shared_dir = '/app/models'
else:
    # Running locally - use shared folder
    shared_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../shared')
    )

with open(os.path.join(shared_dir, 'x_scaler.pkl'), 'rb') as file:
    scaler_x = pickle.load(file)

with open(os.path.join(shared_dir, 'y_scaler.pkl'), 'rb') as file:
    scaler_y = pickle.load(file)

# Load the latest model from shared folder
model_files = [
    f for f in os.listdir(shared_dir)
    if f.startswith('model_') and f.endswith('.keras')
]
if not model_files:
    raise FileNotFoundError("No model files found in the shared directory")

# Sort by filename to get the latest (assuming date format in filename)
latest_model = sorted(model_files)[-1]
model_path = os.path.join(shared_dir, latest_model)
model = tensorflow.keras.models.load_model(model_path)


app = FastAPI(
    title="Blood Donation Prediction API",
    description="API for predicting blood donation amounts in Malaysia",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API is running"}


@app.post("/predict", summary="Predict blood donation amount")
def predict(request: DonationPredictionRequest):
    """
    Predict blood donation amount based on historical data and features
    """
    # Extract lags from request
    features = request.model_dump()
    seq_features = np.array([
        features['lag1'], features['lag2'], features['lag3'],
        features['lag4'], features['lag5'], features['lag6'], features['lag7']
    ]).reshape(-1, 1)

    # Extract other features from request
    prediction_date = datetime.datetime.strptime(features['nextday'], '%Y%m%d')

    weekday = prediction_date.isoweekday()
    month = prediction_date.month
    day_of_year = prediction_date.timetuple().tm_yday

    other_features = np.array([
        weekday, month, day_of_year,
        features['high_donation_holiday'], features['low_donation_holiday'],
        features['religion_or_culture_holiday'], features['other_holiday']
    ]).reshape(1, -1)

    # Scale features
    x_seq = scaler_y.transform(seq_features).reshape(1, 7, 1)
    x_feat = scaler_x.transform(other_features)

    # Make prediction
    pred_scaled = model.predict([x_seq, x_feat])
    prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

    return {
        "prediction": float(prediction),
        "input_features": {
            "lags": [features[f'lag{i}'] for i in range(1, 8)],
            "prediction_date": features['nextday'],
            "holiday_flags": {
                "high_donation": features['high_donation_holiday'],
                "low_donation": features['low_donation_holiday'],
                "religion_culture": features['religion_or_culture_holiday'],
                "other": features['other_holiday']
            }
        }
    }
