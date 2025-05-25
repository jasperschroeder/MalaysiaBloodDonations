import pickle
from fastapi import FastAPI
import ml_utils
import numpy as np
from setup_and_validation import DonationPredictionRequest
import datetime


with open('x_scaler.pkl', 'rb') as file:
    scaler_x = pickle.load(file)
    
with open('y_scaler.pkl', 'rb') as file:
    scaler_y = pickle.load(file)

model = ml_utils.get_best_model(
    ml_utils.get_or_create_mlflow_experiment(experiment_name='NewTest5'),
    'metrics.val_loss'
)
   
app = FastAPI(title="Blood Donation Prediction API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", summary="Predict blood donation amount")
def predict(request: DonationPredictionRequest):
    
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

    return {"prediction": float(prediction)}
