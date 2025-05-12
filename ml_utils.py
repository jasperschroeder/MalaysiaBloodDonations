import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input

import mlflow

import warnings
warnings.filterwarnings("ignore") 


def create_sequences_for_lstm(data: np.ndarray, seq_length: int = 30):
    X = [data[i:(i+seq_length)] for i in range(len(data) - seq_length)]
    y = [data[i + seq_length][0] for i in range(len(data) - seq_length)]
    return np.array(X), np.array(y)


def create_train_val_test(X: np.ndarray, y: np.ndarray, train_frac: float, val_frac: float):
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1")
    
    train_size = int(len(X) * train_frac)
    val_size = int(len(X) * val_frac)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:(train_size + val_size)], y[train_size:(train_size + val_size)]
    X_test, y_test = X[(train_size + val_size):], y[(train_size + val_size):]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_early_stopping(monitor:str, patience:int, restore_best_weights:bool=True, **kwargs):
    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, **kwargs)


def build_model(input_shape, units:int=30, activation:str|None='relu', dropout:float=0.2,
                optimizer:str='adam', loss:str='mse', metrics:list=['mae']):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, activation=activation, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def get_or_create_mlflow_experiment(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        return experiment_id
    return experiment.experiment_id


def run_experiment(X_train, y_train, X_val, y_val, units:int, activation: str, dropout: float, optimizer: str, experiment_id: str):

    mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

    with mlflow.start_run(experiment_id=experiment_id):
        early_stopping = create_early_stopping(monitor='val_loss', patience=15)

        model = build_model(X_train.shape[1:], units=units, activation=activation, dropout=dropout,
                                        optimizer=optimizer, loss='mse', metrics=['mae'])
            
        _ = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=500,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=None,
            shuffle=False
        )

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss[0]}, Validation MAE: {val_loss[1]}")
        
        mlflow.log_param("units", units)
        mlflow.log_param("activation", activation)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("dropout", dropout)
