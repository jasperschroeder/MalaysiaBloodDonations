from typing import Dict, Iterator, Optional, Tuple

import mlflow.tensorflow
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Concatenate, Dense, Dropout, GRU, Input, LSTM, SimpleRNN
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import mlflow
import absl.logging
import warnings

# Support both package imports (src.donations.ml_utils) and top-level
# imports (ml_utils, as used by tests via pytest.ini pythonpath).
try:
    from .evaluation import compute_regression_metrics
except ImportError:
    from evaluation import compute_regression_metrics

absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")


def create_early_stopping(monitor: str, patience: int, restore_best_weights: bool = True, **kwargs):
    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, **kwargs)


def get_or_create_mlflow_experiment(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        return experiment_id
    return experiment.experiment_id


def train_val_test_split_feature_data(
    X_seq: np.ndarray,
    X_features: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float
):
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1")

    train_size = int(len(X_seq) * train_frac)
    val_size = int(len(X_seq) * val_frac)

    X_seq_train = X_seq[:train_size]
    X_seq_val = X_seq[train_size:(train_size + val_size)]
    X_seq_test = X_seq[(train_size + val_size):]

    X_features_train = X_features[:train_size]
    X_features_val = X_features[train_size:(train_size + val_size)]
    X_features_test = X_features[(train_size + val_size):]

    y_train = y[:train_size]
    y_val = y[train_size:(train_size + val_size)]
    y_test = y[(train_size + val_size):]

    return X_seq_train, X_seq_val, X_seq_test, X_features_train, X_features_val, X_features_test, y_train, y_val, y_test


def build_seq_model(
    seq_shape, features_shape, seq_type: str, seq_units: int, dense_units: int, activation: str, dropout: float,
    optimizer: str, learning_rate: float = 0.001, loss: str = 'mse', metrics: list = ['mae']
):

    # inputs
    seq_input = Input(shape=(seq_shape, 1))
    features_input = Input(shape=(features_shape,))

    # seq branch
    if seq_type == 'LSTM':
        x_seq = LSTM(seq_units, activation=activation, return_sequences=False)(seq_input)
    elif seq_type == 'GRU':
        x_seq = GRU(seq_units, activation=activation, return_sequences=False)(seq_input)
    else:
        x_seq = SimpleRNN(seq_units, activation=activation, return_sequences=False)(seq_input)
    x_seq = Dropout(dropout)(x_seq)

    # Dense branch
    x = Concatenate()([x_seq, features_input])
    x = Dense(dense_units, activation='relu')(x)  # Setting relu as default activation for dense layers
    x = Dropout(dropout)(x)
    output = Dense(1)(x)

    # Selecting optimizer with learning rate
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    model = Model(inputs=[seq_input, features_input], outputs=output)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model


def run_experiment(
    X_seq_train, X_features_train, y_train, X_seq_val, X_features_val, y_val, seq_type: str, seq_units: int,
    dense_units: int, activation: str, dropout: float, optimizer: str, learning_rate: float, batch_size: int,
    experiment_id: str, extra_params: Optional[Dict] = None
):
    """Train one sequence model and log a complete record to MLflow.

    Returns the Keras training history so callers can inspect convergence.
    """

    mlflow.tensorflow.autolog(log_models=True, log_datasets=False, silent=True)

    with mlflow.start_run(experiment_id=experiment_id):
        early_stopping = create_early_stopping(monitor='val_loss', patience=25)

        model = build_seq_model(
            seq_shape=X_seq_train.shape[1], features_shape=X_features_train.shape[1], seq_type=seq_type,
            seq_units=seq_units, dense_units=dense_units, activation=activation, dropout=dropout,
            optimizer=optimizer, learning_rate=learning_rate
        )

        history = model.fit(
            [X_seq_train, X_features_train],
            y_train,
            validation_data=([X_seq_val, X_features_val], y_val),
            epochs=1000,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0,
            shuffle=False
        )

        mlflow.log_param("seq_type", seq_type)
        mlflow.log_param("seq_units", seq_units)
        mlflow.log_param("dense_units", dense_units)
        mlflow.log_param("activation", activation)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        if extra_params:
            mlflow.log_params({k: str(v) for k, v in extra_params.items()})

        mlflow.log_metric("epochs_trained", len(history.history["loss"]))
        mlflow.log_metric("final_train_loss", float(history.history["loss"][-1]))
        mlflow.log_metric("final_val_loss", float(history.history["val_loss"][-1]))

    return history


def log_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None,
    prefix: str = "test"
) -> Dict[str, float]:
    """Compute the standard metric bundle and log it to the active MLflow run."""
    metrics = compute_regression_metrics(y_true, y_pred, y_prev=y_prev)
    mlflow.log_metrics({f"{prefix}_{name}": value for name, value in metrics.items()})
    return metrics


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    val_size: Optional[int] = None,
    gap: int = 0,
    min_train_size: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) expanding-window time-series folds.

    The validation window of each fold always lies strictly after the
    training window, with an optional purge gap between them (use at least
    the sequence window size so lag features cannot leak across the fold
    boundary).
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    if min_train_size is None:
        min_train_size = max(int(n_samples * 0.5), 1)
    if val_size is None:
        val_size = max((n_samples - min_train_size - gap) // n_splits, 1)

    if min_train_size + gap + val_size > n_samples:
        raise ValueError(
            "Not enough samples for the requested split configuration "
            f"(n_samples={n_samples}, min_train_size={min_train_size}, "
            f"gap={gap}, val_size={val_size})"
        )

    for i in range(n_splits):
        train_end = min_train_size + i * val_size
        val_start = train_end + gap
        val_end = min(val_start + val_size, n_samples)
        if val_end - val_start < 1:
            break
        yield np.arange(0, train_end), np.arange(val_start, val_end)


def scale_train_val(
    X_seq_train: np.ndarray,
    X_features_train: np.ndarray,
    y_train: np.ndarray,
    X_seq_val: Optional[np.ndarray] = None,
    X_features_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    scaler_cls=None
) -> Dict:
    """Fit scalers on training data only and transform train/val partitions.

    Prevents the data leakage caused by fitting scalers before the split.
    The sequence array is scaled with the target (y) scaler, matching the
    original training pipeline.

    Returns a dict with scaled arrays ('X_seq_train', 'X_features_train',
    'y_train', and 'X_seq_val'/'X_features_val'/'y_val' when provided) plus
    the fitted 'y_scaler' and 'x_scaler' for later inverse transforms.
    """
    if scaler_cls is None:
        from sklearn.preprocessing import RobustScaler
        scaler_cls = RobustScaler

    window_size = X_seq_train.shape[1]

    y_scaler = scaler_cls()
    x_scaler = scaler_cls()

    # Fit the target scaler on y_train first (matching the original
    # pipeline), then reuse it for the donation-scale sequence lags.
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    result = {
        "X_seq_train": y_scaler.transform(
            X_seq_train.reshape(-1, 1)
        ).reshape(-1, window_size, 1),
        "X_features_train": x_scaler.fit_transform(X_features_train),
        "y_train": y_train_scaled,
        "y_scaler": y_scaler,
        "x_scaler": x_scaler,
    }

    if X_seq_val is not None:
        result["X_seq_val"] = y_scaler.transform(
            X_seq_val.reshape(-1, 1)
        ).reshape(-1, window_size, 1)
    if X_features_val is not None:
        result["X_features_val"] = x_scaler.transform(X_features_val)
    if y_val is not None:
        result["y_val"] = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    return result


def get_best_model(experiment_id, metric: str = "metrics.val_loss"):
    model_dfs = mlflow.search_runs(experiment_id)
    best_run_id = model_dfs.sort_values(metric, ascending=True)['run_id'].iloc[0]
    return mlflow.tensorflow.load_model(f"runs:/{best_run_id}/model")
