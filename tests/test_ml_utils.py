import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from src.donations.ml_utils import (  # noqa: E402
    create_early_stopping,
    get_or_create_mlflow_experiment,
    train_val_test_split_feature_data,
    build_seq_model,
    get_best_model,
    walk_forward_splits,
    scale_train_val,
    log_regression_metrics
)


def test_create_early_stopping_basic():
    cb = create_early_stopping(monitor='val_loss', patience=5)
    assert isinstance(cb, EarlyStopping)
    assert cb.monitor == 'val_loss'
    assert cb.patience == 5
    assert cb.restore_best_weights is True


def test_create_early_stopping_kwargs():
    cb = create_early_stopping(monitor='val_acc', patience=3, restore_best_weights=False, min_delta=0.01)
    assert cb.monitor == 'val_acc'
    assert cb.patience == 3
    assert cb.restore_best_weights is False
    assert cb.min_delta == 0.01


def test_get_or_create_mlflow_experiment_exists(monkeypatch):
    class DummyExperiment:
        experiment_id = "123"

    def dummy_get_experiment_by_name(name):
        return DummyExperiment()
    monkeypatch.setattr("ml_utils.mlflow.get_experiment_by_name", dummy_get_experiment_by_name)
    monkeypatch.setattr("ml_utils.mlflow.create_experiment", lambda name: "should_not_be_called")
    result = get_or_create_mlflow_experiment("existing_experiment")
    assert result == "123"


def test_get_or_create_mlflow_experiment_creates(monkeypatch):
    def dummy_get_experiment_by_name(name):
        return None

    def dummy_create_experiment(name):
        return "456"
    monkeypatch.setattr("ml_utils.mlflow.get_experiment_by_name", dummy_get_experiment_by_name)
    monkeypatch.setattr("ml_utils.mlflow.create_experiment", dummy_create_experiment)
    result = get_or_create_mlflow_experiment("new_experiment")
    assert result == "456"


def test_train_val_test_split_feature_data_happy_path():
    import numpy as np
    X_seq = np.arange(20).reshape(20, 1)
    X_features = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    train_frac = 0.5
    val_frac = 0.25
    result = train_val_test_split_feature_data(X_seq, X_features, y, train_frac, val_frac)
    (X_seq_train, X_seq_val, X_seq_test, X_features_train,
     X_features_val, X_features_test, y_train, y_val, y_test) = result

    assert X_seq_train.shape[0] == 10
    assert X_seq_val.shape[0] == 5
    assert X_seq_test.shape[0] == 5
    assert X_features_train.shape[0] == 10
    assert X_features_val.shape[0] == 5
    assert X_features_test.shape[0] == 5
    assert y_train.shape[0] == 10
    assert y_val.shape[0] == 5
    assert y_test.shape[0] == 5


def test_train_val_test_split_feature_data_invalid_sum():
    import numpy as np
    X_seq = np.zeros((10, 1))
    X_features = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError) as excinfo:
        train_val_test_split_feature_data(X_seq, X_features, y, 0.7, 0.3)
    assert "train_frac + val_frac must be less than 1" in str(excinfo.value)


def test_train_val_test_split_feature_data_invalid_range():
    import numpy as np
    X_seq = np.zeros((10, 1))
    X_features = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError) as excinfo:
        train_val_test_split_feature_data(X_seq, X_features, y, 0, 0.5)
    assert "train_frac and val_frac must be between 0 and 1" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        train_val_test_split_feature_data(X_seq, X_features, y, 0.5, 1)
    assert "train_frac + val_frac must be less than 1" in str(excinfo.value)


def test_build_seq_model_lstm():
    model = build_seq_model(
        seq_shape=5,
        features_shape=3,
        seq_type='LSTM',
        seq_units=4,
        dense_units=2,
        activation='tanh',
        dropout=0.1,
        optimizer='adam',
        learning_rate=0.01,
        loss='mse',
        metrics=['mae']
    )
    # Check model input/output shapes
    assert len(model.inputs) == 2
    assert model.inputs[0].shape[1] == 5
    assert model.inputs[1].shape[1] == 3
    assert model.output_shape[-1] == 1

    # Check optimizer and loss
    assert model.loss == 'mse'

    # Fix: check model.metrics_names for 'mae'
    assert 'mae' in model._compile_metrics._user_metrics


def test_build_seq_model_gru():
    model = build_seq_model(
        seq_shape=4,
        features_shape=2,
        seq_type='GRU',
        seq_units=3,
        dense_units=2,
        activation='relu',
        dropout=0.2,
        optimizer='rmsprop',
        learning_rate=0.005,
        loss='mae',
        metrics=['mse']
    )
    assert model.inputs[0].shape[1] == 4
    assert model.inputs[1].shape[1] == 2
    assert model.loss == 'mae'


def test_build_seq_model_rnn_custom_optimizer():
    import tensorflow as tf
    custom_opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model = build_seq_model(
        seq_shape=3,
        features_shape=1,
        seq_type='RNN',
        seq_units=2,
        dense_units=2,
        activation='relu',
        dropout=0.0,
        optimizer=custom_opt,
        learning_rate=0.01,
        loss='mse',
        metrics=['mae']
    )
    assert isinstance(model.optimizer, tf.keras.optimizers.SGD)


def test_get_best_model(monkeypatch):
    # Mock mlflow and pandas DataFrame
    class DummyRow:
        run_id = 'abc123'

    class DummySeries:
        @property
        def iloc(self):
            class ILoc:
                def __getitem__(self, idx):
                    return DummyRow()
            return ILoc()

    class DummySorted:
        def __getitem__(self, key):
            # key is 'run_id'
            return DummySeries()

    class DummyDF:
        def sort_values(self, metric, ascending):
            return DummySorted()

    class DummyMLflowTF:
        @staticmethod
        def load_model(path):
            return "dummy_model"

    def dummy_search_runs(experiment_id):
        return DummyDF()
    monkeypatch.setattr("ml_utils.mlflow.search_runs", dummy_search_runs)
    monkeypatch.setattr("ml_utils.mlflow.tensorflow.load_model", DummyMLflowTF.load_model)
    result = get_best_model("exp_id")
    assert result == "dummy_model"
    monkeypatch.setattr("ml_utils.mlflow.search_runs", dummy_search_runs)
    monkeypatch.setattr("ml_utils.mlflow.tensorflow.load_model", DummyMLflowTF.load_model)
    result = get_best_model("exp_id")
    assert result == "dummy_model"


def test_walk_forward_splits_no_leakage_and_expanding():
    splits = list(walk_forward_splits(100, n_splits=4, gap=7, min_train_size=50))
    assert len(splits) == 4
    for train_idx, val_idx in splits:
        assert train_idx[0] == 0
        assert val_idx.min() >= train_idx.max() + 7 + 1  # purge gap respected
    # expanding window: each fold trains on more data
    train_sizes = [len(t) for t, _ in splits]
    assert train_sizes == sorted(train_sizes) and len(set(train_sizes)) == 4


def test_walk_forward_splits_covers_tail():
    splits = list(walk_forward_splits(100, n_splits=2, val_size=10, gap=0, min_train_size=50))
    assert splits[-1][1][-1] == 69  # 50 train + 10 val (fold 1) + 10 val (fold 2)


def test_walk_forward_splits_invalid():
    with pytest.raises(ValueError):
        list(walk_forward_splits(50, n_splits=0))
    with pytest.raises(ValueError):
        list(walk_forward_splits(50, n_splits=3, val_size=20, min_train_size=40))


def test_scale_train_val_fits_on_train_only():
    rng = np.random.default_rng(0)
    X_seq_train = rng.normal(500, 50, size=(20, 7, 1))
    X_feat_train = rng.normal(0, 1, size=(20, 3))
    y_train = rng.normal(500, 50, size=20)
    X_seq_val = rng.normal(5000, 50, size=(5, 7, 1))  # very different distribution
    X_feat_val = rng.normal(100, 1, size=(5, 3))
    y_val = rng.normal(5000, 50, size=5)

    out = scale_train_val(X_seq_train, X_feat_train, y_train,
                          X_seq_val, X_feat_val, y_val)

    # Scaler statistics must come from training data only (leakage check).
    assert out["y_scaler"].center_[0] == pytest.approx(np.median(y_train))
    assert out["x_scaler"].center_[0] == pytest.approx(np.median(X_feat_train[:, 0]))

    # Shapes preserved; val values are transformed with train statistics.
    assert out["X_seq_train"].shape == (20, 7, 1)
    assert out["X_seq_val"].shape == (5, 7, 1)
    assert out["X_features_val"].shape == (5, 3)
    assert out["y_train"].shape == (20,)
    assert out["y_val"].shape == (5,)

    # Inverse transform round-trips.
    recovered = out["y_scaler"].inverse_transform(out["y_val"].reshape(-1, 1)).flatten()
    assert np.allclose(recovered, y_val)


def test_scale_train_val_train_only():
    out = scale_train_val(
        np.arange(14.0).reshape(2, 7, 1),
        np.ones((2, 3)),
        np.array([100.0, 200.0]),
    )
    assert "X_seq_val" not in out
    assert "y_scaler" in out and "x_scaler" in out


def test_log_regression_metrics(monkeypatch):
    logged = {}

    def fake_log_metrics(d):
        logged.update(d)

    monkeypatch.setattr("mlflow.log_metrics", fake_log_metrics)
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    metrics = log_regression_metrics(y_true, y_pred, prefix="holdout")
    assert set(logged) == {f"holdout_{k}" for k in metrics}
    assert logged["holdout_mae"] == pytest.approx(metrics["mae"])
