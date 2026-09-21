"""Regression metrics and interval evaluation for donation forecasts.

Shared by notebooks, MLflow logging, and (later) the API so that every
reported number is computed the same way.
"""

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_INTERVAL_QUANTILES = (0.025, 0.10, 0.90, 0.975)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in percent; zero targets are excluded to avoid division blow-ups."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """Share of points where the predicted direction of change matches the actual one.

    Direction is measured against the previous actual value (y_prev).
    Ties (no change) count as correct only if both sides are ties.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    y_prev = np.asarray(y_prev, dtype=float).flatten()
    if not (len(y_true) == len(y_pred) == len(y_prev)):
        raise ValueError("y_true, y_pred and y_prev must have equal length.")
    if len(y_true) == 0:
        return float("nan")
    actual_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return float(np.mean(actual_dir == pred_dir))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Standard metric bundle: MAE, RMSE, MAPE (%), R², optional direction accuracy."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }
    if y_prev is not None:
        metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred, y_prev)
    return metrics


def residual_quantiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: Sequence[float] = DEFAULT_INTERVAL_QUANTILES,
) -> Dict[float, float]:
    """Quantiles of the residuals (y_true - y_pred) for interval calibration."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    if len(y_true) == 0:
        raise ValueError("Cannot compute residual quantiles on empty arrays.")
    residuals = y_true - y_pred
    return {float(q): float(np.quantile(residuals, q)) for q in quantiles}


def prediction_intervals(
    y_pred: np.ndarray,
    quantile_residuals: Dict[float, float],
    clip_non_negative: bool = True,
) -> Dict[str, np.ndarray]:
    """Build prediction intervals from residual quantiles.

    Expects the default four quantiles (2.5%, 10%, 90%, 97.5%) and returns
    80% and 95% interval bounds.
    """
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    lower_95 = y_pred + quantile_residuals[0.025]
    lower_80 = y_pred + quantile_residuals[0.10]
    upper_80 = y_pred + quantile_residuals[0.90]
    upper_95 = y_pred + quantile_residuals[0.975]
    if clip_non_negative:
        lower_95 = np.maximum(lower_95, 0.0)
        lower_80 = np.maximum(lower_80, 0.0)
    return {
        "lower_95": lower_95,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "upper_95": upper_95,
    }


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical share of observations inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    lower = np.asarray(lower, dtype=float).flatten()
    upper = np.asarray(upper, dtype=float).flatten()
    if not (len(y_true) == len(lower) == len(upper)):
        raise ValueError("y_true, lower and upper must have equal length.")
    if np.any(lower > upper):
        raise ValueError("Interval lower bound exceeds upper bound.")
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
