import numpy as np
import pytest

from src.donations import evaluation as ev


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([100.0, 200.0, 300.0, 400.0])
        m = ev.compute_regression_metrics(y, y.copy())
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["mape"] == 0.0
        assert m["r2"] == 1.0

    def test_known_values(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        m = ev.compute_regression_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(15.0)
        assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
        assert m["mape"] == pytest.approx((0.10 + 0.10) / 2 * 100)

    def test_mape_excludes_zero_targets(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([50.0, 110.0])
        m = ev.compute_regression_metrics(y_true, y_pred)
        assert m["mape"] == pytest.approx(10.0)

    def test_directional_accuracy(self):
        y_prev = np.array([100.0, 100.0, 100.0])
        y_true = np.array([110.0, 90.0, 100.0])
        y_pred = np.array([105.0, 95.0, 100.0])
        m = ev.compute_regression_metrics(y_true, y_pred, y_prev=y_prev)
        assert m["directional_accuracy"] == pytest.approx(1.0)

        y_pred_wrong = np.array([95.0, 105.0, 105.0])
        m = ev.compute_regression_metrics(y_true, y_pred_wrong, y_prev=y_prev)
        assert m["directional_accuracy"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ev.compute_regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ev.compute_regression_metrics(np.array([]), np.array([]))


class TestIntervals:
    def test_residual_quantiles(self):
        y_true = np.arange(1.0, 101.0)
        y_pred = np.zeros(100)
        q = ev.residual_quantiles(y_true, y_pred, quantiles=(0.025, 0.975))
        assert q[0.025] == pytest.approx(np.quantile(y_true, 0.025))
        assert q[0.975] == pytest.approx(np.quantile(y_true, 0.975))

    def test_prediction_intervals_order_and_clip(self):
        y_pred = np.array([10.0, 100.0])
        q = {0.025: -50.0, 0.10: -20.0, 0.90: 20.0, 0.975: 50.0}
        iv = ev.prediction_intervals(y_pred, q)
        # First point clipped to zero (donations cannot be negative).
        assert iv["lower_95"][0] == 0.0
        assert iv["lower_80"][0] == 0.0
        assert iv["upper_95"][0] == 60.0
        for lo_95, lo_80, up_80, up_95 in zip(
            iv["lower_95"], iv["lower_80"], iv["upper_80"], iv["upper_95"]
        ):
            assert lo_95 <= lo_80 <= up_80 <= up_95

    def test_interval_coverage(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.0, 2.0, 0.0, 5.0])
        upper = np.array([2.0, 3.0, 4.0, 6.0])
        # Points 1, 2, 3 inside; point 4 (4.0 < lower 5.0) outside.
        assert ev.interval_coverage(y_true, lower, upper) == pytest.approx(0.75)

    def test_interval_coverage_invalid_bounds(self):
        with pytest.raises(ValueError, match="lower bound"):
            ev.interval_coverage(np.array([1.0]), np.array([2.0]), np.array([1.0]))
