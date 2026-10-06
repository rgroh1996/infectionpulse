import numpy as np
import pandas as pd
import pytest

from infectionpulse.respiratory_model import (
    FullPosterior,
    RespiratoryModel,
    quantile_grid_probability,
)


def posterior(rows=1):
    return FullPosterior(
        [-2.0, -1.0, 1.0, 2.0], np.tile(np.log([0.2, 0.5, 0.3]), (rows, 1))
    )


def test_tail_probability_is_not_finite_bar_clamping():
    p = posterior(5)
    np.testing.assert_allclose(
        p.survival([-2, -1, 0, 1, 2]), [0.9, 0.8, 0.55, 0.3, 0.15]
    )
    assert posterior().survival([4])[0] > 0
    assert posterior().survival([-4])[0] < 1


def test_full_support_quantile_inverse_in_both_tails():
    q = [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9, 0.99, 0.999]
    values = posterior().quantiles(q)[0]
    assert np.all(np.diff(values) > 0)
    assert values[0] < -2 and values[-1] > 2
    np.testing.assert_allclose(1 - posterior(len(q)).survival(values), q, atol=1e-12)


def test_posterior_unit_and_shape_contract():
    p = posterior(3)
    response = {
        "borders": p.borders,
        "logits": np.tile(np.log([0.2, 0.5, 0.3]), (3, 1)),
        "mean": p.mean(),
        "median": p.quantiles([0.5])[:, 0],
    }
    FullPosterior.from_response(response, 3)
    response["mean"] = response["mean"] * 10 + 1
    with pytest.raises(ValueError, match="unit contract"):
        FullPosterior.from_response(response, 3)
    with pytest.raises(ValueError):
        FullPosterior([0, 1, 2], [[-np.inf, -np.inf]])


def test_constant_target_and_serialization_without_hosted_calls(tmp_path):
    frame = pd.DataFrame(
        {
            "incidence": [10.0] * 50,
            "inc_lag_1": [9.0] * 50,
            "target_incidence": [12.0] * 50,
        }
    )
    model = RespiratoryModel("TabPFN-3.5", ["incidence", "inc_lag_1"]).fit(frame)
    raw = model.raw(frame, np.full(50, 11), tmp_path)
    np.testing.assert_allclose(raw.q50, 12)
    assert raw.probability_raw.eq(1).all()
    model.calibrate(frame, raw, np.full(50, 11), "sigmoid")
    assert model.calibration_status == "insufficient_events"
    model.save(tmp_path / "model.joblib")
    restored = RespiratoryModel.load(tmp_path / "model.joblib")
    pd.testing.assert_frame_equal(raw, restored.raw(frame, np.full(50, 11), tmp_path))


def test_calibration_needs_twenty_events_each_class():
    frame = pd.DataFrame({"target_incidence": [1.0] * 40 + [20.0] * 10})
    raw = pd.DataFrame(
        {
            "q10": [0.0] * 50,
            "q50": [5.0] * 50,
            "q90": [25.0] * 50,
            "probability_raw": [0.2] * 50,
        }
    )
    model = RespiratoryModel("LightGBM", ["incidence"])
    model.calibrate(frame, raw, np.full(50, 10), "sigmoid")
    assert model.calibrator is None
    assert model.calibration_status == "insufficient_events"


def test_quantile_response_cache_changes_threshold_without_rebilling(tmp_path):
    class FakeHosted:
        model_id_ = "unit-test-server-model"
        model_path = "v3.5_default"
        _last_meta = {
            "tabpfn_config": {"model_path": "tabpfn-v3.5-20260909.safetensors"},
            "billing_model_version": "v3.5",
        }
        calls = 0

        def predict(self, x, **kwargs):
            assert kwargs["output_type"] == "quantiles"
            self.calls += 1
            p = posterior(len(x))
            return p.quantiles(kwargs["quantiles"]).T

    frame = pd.DataFrame({"incidence": [10.0, 15.0], "inc_lag_1": [9.0, 12.0]})
    model = RespiratoryModel("TabPFN-3.5", ["incidence", "inc_lag_1"])
    model.medians = pd.Series({"incidence": 0.0, "inc_lag_1": 0.0})
    model.constant_target = None
    model.estimators = [FakeHosted()]
    low = model.raw(frame, [10.0, 15.0], tmp_path)
    high = model.raw(frame, [100.0, 100.0], tmp_path)
    assert model.estimators[0].calls == 1
    assert (low.probability_raw > high.probability_raw).all()
    np.testing.assert_allclose(low.q50, high.q50)


def test_dense_grid_inversion_reports_uncertain_tails():
    grid = np.arange(1, 100) / 100
    probability, low, high, inside = quantile_grid_probability(
        np.tile(np.arange(1, 100), (3, 1)), grid, [0.0, 50.0, 100.0]
    )
    np.testing.assert_allclose(probability, [0.99, 0.5, 0.01])
    np.testing.assert_allclose([low[0], high[0]], [0.99, 1.0])
    np.testing.assert_allclose([low[2], high[2]], [0.0, 0.01])
    assert inside.tolist() == [False, True, False]
