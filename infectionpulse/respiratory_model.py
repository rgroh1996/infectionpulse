"""Small-data ARE models and an explicit full-support posterior contract.

No SDK or tree library is imported at module import time. Run hosted and tree
jobs in separate spawned processes on macOS (their OpenMP runtimes conflict).
"""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import erfc, ndtri, softmax
from sklearn.linear_model import LogisticRegression

VERSION = 1
BATCH_ROWS = 256
QUANTILES = np.array([0.1, 0.5, 0.9])
HOSTED_QUANTILES = np.arange(1, 100) / 100
TREE_QUANTILES = np.array(
    [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)
DEFAULT_CONFIG = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2}
SEARCH_CONFIGS = [
    DEFAULT_CONFIG,
    {"n_estimators": 240, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 240, "learning_rate": 0.03, "max_depth": 3},
]


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


class FullPosterior:
    """Uniform interior bars and half-normal outer bars, in fitted-y units.

    The installed SDK's FullSupportBarDistribution inherits finite-support
    cdf/icdf methods. These formulas instead match its overridden pi/mean/pdf.
    """

    def __init__(self, borders, logits):
        self.borders, logits = array(borders), array(logits)
        if (
            self.borders.ndim != 1
            or len(self.borders) < 3
            or not np.isfinite(self.borders).all()
            or np.any(np.diff(self.borders) <= 0)
        ):
            raise ValueError("Posterior requires finite strictly increasing borders")
        if logits.ndim != 2 or logits.shape[1] != len(self.borders) - 1:
            raise ValueError("Posterior row/bucket shape mismatch")
        if (
            np.isnan(logits).any()
            or np.isposinf(logits).any()
            or not np.isfinite(logits).any(axis=1).all()
        ):
            raise ValueError("Invalid posterior logits")
        self.probabilities = softmax(logits, axis=1)
        self.left_scale = (self.borders[1] - self.borders[0]) / ndtri(0.75)
        self.right_scale = (self.borders[-1] - self.borders[-2]) / ndtri(0.75)

    def survival(self, threshold):
        y = np.broadcast_to(array(threshold), (len(self.probabilities),))
        b, p = self.borders, self.probabilities
        bucket = np.clip(np.searchsorted(b, y, side="right") - 1, 0, len(b) - 2)
        rows = np.arange(len(y))
        c = np.cumsum(p, axis=1) - p
        fraction = np.clip((y - b[bucket]) / np.diff(b)[bucket], 0, 1)
        result = 1 - c[rows, bucket] - fraction * p[rows, bucket]
        left, right = y <= b[1], y >= b[-2]
        result[left] = 1 - p[left, 0] * erfc(
            (b[1] - y[left]) / (self.left_scale * np.sqrt(2))
        )
        result[right] = p[right, -1] * erfc(
            (y[right] - b[-2]) / (self.right_scale * np.sqrt(2))
        )
        return np.clip(result, 0, 1)

    def quantiles(self, quantiles=QUANTILES):
        qs = array(quantiles)
        if qs.ndim != 1 or not ((qs > 0) & (qs < 1)).all():
            raise ValueError("Quantiles must lie strictly between zero and one")
        p, b = self.probabilities, self.borders
        cumulative = np.cumsum(p, axis=1)
        output = np.empty((len(p), len(qs)))
        for j, q in enumerate(qs):
            for i in range(len(p)):
                if q < p[i, 0]:
                    output[i, j] = b[1] - self.left_scale * ndtri(1 - q / (2 * p[i, 0]))
                elif q > 1 - p[i, -1]:
                    output[i, j] = b[-2] + self.right_scale * ndtri(
                        0.5 + (q - (1 - p[i, -1])) / (2 * p[i, -1])
                    )
                else:
                    k = min(np.searchsorted(cumulative[i], q, side="right"), len(b) - 2)
                    before = cumulative[i, k] - p[i, k]
                    output[i, j] = b[k] + (b[k + 1] - b[k]) * (q - before) / p[i, k]
        return output

    def mean(self):
        means = (self.borders[:-1] + self.borders[1:]) / 2
        means[0] = self.borders[1] - self.left_scale * np.sqrt(2 / np.pi)
        means[-1] = self.borders[-2] + self.right_scale * np.sqrt(2 / np.pi)
        return self.probabilities @ means

    @classmethod
    def from_response(cls, response, n_rows):
        missing = {"borders", "logits", "mean", "median"} - response.keys()
        if missing:
            raise ValueError(f"Incomplete hosted posterior: {sorted(missing)}")
        posterior = cls(response["borders"], response["logits"])
        if len(posterior.probabilities) != n_rows:
            raise ValueError("Hosted posterior has wrong number of rows")
        # A normalized-border / raw-target mismatch fails here rather than
        # generating plausible-looking event probabilities in the wrong units.
        if not np.allclose(
            posterior.mean(), array(response["mean"]).reshape(-1), rtol=2e-3, atol=2e-4
        ):
            raise ValueError("Hosted posterior mean/unit contract failed")
        med = array(response["median"]).reshape(-1)
        interior = (posterior.probabilities[:, 0] < 0.5) & (
            posterior.probabilities[:, -1] < 0.5
        )
        if not np.allclose(
            posterior.quantiles([0.5])[interior, 0], med[interior], rtol=2e-3, atol=2e-4
        ):
            raise ValueError("Hosted posterior interior-quantile contract failed")
        return posterior


def quantile_grid_probability(values, levels, thresholds):
    """Invert an explicit quantile grid, retaining interval/tail limitations.

    This is a quantile-grid approximation, never a reconstructed full CDF.
    Outside the supplied grid, probabilities are conservatively capped at
    the outer grid levels and the service withholds a displayed percentage.
    """
    values, levels, thresholds = array(values), array(levels), array(thresholds)
    if values.ndim != 2 or values.shape != (len(thresholds), len(levels)):
        raise ValueError("Quantile-grid shape mismatch")
    if not np.isfinite(values).all() or not np.isfinite(thresholds).all():
        raise ValueError("Nonfinite quantile grid")
    if np.any(np.diff(values, axis=1) < -1e-6) or np.any(np.diff(levels) <= 0):
        raise ValueError("Unordered quantile grid")
    estimates, lows, highs, inside = [], [], [], []
    for row, threshold in zip(values, thresholds):
        lower = max(0, int(np.searchsorted(row, threshold, side="left")) - 1)
        upper = min(len(levels) - 1, int(np.searchsorted(row, threshold, side="right")))
        in_grid = bool(row[0] <= threshold <= row[-1])
        if threshold < row[0]:
            estimate, low, high = 1 - levels[0], 1 - levels[0], 1.0
        elif threshold > row[-1]:
            estimate, low, high = 1 - levels[-1], 0.0, 1 - levels[-1]
        else:
            estimate = 1 - float(np.interp(threshold, row, levels))
            low, high = 1 - levels[upper], 1 - levels[lower]
        estimates.append(estimate)
        lows.append(low)
        highs.append(high)
        inside.append(in_grid)
    return (
        np.asarray(estimates),
        np.asarray(lows),
        np.asarray(highs),
        np.asarray(inside),
    )


def feature_values(frame, columns):
    x = frame[list(columns)].astype(float).copy()
    for column in [
        "incidence",
        "inc_lag_1",
        "inc_lag_2",
        "inc_lag_3",
        "inc_lag_4",
        "inc_lag_52",
        "rolling_mean3",
        "seasonal_naive_incidence",
    ]:
        if column in x:
            if (x[column].dropna() < 0).any():
                raise ValueError("Negative ARE incidence")
            x[column] = np.log1p(x[column])
    if "latest_change" in x:
        x["latest_change"] = x.incidence - x.inc_lag_1
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Infinite respiratory feature")
    return x


def make_tree(name, quantile, config=DEFAULT_CONFIG):
    options = dict(**config, random_state=42, n_jobs=1)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="quantile",
            alpha=float(quantile),
            verbosity=-1,
            num_leaves=2 ** options["max_depth"],
            min_child_samples=5,
            reg_lambda=1,
            **options,
        )
    if name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=float(quantile),
            tree_method="hist",
            reg_lambda=1,
            **options,
        )
    raise ValueError(name)


class RespiratoryModel:
    def __init__(self, name, columns, config=None, transform="log_change"):
        if name not in {"TabPFN-3.5", "LightGBM", "XGBoost"}:
            raise ValueError(name)
        if transform not in {"log_change", "log_level"}:
            raise ValueError(transform)
        self.name, self.columns, self.transform = name, list(columns), transform
        self.config, self.version = config or DEFAULT_CONFIG.copy(), VERSION
        self.api_version, self.calibration_status = {}, "uncalibrated"
        self.calibrator, self.interval_adjustment = None, 0.0

    def offset(self, frame):
        return (
            np.log1p(frame.incidence.to_numpy())
            if self.transform == "log_change"
            else np.zeros(len(frame))
        )

    def x(self, frame):
        values = feature_values(frame, self.columns)
        return np.column_stack(
            [values.fillna(self.medians), values.isna().astype(float)]
        )

    def fit(self, context):
        self.medians = feature_values(context, self.columns).median().fillna(0)
        x = self.x(context)
        y = np.log1p(context.target_incidence.to_numpy()) - self.offset(context)
        if not np.isfinite(y).all() or len(context) < 5:
            raise ValueError("Need at least five finite labeled context rows")
        self.constant_target = float(y[0]) if np.ptp(y) < 1e-12 else None
        if self.constant_target is not None:
            self.estimators = []
        elif self.name == "TabPFN-3.5":
            from infectionpulse.tabpfn_api import require_api_token

            require_api_token()
            from tabpfn_client import TabPFNRegressor

            self.estimators = [
                TabPFNRegressor.create_default_for_version(
                    "v3.5", n_estimators=8, random_state=42, fit_mode="fit_with_cache"
                ).fit(x, y)
            ]
        else:
            self.estimators = [
                make_tree(self.name, q, self.config).fit(x, y) for q in TREE_QUANTILES
            ]
        return self

    def raw(self, frame, threshold, cache_dir):
        x, offset = self.x(frame), self.offset(frame)
        z_threshold = np.log1p(np.asarray(threshold)) - offset
        if self.constant_target is not None:
            zq = np.full((len(frame), 3), self.constant_target)
            probability = (self.constant_target > z_threshold).astype(float)
            low, high, inside = (
                probability,
                probability,
                np.ones(len(frame), dtype=bool),
            )
        elif self.name != "TabPFN-3.5":
            z = np.sort(
                np.column_stack([m.predict(x) for m in self.estimators]), axis=1
            )
            zq = z[:, [int(np.flatnonzero(TREE_QUANTILES == q)[0]) for q in QUANTILES]]
            probability, low, high, inside = quantile_grid_probability(
                z, TREE_QUANTILES, z_threshold
            )
        else:
            from infectionpulse.tabpfn_api import verify_api_version

            estimator = self.estimators[0]
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            quantile_parts = []
            for start in range(0, len(x), BATCH_ROWS):
                batch = x[start : start + BATCH_ROWS]
                key = hashlib.sha256(
                    batch.tobytes()
                    + HOSTED_QUANTILES.tobytes()
                    + str(estimator.model_id_).encode()
                ).hexdigest()
                path = cache_dir / f"{key}_quantiles99.joblib"
                if path.exists():
                    saved = joblib.load(path)
                    response, self.api_version = saved["response"], saved["version"]
                else:
                    response = estimator.predict(
                        batch,
                        output_type="quantiles",
                        quantiles=HOSTED_QUANTILES.tolist(),
                    )
                    self.api_version = {
                        **verify_api_version(estimator),
                        "prediction_interface": "explicit_99_quantiles",
                        "full_posterior_available": False,
                    }
                    response = array(response)
                    if (
                        response.shape != (len(HOSTED_QUANTILES), len(batch))
                        or not np.isfinite(response).all()
                    ):
                        raise ValueError(
                            "Hosted explicit 99-quantile shape contract failed"
                        )
                    if np.any(np.diff(response, axis=0) < -1e-6):
                        raise ValueError("Hosted quantiles are unordered")
                    atomic_dump(
                        {"response": response, "version": self.api_version}, path
                    )
                if array(response).shape != (len(HOSTED_QUANTILES), len(batch)):
                    raise ValueError("Cached dense quantile contract failed")
                quantile_parts.append(array(response).T)
            z = np.concatenate(quantile_parts)
            zq = z[:, [9, 49, 89]]
            probability, low, high, inside = quantile_grid_probability(
                z, HOSTED_QUANTILES, z_threshold
            )
        with np.errstate(over="raise", invalid="raise"):
            q = np.maximum(0, np.expm1(zq + offset[:, None]))
        if not np.isfinite(q).all() or not np.isfinite(probability).all():
            raise ValueError("Nonfinite posterior prediction")
        return pd.DataFrame(
            {
                "q10": q[:, 0],
                "q50": q[:, 1],
                "q90": q[:, 2],
                "probability_raw": np.clip(probability, 0, 1),
                "probability_lower_bound": low,
                "probability_upper_bound": high,
                "probability_threshold_in_grid": inside,
                "probability_method": "quantile_grid_approximation",
            },
            index=frame.index,
        )

    def calibrate(self, truth, raw, thresholds, method="raw"):
        labels = (truth.target_incidence.to_numpy() > np.asarray(thresholds)).astype(
            int
        )
        self.calibration_status = "uncalibrated_raw"
        if min(int(labels.sum()), int((1 - labels).sum())) < 20:
            self.calibration_status = "insufficient_events"
        elif method == "sigmoid":
            p = np.clip(raw.probability_raw.to_numpy(), 1e-6, 1 - 1e-6)
            self.calibrator = LogisticRegression(C=1).fit(
                np.log(p / (1 - p)).reshape(-1, 1), labels
            )
            self.calibration_status = "sigmoid"
        y = truth.target_incidence.to_numpy()
        error = np.maximum(raw.q10.to_numpy() - y, y - raw.q90.to_numpy())
        rank = min(len(error), int(np.ceil((len(error) + 1) * 0.8)))
        self.interval_adjustment = max(0.0, float(np.sort(error)[rank - 1]))
        return self

    def adjusted(self, raw):
        result = raw.copy()
        probability = raw.probability_raw.to_numpy()
        if self.calibrator is not None:
            p = np.clip(probability, 1e-6, 1 - 1e-6)
            probability = self.calibrator.predict_proba(
                np.log(p / (1 - p)).reshape(-1, 1)
            )[:, 1]
        result["probability"] = probability
        result["interval_low"] = np.maximum(0, raw.q10 - self.interval_adjustment)
        result["interval_high"] = raw.q90 + self.interval_adjustment
        result["calibration_status"] = self.calibration_status
        return result

    def save(self, path):
        state = self.__dict__.copy()
        if self.name == "TabPFN-3.5":
            state["estimators"] = [m.save_model() for m in self.estimators]
        atomic_dump(state, path)

    @classmethod
    def load(cls, path):
        state = joblib.load(path)
        if state["version"] != VERSION:
            raise ValueError("Incompatible respiratory model")
        if state["name"] == "TabPFN-3.5" and state["estimators"]:
            from tabpfn_client import TabPFNRegressor

            state["estimators"] = [
                TabPFNRegressor.load_model(m) for m in state["estimators"]
            ]
        model = cls(state["name"], state["columns"])
        model.__dict__.update(state)
        return model


def atomic_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)
