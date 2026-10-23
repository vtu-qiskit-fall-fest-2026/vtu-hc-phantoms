"""
Sensor drift detection module for industrial fault-detection telemetry.

Methodology:
- Applies the non-parametric two-sample Kolmogorov-Smirnov (KS) test to evaluate whether
  a batch of incoming sensor readings follows the same continuous distribution as a
  reference baseline (e.g., training data or baseline healthy operational window).
- Statistical drift detection monitors *distribution shift* in the input feature space.
- IMPORTANT: Evidence of sensor drift indicates covariate shift (e.g. sensor calibration
  degradation, ambient seasonal shifts), NOT definitive equipment failure. Drift detection
  is unsupervised and never accesses ground-truth target labels.

Features Monitored:
Continuous industrial sensor channels (Air temperature, Process temperature,
Rotational speed, Torque, Tool wear).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

CONTINUOUS_SENSOR_FEATURES: List[str] = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


@dataclass
class FeatureDriftResult:
    """Individual feature drift evaluation results."""

    feature_name: str
    statistic: float
    p_value: float
    drift_detected: bool
    reference_mean: float
    test_mean: float
    reference_std: float
    test_std: float
    sample_size_ref: int
    sample_size_test: int
    warning: Optional[str] = None


@dataclass
class DriftReport:
    """Summary of multi-feature sensor drift detection."""

    p_value_threshold: float
    features_evaluated: List[str]
    drifted_features: List[str]
    drift_detected: bool
    feature_results: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    summary_text: str = ""


class SensorDriftDetector:
    """
    Two-sample Kolmogorov-Smirnov sensor drift detector.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for the hypothesis test. A p-value < alpha flags drift.
    min_samples : int, default=10
        Minimum number of samples required in the test batch to perform reliable testing.
    """

    def __init__(self, alpha: float = 0.05, min_samples: int = 10):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Significance level alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha
        self.min_samples = min_samples
        self.reference_data: Optional[pd.DataFrame] = None

    def fit_reference(self, reference_df: pd.DataFrame, features: Optional[List[str]] = None) -> "SensorDriftDetector":
        """
        Store reference sensor observations (e.g. from the training partition).

        Parameters
        ----------
        reference_df : pd.DataFrame
            DataFrame containing baseline sensor readings.
        features : list of str, optional
            Columns to monitor. Defaults to continuous sensor features present in reference_df.
        """
        if reference_df is None or len(reference_df) < self.min_samples:
            raise ValueError(
                f"Reference dataset must contain at least {self.min_samples} rows, "
                f"got {len(reference_df) if reference_df is not None else 0}"
            )

        if features is None:
            features = [c for c in CONTINUOUS_SENSOR_FEATURES if c in reference_df.columns]
        else:
            features = [c for c in features if c in reference_df.columns]

        if not features:
            raise ValueError("No valid sensor features found to monitor in reference_df.")

        self.features_ = features
        self.reference_data = reference_df[features].copy()
        return self

    def detect_drift(self, test_df: pd.DataFrame) -> DriftReport:
        """
        Test a new batch of sensor readings against the fitted reference distribution.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame containing incoming sensor readings.

        Returns
        -------
        report : DriftReport
            Structured report with per-feature test statistics, p-values, and overall drift flags.
        """
        if self.reference_data is None:
            raise RuntimeError("Reference data has not been fitted. Call fit_reference() first.")

        if test_df is None or len(test_df) == 0:
            raise ValueError("Test batch cannot be empty or None.")

        feature_results: Dict[str, FeatureDriftResult] = {}
        drifted_features: List[str] = []

        is_small_batch = len(test_df) < self.min_samples

        for feature in self.features_:
            if feature not in test_df.columns:
                res = FeatureDriftResult(
                    feature_name=feature,
                    statistic=np.nan,
                    p_value=np.nan,
                    drift_detected=False,
                    reference_mean=float(self.reference_data[feature].mean()),
                    test_mean=np.nan,
                    reference_std=float(self.reference_data[feature].std()),
                    test_std=np.nan,
                    sample_size_ref=len(self.reference_data),
                    sample_size_test=len(test_df),
                    warning=f"Feature '{feature}' missing from test batch",
                )
                feature_results[feature] = res
                continue

            ref_series = self.reference_data[feature].dropna().to_numpy(dtype=float)
            test_series = test_df[feature].dropna().to_numpy(dtype=float)

            if len(test_series) == 0:
                res = FeatureDriftResult(
                    feature_name=feature,
                    statistic=np.nan,
                    p_value=np.nan,
                    drift_detected=False,
                    reference_mean=float(np.mean(ref_series)),
                    test_mean=np.nan,
                    reference_std=float(np.std(ref_series)),
                    test_std=np.nan,
                    sample_size_ref=len(ref_series),
                    sample_size_test=0,
                    warning="Test feature values are all NaN",
                )
                feature_results[feature] = res
                continue

            # Check for constant/zero variance
            ref_var = float(np.var(ref_series))
            test_var = float(np.var(test_series))
            warning_msg = None
            if is_small_batch:
                warning_msg = f"Batch size ({len(test_series)}) < min_samples ({self.min_samples})"
            elif ref_var == 0.0 and test_var == 0.0:
                warning_msg = "Both reference and test features are constant"

            # Execute two-sample Kolmogorov-Smirnov test
            try:
                ks_res = stats.ks_2samp(ref_series, test_series)
                stat = float(ks_res.statistic)
                p_val = float(ks_res.pvalue)
                drift_flag = bool(p_val < self.alpha)
            except Exception as e:
                stat = np.nan
                p_val = np.nan
                drift_flag = False
                warning_msg = f"KS test error: {e}"

            if drift_flag:
                drifted_features.append(feature)

            feature_results[feature] = FeatureDriftResult(
                feature_name=feature,
                statistic=stat,
                p_value=p_val,
                drift_detected=drift_flag,
                reference_mean=float(np.mean(ref_series)),
                test_mean=float(np.mean(test_series)),
                reference_std=float(np.std(ref_series)),
                test_std=float(np.std(test_series)),
                sample_size_ref=len(ref_series),
                sample_size_test=len(test_series),
                warning=warning_msg,
            )

        overall_drift = len(drifted_features) > 0

        if overall_drift:
            summary = (
                f"DRIFT WARNING: Statistically significant distribution shift detected in "
                f"{len(drifted_features)}/{len(self.features_)} monitored features "
                f"(p < {self.alpha}): {', '.join(drifted_features)}."
            )
        else:
            summary = (
                f"NO DRIFT DETECTED: All {len(self.features_)} monitored features are consistent "
                f"with reference distributions at alpha={self.alpha}."
            )

        return DriftReport(
            p_value_threshold=self.alpha,
            features_evaluated=self.features_,
            drifted_features=drifted_features,
            drift_detected=overall_drift,
            feature_results=feature_results,
            summary_text=summary,
        )


def inject_synthetic_drift(
    df: pd.DataFrame,
    drift_magnitudes: Optional[Dict[str, float]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Inject a reproducible synthetic sensor shift into numerical features.

    Note: The original dataset does NOT naturally have known ground-truth drift.
    This helper applies controlled additive/multiplicative perturbations specifically
    for benchmarking and verifying detector sensitivity.

    Parameters
    ----------
    df : pd.DataFrame
        Clean input DataFrame.
    drift_magnitudes : dict of str -> float, optional
        Standard deviation multiplier per feature.
        e.g., {'Air temperature': 3.0} shifts Air temperature by +3.0 * reference std.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    drifted_df : pd.DataFrame
        Copy of DataFrame with synthetic drift injected into specified columns.
    """
    drifted = df.copy()
    if drift_magnitudes is None:
        # Default scenario: temperature increase (+2.5 sigma) and tool wear acceleration (+2.0 sigma)
        drift_magnitudes = {
            "Air temperature": 2.5,
            "Process temperature": 2.5,
            "Tool wear": 2.0,
        }

    for col, scale in drift_magnitudes.items():
        if col in drifted.columns and np.issubdtype(drifted[col].dtype, np.number):
            col_std = float(df[col].std())
            if col_std == 0.0 or np.isnan(col_std):
                col_std = 1.0
            shift = scale * col_std
            drifted[col] = drifted[col] + shift

    return drifted
