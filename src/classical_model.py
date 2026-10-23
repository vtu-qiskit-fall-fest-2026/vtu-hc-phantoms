"""
Classical baseline model module for industrial fault detection.

Builds a scikit-learn Pipeline with:
1. ColumnTransformer:
   - One-hot encoding for categorical product variants ('Type')
   - Robust/Standard scaling for continuous sensor features
2. Support Vector Classifier (SVC) with an RBF kernel and balanced class weighting.

Class Imbalance Strategy:
In industrial predictive maintenance datasets such as AI4I 2020, failure instances
represent only ~3.4% of all observations. Unweighted models tend to default to the
majority class (normal), missing critical machine breakdowns.
To address this without synthetically interpolating noisy sensor coordinates,
we configure `class_weight='balanced'` in the SVM. This automatically adjusts
weights inversely proportional to class frequencies:
    w_j = n_samples / (n_classes * n_samples_j)
penalizing false negatives significantly more than false positives.
"""

from typing import Any, Dict, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def build_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct a ColumnTransformer preprocessor for sensor features.

    Parameters
    ----------
    categorical_cols : list of str, optional
        Categorical feature names (defaults to ['Type']).
    numeric_cols : list of str, optional
        Numerical sensor feature names.

    Returns
    -------
    preprocessor : ColumnTransformer
    """
    if categorical_cols is None:
        categorical_cols = ["Type"]

    if numeric_cols is None:
        numeric_cols = [
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
        ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            ),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def create_classical_pipeline(
    svm_params: Optional[Dict[str, Any]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    Create an end-to-end scikit-learn Pipeline with preprocessing and an SVM classifier.

    Parameters
    ----------
    svm_params : dict, optional
        Keyword arguments for scikit-learn's SVC. Defaults to:
        - kernel: 'rbf'
        - class_weight: 'balanced'
        - probability: True (for uncertainty estimation/probability outputs)
        - random_state: 42
    categorical_cols : list of str, optional
        List of categorical column names.
    numeric_cols : list of str, optional
        List of numerical column names.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    preprocessor = build_preprocessor(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    default_svm_params: Dict[str, Any] = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    }
    if svm_params is not None:
        default_svm_params.update(svm_params)

    classifier = SVC(**default_svm_params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline
