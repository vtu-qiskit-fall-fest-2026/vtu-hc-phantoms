"""
Data loader module for the UCI AI4I 2020 Predictive Maintenance Dataset.

Dataset attributes:
- Input Features:
    - Type: Product quality variant ('L', 'M', 'H')
    - Air temperature [K]
    - Process temperature [K]
    - Rotational speed [rpm]
    - Torque [Nm]
    - Tool wear [min]
- Target:
    - Machine failure: binary indicator (0 = normal, 1 = failure)

Note on Leakage Prevention:
Failure mode indicators ('TWF', 'HDF', 'PWF', 'OSF', 'RNF') are component breakdown
causes that directly reveal the occurrence of a machine failure. These are strictly
excluded from input features to prevent target leakage.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_AI4I_2020_ID = 601
FEATURE_NAMES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
TARGET_NAME = "Machine failure"
LEAKAGE_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_ai4i_dataset(
    cache_path: Optional[str | Path] = None,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the AI4I 2020 Predictive Maintenance Dataset.

    Attempts to load from a local cached CSV if available, or fetches from the
    UCI Machine Learning Repository via ucimlrepo.

    Parameters
    ----------
    cache_path : str or Path, optional
        File path to save/load the cached CSV. Defaults to 'data/ai4i2020.csv'.
    force_download : bool, default=False
        If True, re-fetches from UCI repo even if cached file exists.

    Returns
    -------
    X : pd.DataFrame
        Clean feature dataframe containing sensor measurements and product type.
    y : pd.Series
        Binary target variable indicating 'Machine failure' (0 or 1).
    """
    if cache_path is None:
        cache_path = Path("data") / "ai4i2020.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path)
    else:
        dataset = fetch_ucirepo(id=UCI_AI4I_2020_ID)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()

        # Combine into one full dataframe for caching
        df = pd.concat([features, targets], axis=1)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return get_features_and_target(df)


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target y from the raw dataset, ensuring
    leakage prevention.

    Parameters
    ----------
    df : pd.DataFrame
        Full AI4I 2020 dataset dataframe.

    Returns
    -------
    X : pd.DataFrame
        Features without target or leakage columns.
    y : pd.Series
        Machine failure target labels.
    """
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_NAME}' not found in DataFrame.")

    y = df[TARGET_NAME].astype(int)

    # Validate feature presence
    available_features = [col for col in FEATURE_NAMES if col in df.columns]
    if not available_features:
        raise ValueError("None of the expected feature columns were found.")

    X = df[available_features].copy()

    # Double check that no target or leakage column is included in X
    for col in [TARGET_NAME] + LEAKAGE_TARGETS:
        if col in X.columns:
            X = X.drop(columns=[col])

    return X, y
