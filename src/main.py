"""
Main entry point for running the Phase 2 Classical Baseline.

Executes:
1. Loading the UCI AI4I 2020 Predictive Maintenance Dataset.
2. Stratified train/test splitting to maintain class proportions.
3. Training the classical SVM pipeline with balanced class weights.
4. Evaluating predictions on the held-out test set (Accuracy, Precision, Recall, F1, Confusion Matrix).
"""

from typing import Any, Dict
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.classical_model import create_classical_pipeline
from src.data_loader import load_ai4i_dataset


def run_baseline(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate the classical SVM baseline model.

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction of samples allocated to test set.
    random_state : int, default=42
        Seed for reproducible splitting and model training.

    Returns
    -------
    results : dict
        Dictionary containing trained pipeline, test predictions, and evaluation metrics.
    """
    print("=" * 70)
    print("Phase 2: Classical Baseline (AI4I 2020 Predictive Maintenance)")
    print("=" * 70)

    print("\n[1/4] Loading dataset...")
    X, y = load_ai4i_dataset()
    print(f"Loaded {len(X)} samples with {X.shape[1]} input features.")
    print(f"Target distribution:\n{y.value_counts(normalize=True).rename({0: 'Normal (0)', 1: 'Failure (1)'}).map('{:.2%}'.format)}")

    print(f"\n[2/4] Splitting into train/test (test_size={test_size}, stratify=y, seed={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples (Failures: {y_train.sum()})")
    print(f"Testing set:  {X_test.shape[0]} samples (Failures: {y_test.sum()})")

    print("\n[3/4] Building and training Classical SVM Pipeline (RBF kernel, balanced class weights)...")
    pipeline = create_classical_pipeline(svm_params={"random_state": random_state})
    pipeline.fit(X_train, y_train)

    print("\n[4/4] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, digits=4)
    report_text = classification_report(
        y_test, y_pred, target_names=["Normal (0)", "Failure (1)"], digits=4
    )

    print("\n" + "-" * 70)
    print("EVALUATION REPORT:")
    print("-" * 70)
    print(report_text)
    print("Confusion Matrix:")
    print(f"                Predicted Normal  Predicted Failure")
    print(f"Actual Normal        {cm[0, 0]:<16}  {cm[0, 1]:<16}")
    print(f"Actual Failure       {cm[1, 0]:<16}  {cm[1, 1]:<16}")
    print("-" * 70)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


if __name__ == "__main__":
    run_baseline()
