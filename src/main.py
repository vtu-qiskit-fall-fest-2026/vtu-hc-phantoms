"""
Main entry point for running Phase 2 and Phase 3:
- Full classical baseline on the entire 10,000-sample dataset
- Fair, controlled benchmark of Classical SVM vs Quantum-Kernel SVM on a stratified subset
"""

import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.classical_model import create_classical_pipeline
from src.data_loader import load_ai4i_dataset
from src.quantum_model import (
    QuantumKernelClassifier,
    build_quantum_preprocessor,
    get_stratified_quantum_subset,
)


def run_baseline(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate the full classical SVM baseline model on all 10,000 samples.
    """
    print("=" * 75)
    print("Phase 2: Full Classical Baseline (AI4I 2020 Predictive Maintenance)")
    print("=" * 75)

    print("\n[1/4] Loading dataset...")
    X, y = load_ai4i_dataset()
    print(f"Loaded {len(X)} samples with {X.shape[1]} input features.")
    print(
        f"Target distribution:\n"
        f"{y.value_counts(normalize=True).rename({0: 'Normal (0)', 1: 'Failure (1)'}).map('{:.2%}'.format)}"
    )

    print(f"\n[2/4] Splitting into train/test (test_size={test_size}, stratify=y, seed={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples (Failures: {y_train.sum()})")
    print(f"Testing set:  {X_test.shape[0]} samples (Failures: {y_test.sum()})")

    print("\n[3/4] Building and training Classical SVM Pipeline (RBF kernel, balanced class weights)...")
    pipeline = create_classical_pipeline(svm_params={"random_state": random_state})
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"\n[4/4] Evaluating on test set (training took {train_time:.2f}s)...")
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_dict = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, digits=4)
    report_text = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["Normal (0)", "Failure (1)"], digits=4
    )

    print("\n" + "-" * 75)
    print("FULL CLASSICAL BASELINE EVALUATION REPORT:")
    print("-" * 75)
    print(report_text)
    print("Confusion Matrix:")
    print("                Predicted Normal  Predicted Failure")
    print(f"Actual Normal        {cm[0, 0]:<16}  {cm[0, 1]:<16}")
    print(f"Actual Failure       {cm[1, 0]:<16}  {cm[1, 1]:<16}")
    print("-" * 75)

    return {
        "pipeline": pipeline,
        "train_time": train_time,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


def run_quantum_comparison(
    subset_size: int = 80,
    test_size: float = 0.25,
    n_qubits: int = 4,
    reps: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run a controlled head-to-head comparison between Classical SVM (RBF) and
    Quantum-Kernel SVM (ZZFeatureMap) on the exact same stratified subset.

    Parameters
    ----------
    subset_size : int, default=80
        Total samples extracted for the quantum simulation experiment.
    test_size : float, default=0.25
        Fraction allocated to testing (60 train / 20 test).
    n_qubits : int, default=4
        Number of qubits / PCA dimensions for quantum circuit encoding.
    reps : int, default=1
        Feature map circuit repetitions.
    random_state : int, default=42
        Reproducibility seed.
    """
    print("\n" + "=" * 75)
    print("Phase 3: Classical vs. Quantum-Kernel SVM Comparison (Stratified Subset)")
    print("=" * 75)
    print(
        f"Configuration: subset_size={subset_size}, test_size={test_size}, "
        f"n_qubits={n_qubits}, reps={reps}, seed={random_state}"
    )

    X_full, y_full = load_ai4i_dataset()
    X_sub, y_sub = get_stratified_quantum_subset(
        X_full, y_full, n_samples=subset_size, random_state=random_state
    )

    print(
        f"\nExtracted stratified subset: {len(X_sub)} samples "
        f"(Failures: {y_sub.sum()} / {len(y_sub)} = {y_sub.mean():.1%})"
    )

    # Split train/test using identical indices for both models
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=test_size, random_state=random_state, stratify=y_sub
    )
    print(f"Train samples: {len(X_train)} (Failures: {y_train.sum()})")
    print(f"Test samples:  {len(X_test)}  (Failures: {y_test.sum()})")

    # 1. Preprocessing for Quantum (fitted strictly on X_train to prevent leakage)
    q_preprocessor = build_quantum_preprocessor(n_components=n_qubits, random_state=random_state)
    q_preprocessor.fit(X_train)
    X_train_q = q_preprocessor.transform(X_train)
    X_test_q = q_preprocessor.transform(X_test)

    # 2. Train Classical SVM on the subset with matching preprocessed features
    print("\n--- Model A: Classical SVM (RBF kernel, balanced weights on subset) ---")
    classical_svc = SVC(kernel="rbf", class_weight="balanced", random_state=random_state)
    t0_c = time.time()
    classical_svc.fit(X_train_q, y_train)
    c_train_time = time.time() - t0_c
    y_pred_c = classical_svc.predict(X_test_q)

    cm_c = confusion_matrix(y_test, y_pred_c, labels=[0, 1])
    rep_c = classification_report(
        y_test, y_pred_c, labels=[0, 1], output_dict=True, zero_division=0
    )
    rep_text_c = classification_report(
        y_test,
        y_pred_c,
        labels=[0, 1],
        target_names=["Normal (0)", "Failure (1)"],
        digits=4,
        zero_division=0,
    )
    print(f"Classical Training time: {c_train_time:.3f}s")
    print(rep_text_c)
    print("Confusion Matrix:")
    print(f"Actual Normal:  [TN={cm_c[0, 0]}, FP={cm_c[0, 1]}]")
    print(f"Actual Failure: [FN={cm_c[1, 0]}, TP={cm_c[1, 1]}]")

    # 3. Train Quantum-Kernel SVM on the subset
    print(f"\n--- Model B: Quantum-Kernel SVM ({n_qubits} qubits, reps={reps}, ZZFeatureMap) ---")
    print(f"Evaluating {len(X_train_q)}x{len(X_train_q)} quantum kernel Gram matrix via Statevector simulation...")
    q_clf = QuantumKernelClassifier(
        n_qubits=n_qubits,
        reps=reps,
        entanglement="linear",
        class_weight="balanced",
        random_state=random_state,
    )
    t0_q = time.time()
    q_clf.fit(X_train_q, y_train.values)
    q_train_time = time.time() - t0_q

    print(f"Evaluating {len(X_test_q)}x{len(X_train_q)} test kernel Gram matrix...")
    t0_eval = time.time()
    y_pred_q = q_clf.predict(X_test_q)
    q_eval_time = time.time() - t0_eval

    cm_q = confusion_matrix(y_test, y_pred_q, labels=[0, 1])
    rep_q = classification_report(
        y_test, y_pred_q, labels=[0, 1], output_dict=True, zero_division=0
    )
    rep_text_q = classification_report(
        y_test,
        y_pred_q,
        labels=[0, 1],
        target_names=["Normal (0)", "Failure (1)"],
        digits=4,
        zero_division=0,
    )
    print(f"Quantum Kernel Train time: {q_train_time:.3f}s | Test eval time: {q_eval_time:.3f}s")
    print(rep_text_q)
    print("Confusion Matrix:")
    print(f"Actual Normal:  [TN={cm_q[0, 0]}, FP={cm_q[0, 1]}]")
    print(f"Actual Failure: [FN={cm_q[1, 0]}, TP={cm_q[1, 1]}]")

    # 4. Comparative Summary
    print("\n" + "=" * 75)
    print("CONTROLLED EXPERIMENT SUMMARY (Classical RBF vs. Quantum Kernel)")
    print("=" * 75)
    print(f"{'Metric':<25} {'Classical SVM (Subset)':<25} {'Quantum-Kernel SVM':<25}")
    print("-" * 75)
    print(f"{'Accuracy':<25} {rep_c['accuracy']:<25.4f} {rep_q['accuracy']:<25.4f}")
    print(f"{'Failure Precision':<25} {rep_c.get('1', {}).get('precision', 0):<25.4f} {rep_q.get('1', {}).get('precision', 0):<25.4f}")
    print(f"{'Failure Recall':<25} {rep_c.get('1', {}).get('recall', 0):<25.4f} {rep_q.get('1', {}).get('recall', 0):<25.4f}")
    print(f"{'Failure F1-Score':<25} {rep_c.get('1', {}).get('f1-score', 0):<25.4f} {rep_q.get('1', {}).get('f1-score', 0):<25.4f}")
    print(f"{'Macro Avg F1':<25} {rep_c['macro avg']['f1-score']:<25.4f} {rep_q['macro avg']['f1-score']:<25.4f}")
    print(f"{'Training / Kernel Time':<25} {f'{c_train_time:.3f}s':<25} {f'{q_train_time:.3f}s':<25}")
    print("-" * 75)

    return {
        "classical": {
            "train_time": c_train_time,
            "confusion_matrix": cm_c,
            "report": rep_c,
        },
        "quantum": {
            "train_time": q_train_time,
            "eval_time": q_eval_time,
            "confusion_matrix": cm_q,
            "report": rep_q,
        },
    }


if __name__ == "__main__":
    # 1. Full Classical Baseline on 10,000 samples
    run_baseline()
    # 2. Head-to-Head Controlled Comparison on 80 samples
    run_quantum_comparison(subset_size=80, test_size=0.25, n_qubits=4, reps=1)
