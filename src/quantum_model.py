"""
Quantum-Kernel Support Vector Machine module for industrial fault detection.

This module provides:
1. Dimensionality reduction & quantum scaling pipeline:
   - `ColumnTransformer` with `OneHotEncoder` and `StandardScaler` (fitted strictly on training data)
   - `PCA` dimensionality reduction to compress sensor features down to `n_qubits`
   - `MinMaxScaler(feature_range=(0, pi))` to map values appropriately into quantum rotation angles
2. Quantum feature map and kernel estimation:
   - `zz_feature_map` (or `ZZFeatureMap`) with configurable repetitions and entanglement
   - `FidelityQuantumKernel` computed using Qiskit statevector simulation
3. `QuantumKernelClassifier`:
   - A scikit-learn-compatible wrapper wrapping `SVC(kernel='precomputed', class_weight='balanced')`
   - Evaluates train/test gram matrices via the quantum simulator without data leakage
4. Reproducible stratified subset sampling for tractable quantum simulation experiments.

Hardware / Simulation Details:
- Number of Qubits: Configurable (default = 4 qubits)
- Feature Map: ZZ Feature Map (`reps=1`, `entanglement='linear'`)
- Circuit Depth: ~11 gates for 4 qubits (reps=1)
- Quantum Simulator: Qiskit `StatevectorSampler` / statevector simulation fidelity
- Complexity: O(N^2) pairwise quantum circuit evaluations
"""

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

import qiskit.circuit.library as qcl
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.classical_model import build_preprocessor


def create_quantum_feature_map(
    n_qubits: int = 4,
    reps: int = 1,
    entanglement: str = "linear",
):
    """
    Construct a ZZFeatureMap circuit for quantum encoding.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits (matches feature dimension).
    reps : int, default=1
        Number of feature map repetitions.
    entanglement : str, default='linear'
        Entanglement strategy ('linear', 'full', 'circular').

    Returns
    -------
    circuit : QuantumCircuit
    """
    if hasattr(qcl, "zz_feature_map"):
        return qcl.zz_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    return qcl.ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement=entanglement,
    )


def build_quantum_preprocessor(
    n_components: int = 4,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a feature scaling and PCA compression pipeline for quantum circuits.

    Pipeline stages:
    1. Base ColumnTransformer (OneHotEncoder for 'Type', StandardScaler for numeric sensors)
    2. PCA(n_components) to project into target qubit dimension
    3. MinMaxScaler(feature_range=(0, pi)) to scale inputs into valid rotation angles

    Parameters
    ----------
    n_components : int, default=4
        Number of PCA principal components (target qubits).
    random_state : int, default=42
        Random state for PCA reproducibility.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    base_preprocessor = build_preprocessor()
    preprocessor = Pipeline(
        steps=[
            ("encoder_and_scaler", base_preprocessor),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            ("quantum_angle_scaler", MinMaxScaler(feature_range=(0.0, np.pi))),
        ]
    )
    return preprocessor


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Quantum-Kernel Support Vector Classifier.

    Evaluates the Gram matrix using Qiskit's `FidelityQuantumKernel` and
    fits an underlying `SVC(kernel='precomputed')`.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
        C: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

        self.feature_map_ = None
        self.quantum_kernel_ = None
        self.svc_ = None
        self.X_train_ = None
        self.train_time_ = 0.0

    def _init_kernel(self):
        self.feature_map_ = create_quantum_feature_map(
            n_qubits=self.n_qubits,
            reps=self.reps,
            entanglement=self.entanglement,
        )
        self.quantum_kernel_ = FidelityQuantumKernel(feature_map=self.feature_map_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelClassifier":
        """
        Compute the training Gram matrix using the quantum simulator and fit SVC.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_qubits)
            Preprocessed training features scaled to [0, pi].
        y : np.ndarray of shape (n_samples,)
            Target labels.
        """
        self._init_kernel()
        X_arr = np.asarray(X)
        self.X_train_ = X_arr

        t0 = time.time()
        K_train = self.quantum_kernel_.evaluate(x_vec=X_arr)

        self.svc_ = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.svc_.fit(K_train, y)
        self.train_time_ = time.time() - t0
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate test-vs-train Gram matrix and predict labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_qubits)
            Preprocessed test features.
        """
        if self.svc_ is None or self.X_train_ is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        X_arr = np.asarray(X)
        K_test = self.quantum_kernel_.evaluate(x_vec=X_arr, y_vec=self.X_train_)
        return self.svc_.predict(K_test)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate decision function for test samples."""
        if self.svc_ is None or self.X_train_ is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        X_arr = np.asarray(X)
        K_test = self.quantum_kernel_.evaluate(x_vec=X_arr, y_vec=self.X_train_)
        return self.svc_.decision_function(K_test)


def get_stratified_quantum_subset(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int = 100,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract a reproducible, stratified subset of the dataset for quantum experiments.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full target vector.
    n_samples : int, default=100
        Number of samples to draw.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_sub : pd.DataFrame
    y_sub : pd.Series
    """
    if n_samples >= len(X):
        return X.copy(), y.copy()

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    subset_idx, _ = next(sss.split(X, y))
    return X.iloc[subset_idx].copy(), y.iloc[subset_idx].copy()
