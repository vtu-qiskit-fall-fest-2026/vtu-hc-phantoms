"""
Unit and integration tests for Phase 2 & Phase 3:
- Data loader & leakage prevention
- Classical baseline pipeline
- Quantum feature map & kernel construction
- Quantum preprocessor dimensions & scaling
- Quantum classifier fit/predict shapes
- Controlled stratified sampling reproducibility
"""

import unittest
import numpy as np
import pandas as pd

from src.classical_model import build_preprocessor, create_classical_pipeline
from src.data_loader import (
    FEATURE_NAMES,
    LEAKAGE_TARGETS,
    TARGET_NAME,
    get_features_and_target,
)
from src.quantum_model import (
    QuantumKernelClassifier,
    build_quantum_preprocessor,
    create_quantum_feature_map,
    get_stratified_quantum_subset,
)


class TestFaultDetectionProject(unittest.TestCase):
    """Test suite covering classical baseline and quantum-kernel model components."""

    def setUp(self):
        """Generate a deterministic synthetic dataset matching the AI4I schema."""
        rng = np.random.RandomState(42)
        n = 100
        types = rng.choice(["L", "M", "H"], size=n)
        air_temp = rng.normal(300.0, 2.0, size=n)
        process_temp = air_temp + rng.normal(10.0, 1.0, size=n)
        rot_speed = rng.randint(1200, 2800, size=n)
        torque = rng.normal(40.0, 10.0, size=n)
        tool_wear = rng.randint(0, 250, size=n)

        # Synthetic target with ~10% failures
        failure = (tool_wear > 200).astype(int)

        self.synthetic_df = pd.DataFrame(
            {
                "Type": types,
                "Air temperature": air_temp,
                "Process temperature": process_temp,
                "Rotational speed": rot_speed,
                "Torque": torque,
                "Tool wear": tool_wear,
                "Machine failure": failure,
                "TWF": (tool_wear > 220).astype(int),
                "HDF": rng.binomial(1, 0.02, size=n),
                "PWF": rng.binomial(1, 0.02, size=n),
                "OSF": rng.binomial(1, 0.02, size=n),
                "RNF": rng.binomial(1, 0.01, size=n),
            }
        )

    # ----------------- Classical & Data Loader Tests -----------------

    def test_leakage_prevention(self):
        """Ensure target and failure mode indicator columns are strictly stripped from X."""
        X, y = get_features_and_target(self.synthetic_df)

        self.assertNotIn(TARGET_NAME, X.columns)
        for leakage_col in LEAKAGE_TARGETS:
            self.assertNotIn(leakage_col, X.columns)

        self.assertEqual(list(X.columns), FEATURE_NAMES)
        self.assertEqual(len(X), len(self.synthetic_df))
        self.assertEqual(len(y), len(self.synthetic_df))
        self.assertTrue(np.issubdtype(y.dtype, np.integer))

    def test_missing_target_raises_error(self):
        """Ensure get_features_and_target raises ValueError if target is missing."""
        df_no_target = pd.DataFrame({"Type": ["L"], "Air temperature": [300.0]})
        with self.assertRaises(ValueError):
            get_features_and_target(df_no_target)

    def test_classical_preprocessor(self):
        """Test that preprocessor correctly transforms categorical and numeric columns."""
        X, _ = get_features_and_target(self.synthetic_df)
        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X)

        self.assertEqual(X_trans.shape, (len(self.synthetic_df), 7))
        self.assertFalse(np.isnan(X_trans).any())

    def test_classical_pipeline_fit_predict(self):
        """Test that end-to-end classical SVM pipeline fits and predicts with probabilities."""
        X, y = get_features_and_target(self.synthetic_df)
        pipeline = create_classical_pipeline(svm_params={"random_state": 42})

        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        probs = pipeline.predict_proba(X)

        self.assertEqual(preds.shape, (len(X),))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
        self.assertEqual(probs.shape, (len(X), 2))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))

    def test_pipeline_class_weight_balanced(self):
        """Verify default SVM is configured with balanced class weights."""
        pipeline = create_classical_pipeline()
        classifier = pipeline.named_steps["classifier"]
        self.assertEqual(classifier.class_weight, "balanced")
        self.assertTrue(classifier.probability)

    # ----------------- Quantum Model Tests -----------------

    def test_quantum_feature_map_construction(self):
        """Test ZZFeatureMap creation across qubit dimensions and reps."""
        fm_4q = create_quantum_feature_map(n_qubits=4, reps=1, entanglement="linear")
        self.assertEqual(fm_4q.num_qubits, 4)

        fm_2q = create_quantum_feature_map(n_qubits=2, reps=2, entanglement="full")
        self.assertEqual(fm_2q.num_qubits, 2)

    def test_quantum_preprocessor_shape_and_range(self):
        """Verify quantum preprocessor compresses to n_components and bounds within [0, pi]."""
        X, _ = get_features_and_target(self.synthetic_df)
        q_preproc = build_quantum_preprocessor(n_components=4, random_state=42)
        X_q = q_preproc.fit_transform(X)

        self.assertEqual(X_q.shape, (len(X), 4))
        self.assertGreaterEqual(float(X_q.min()), 0.0)
        self.assertLessEqual(float(X_q.max()), np.pi + 1e-6)

    def test_stratified_quantum_subset(self):
        """Verify stratified sampling produces reproducible and exact subset sizes."""
        X, y = get_features_and_target(self.synthetic_df)
        X_sub1, y_sub1 = get_stratified_quantum_subset(X, y, n_samples=30, random_state=42)
        X_sub2, y_sub2 = get_stratified_quantum_subset(X, y, n_samples=30, random_state=42)

        self.assertEqual(len(X_sub1), 30)
        self.assertEqual(len(y_sub1), 30)
        pd.testing.assert_frame_equal(X_sub1, X_sub2)
        pd.testing.assert_series_equal(y_sub1, y_sub2)

    def test_quantum_classifier_fit_predict(self):
        """Test that QuantumKernelClassifier fits on small synthetic data and predicts."""
        rng = np.random.RandomState(42)
        X_train = rng.uniform(0.0, np.pi, size=(8, 2))
        y_train = np.array([0, 1, 0, 0, 1, 0, 0, 1])
        X_test = rng.uniform(0.0, np.pi, size=(4, 2))

        q_clf = QuantumKernelClassifier(
            n_qubits=2, reps=1, class_weight="balanced", random_state=42
        )
        q_clf.fit(X_train, y_train)

        preds = q_clf.predict(X_test)
        dec = q_clf.decision_function(X_test)

        self.assertEqual(preds.shape, (4,))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
        self.assertEqual(dec.shape, (4,))


if __name__ == "__main__":
    unittest.main()
