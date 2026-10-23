"""
Unit and integration tests for Phase 2, Phase 3, Phase 4, and Phase 5:
- Data loader & leakage prevention
- Classical baseline pipeline
- Quantum feature map & kernel construction
- Quantum preprocessor dimensions & scaling
- Quantum classifier fit/predict shapes
- Controlled stratified sampling reproducibility
- Statistical sensor drift detector (KS-test) & synthetic drift injection
- Uncertainty-aware human review alerting layer
- Dashboard helper functions (single sensor evaluation & pipeline caching)
"""

import unittest
import numpy as np
import pandas as pd

from src.classical_model import build_preprocessor, create_classical_pipeline
from src.dashboard import evaluate_single_sensor_reading
from src.data_loader import (
    FEATURE_NAMES,
    LEAKAGE_TARGETS,
    TARGET_NAME,
    get_features_and_target,
)
from src.drift_detector import (
    CONTINUOUS_SENSOR_FEATURES,
    SensorDriftDetector,
    inject_synthetic_drift,
)
from src.quantum_model import (
    QuantumKernelClassifier,
    build_quantum_preprocessor,
    create_quantum_feature_map,
    get_stratified_quantum_subset,
)
from src.uncertainty import UncertaintyAlertLayer


class TestFaultDetectionProject(unittest.TestCase):
    """Test suite covering classical baseline, quantum models, drift detection, uncertainty, and dashboard."""

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

    # ----------------- Phase 4: Drift Detection Tests -----------------

    def test_drift_detector_no_drift(self):
        """Test that two batches from identical distributions do not trigger drift."""
        X, _ = get_features_and_target(self.synthetic_df)
        ref = X.iloc[:50]
        test = X.iloc[50:]

        detector = SensorDriftDetector(alpha=0.01)
        detector.fit_reference(ref)
        report = detector.detect_drift(test)

        self.assertFalse(report.drift_detected)
        self.assertEqual(len(report.drifted_features), 0)

    def test_drift_detector_with_synthetic_drift(self):
        """Test that injected synthetic shift is detected with statistical significance."""
        X, _ = get_features_and_target(self.synthetic_df)
        ref = X.iloc[:50]
        test = X.iloc[50:].copy()

        drifted_test = inject_synthetic_drift(test, drift_magnitudes={"Air temperature": 5.0})

        detector = SensorDriftDetector(alpha=0.05)
        detector.fit_reference(ref)
        report = detector.detect_drift(drifted_test)

        self.assertTrue(report.drift_detected)
        self.assertIn("Air temperature", report.drifted_features)
        self.assertLess(report.feature_results["Air temperature"].p_value, 0.05)

    def test_drift_detector_edge_cases(self):
        """Ensure small batches, missing columns, and constant features are handled gracefully."""
        ref = pd.DataFrame({"Air temperature": [300.0] * 20, "Torque": np.linspace(30, 50, 20)})
        detector = SensorDriftDetector(min_samples=10)
        detector.fit_reference(ref, features=["Air temperature", "Torque"])

        small_test = pd.DataFrame({"Air temperature": [300.0] * 5, "Torque": [40.0] * 5})
        report = detector.detect_drift(small_test)
        self.assertIn("Air temperature", report.feature_results)
        self.assertIsNotNone(report.feature_results["Air temperature"].warning)

    # ----------------- Phase 4: Uncertainty Alert Tests -----------------

    def test_uncertainty_alert_thresholds(self):
        """Verify high and low confidence predictions are flagged appropriately."""
        alert_layer = UncertaintyAlertLayer(confidence_threshold=0.80)
        preds = [0, 1, 0, 1]
        probs = [
            [0.95, 0.05],
            [0.25, 0.75],
            [0.55, 0.45],
            [0.10, 0.90],
        ]

        report = alert_layer.evaluate_predictions(preds, probabilities=probs)

        self.assertEqual(report.total_samples, 4)
        self.assertEqual(report.flagged_samples_count, 2)
        self.assertFalse(report.items[0].flag_human_review)
        self.assertTrue(report.items[1].flag_human_review)
        self.assertTrue(report.items[2].flag_human_review)
        self.assertFalse(report.items[3].flag_human_review)

    def test_uncertainty_invalid_and_boundary_inputs(self):
        """Verify handling of invalid probabilities and threshold boundary cases."""
        alert_layer = UncertaintyAlertLayer(confidence_threshold=0.70)
        preds = [0, 1, 0]
        probs = [
            [np.nan, 0.5],
            [0.30, 0.70],
            [0.69, 0.31],
        ]

        report = alert_layer.evaluate_predictions(preds, probabilities=probs)

        self.assertTrue(report.items[0].flag_human_review)
        self.assertIn("Invalid", report.items[0].reason)
        self.assertFalse(report.items[1].flag_human_review)
        self.assertTrue(report.items[2].flag_human_review)

    # ----------------- Phase 5: Dashboard Helper Tests -----------------

    def test_dashboard_evaluate_single_sensor_reading(self):
        """Test the dashboard single-reading evaluation helper function."""
        X, y = get_features_and_target(self.synthetic_df)
        pipeline = create_classical_pipeline(svm_params={"random_state": 42})
        pipeline.fit(X, y)

        sample_reading = {
            "Type": "L",
            "Air temperature": 300.0,
            "Process temperature": 310.0,
            "Rotational speed": 1500,
            "Torque": 40.0,
            "Tool wear": 100,
        }

        res = evaluate_single_sensor_reading(pipeline, sample_reading, confidence_threshold=0.70)

        self.assertIn(res["prediction"], [0, 1])
        self.assertIn("prediction_label", res)
        self.assertIn("probabilities", res)
        self.assertGreaterEqual(res["confidence"], 0.5)
        self.assertIsInstance(res["flag_human_review"], bool)


if __name__ == "__main__":
    unittest.main()
