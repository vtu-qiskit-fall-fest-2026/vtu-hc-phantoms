"""
Unit and integration tests for Phase 2: Data Loader and Classical Baseline.
Compatible with standard library unittest and pytest.
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


class TestPhase2ClassicalBaseline(unittest.TestCase):
    """Test suite for data loading, leakage prevention, and classical SVM modeling."""

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

        # Synthetic target with failures
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

    def test_leakage_prevention(self):
        """Ensure target and failure mode indicator columns are strictly stripped from X."""
        X, y = get_features_and_target(self.synthetic_df)

        self.assertNotIn(TARGET_NAME, X.columns)
        for leakage_col in LEAKAGE_TARGETS:
            self.assertNotIn(leakage_col, X.columns)

        # Verify input feature columns match expected list
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

        # 2 dummy columns for Type ('L', 'M', 'H' -> 2 after drop='first') + 5 numeric features = 7
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


if __name__ == "__main__":
    unittest.main()
