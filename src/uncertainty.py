"""
Uncertainty and human-in-the-loop review alerting layer for industrial fault detection.

Methodology:
- In high-stakes industrial predictive maintenance, an automated binary model decision
  is often insufficient if the model's confidence is borderline or degraded.
- This module inspects prediction confidence (e.g. max class probability or margin from
  decision threshold) and flags uncertain predictions for human expert review.

Important Limitation on Probabilities:
- Outputs from Platt scaling (`SVC(probability=True)`) or sigmoid calibrations are
  heuristically mapped via 5-fold cross-validation. They are NOT guaranteed to be
  strictly calibrated true Bayesian posterior probabilities, particularly when input
  data experiences distribution shift or extreme class imbalance.
- High confidence does NOT guarantee ground-truth correctness, nor does low confidence
  necessarily mean the equipment is broken. Confidence is solely a heuristic measure of
  model decisiveness.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class AlertItem:
    """Individual sample prediction with uncertainty alert status."""

    index: int
    prediction: int
    confidence: float
    flag_human_review: bool
    reason: str
    probabilities: Optional[List[float]] = None


@dataclass
class UncertaintyAlertReport:
    """Batch evaluation of predictions for uncertainty-based human review."""

    total_samples: int
    flagged_samples_count: int
    flagged_fraction: float
    confidence_threshold: float
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    items: List[AlertItem] = field(default_factory=list)
    summary_text: str = ""


class UncertaintyAlertLayer:
    """
    Uncertainty-aware alerting layer.

    Parameters
    ----------
    confidence_threshold : float, default=0.70
        Confidence threshold below which a prediction is flagged for human review.
        For binary classification, confidence ranges from 0.5 (maximum uncertainty)
        to 1.0 (complete certainty).
    """

    def __init__(self, confidence_threshold: float = 0.70):
        if not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in the range [0.5, 1.0], got {confidence_threshold}"
            )
        self.confidence_threshold = confidence_threshold

    def evaluate_predictions(
        self,
        predictions: Union[np.ndarray, List[int], pd.Series],
        probabilities: Optional[Union[np.ndarray, List[List[float]]]] = None,
        confidence_scores: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> UncertaintyAlertReport:
        """
        Evaluate predictions and flag low-confidence decisions for human review.

        Either `probabilities` (e.g. from `predict_proba`, shape (N, 2)) OR
        `confidence_scores` (shape (N,)) must be provided.

        Parameters
        ----------
        predictions : array-like of shape (N,)
            Predicted class labels (0 or 1).
        probabilities : array-like of shape (N, n_classes), optional
            Predicted class probabilities from classifier.
        confidence_scores : array-like of shape (N,), optional
            Pre-computed confidence scores in [0.0, 1.0].

        Returns
        -------
        report : UncertaintyAlertReport
            Structured report detailing flagged samples and alert statistics.
        """
        preds = np.asarray(predictions)
        n = len(preds)
        if n == 0:
            raise ValueError("Predictions array cannot be empty.")

        # Determine confidence per sample
        conf_arr: np.ndarray = np.zeros(n, dtype=float)
        probs_list: Optional[List[Optional[List[float]]]] = None

        if probabilities is not None:
            probs = np.asarray(probabilities, dtype=float)
            if probs.shape[0] != n:
                raise ValueError(
                    f"Probabilities shape {probs.shape} does not match predictions count {n}."
                )

            probs_list = []
            for i in range(n):
                row = probs[i]
                if np.isnan(row).any() or np.isinf(row).any():
                    conf_arr[i] = 0.0
                    probs_list.append(None)
                else:
                    # Check if normalized
                    s = np.sum(row)
                    if s <= 0:
                        conf_arr[i] = 0.0
                        probs_list.append(None)
                    else:
                        norm_row = row / s
                        conf_arr[i] = float(np.max(norm_row))
                        probs_list.append(norm_row.tolist())

        elif confidence_scores is not None:
            raw_conf = np.asarray(confidence_scores, dtype=float)
            if len(raw_conf) != n:
                raise ValueError(
                    f"Confidence scores count {len(raw_conf)} does not match predictions count {n}."
                )

            probs_list = [None] * n
            for i in range(n):
                c = raw_conf[i]
                if np.isnan(c) or np.isinf(c) or c < 0.0:
                    conf_arr[i] = 0.0
                else:
                    conf_arr[i] = min(float(c), 1.0)
        else:
            raise ValueError("Either 'probabilities' or 'confidence_scores' must be provided.")

        items: List[AlertItem] = []
        flagged_count = 0

        for idx in range(n):
            pred = int(preds[idx])
            conf = float(conf_arr[idx])
            sample_probs = probs_list[idx] if probs_list else None

            if np.isnan(conf) or conf == 0.0:
                flag = True
                reason = "Invalid/missing confidence score or malformed probabilities"
            elif conf < self.confidence_threshold:
                flag = True
                reason = (
                    f"Low prediction confidence ({conf:.1%} < "
                    f"threshold {self.confidence_threshold:.1%})"
                )
            else:
                flag = False
                reason = f"High confidence ({conf:.1%} >= {self.confidence_threshold:.1%})"

            if flag:
                flagged_count += 1

            items.append(
                AlertItem(
                    index=idx,
                    prediction=pred,
                    confidence=conf,
                    flag_human_review=flag,
                    reason=reason,
                    probabilities=sample_probs,
                )
            )

        flagged_fraction = flagged_count / n if n > 0 else 0.0
        mean_conf = float(np.mean(conf_arr)) if n > 0 else 0.0
        min_conf = float(np.min(conf_arr)) if n > 0 else 0.0
        max_conf = float(np.max(conf_arr)) if n > 0 else 0.0

        summary = (
            f"UNCERTAINTY ALERT SUMMARY: {flagged_count}/{n} samples ({flagged_fraction:.1%}) "
            f"flagged for human review (confidence < {self.confidence_threshold:.1%}). "
            f"Mean confidence: {mean_conf:.1%}, Range: [{min_conf:.1%}, {max_conf:.1%}]."
        )

        return UncertaintyAlertReport(
            total_samples=n,
            flagged_samples_count=flagged_count,
            flagged_fraction=flagged_fraction,
            confidence_threshold=self.confidence_threshold,
            mean_confidence=mean_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            items=items,
            summary_text=summary,
        )
