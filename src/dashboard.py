"""
Interactive Streamlit Dashboard for Drift-Aware Industrial Fault Detection
Using Hybrid Quantum–Classical Machine Learning.

Modules (in presentation order):
1. Overview & System Telemetry
2. Live Fault Prediction & Uncertainty-Aware Review
3. Sensor Drift Monitoring (Two-Sample KS-Test)
4. Quantum vs. Classical Benchmark (Fair, Controlled Comparison)
5. Unified Alert Management Center (Failure Alerts, Covariate Shift Warnings, Review Flags)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from src.classical_model import create_classical_pipeline
from src.data_loader import (
    FEATURE_NAMES,
    load_ai4i_dataset,
)
from src.drift_detector import (
    CONTINUOUS_SENSOR_FEATURES,
    DriftReport,
    SensorDriftDetector,
    inject_synthetic_drift,
)
from src.quantum_model import (
    QuantumKernelClassifier,
    build_quantum_preprocessor,
    get_stratified_quantum_subset,
)
from src.uncertainty import UncertaintyAlertLayer, UncertaintyAlertReport


# -----------------------------------------------------------------------------
# Cached Model & Data Loaders (Prevents redundant retraining on UI rerun)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_cached_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and cache the raw AI4I 2020 feature matrix and target series."""
    return load_ai4i_dataset()


@st.cache_resource(show_spinner=False)
def get_trained_classical_pipeline(random_state: int = 42):
    """
    Train and cache the classical RBF-SVM baseline pipeline on training data.
    Fitted strictly on training partition to prevent leakage.
    """
    from sklearn.model_selection import train_test_split

    X, y = load_cached_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    pipeline = create_classical_pipeline(svm_params={"random_state": random_state})
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def evaluate_single_sensor_reading(
    pipeline,
    reading_dict: Dict[str, Any],
    confidence_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Evaluate a single incoming sensor reading dictionary using the fitted pipeline.

    Returns structured prediction, probability scores, confidence, and human review status.
    Uses the real, loaded model directly without hardcoding or fabricating results.
    """
    if pipeline is None:
        raise ValueError("Classifier pipeline is None. Cannot run inference.")

    df = pd.DataFrame([reading_dict])
    pred = int(pipeline.predict(df)[0])
    probs = pipeline.predict_proba(df)[0]
    conf = float(np.max(probs))

    alert_layer = UncertaintyAlertLayer(confidence_threshold=confidence_threshold)
    report = alert_layer.evaluate_predictions(
        predictions=[pred],
        probabilities=[probs.tolist()],
    )
    alert_item = report.items[0]

    return {
        "prediction": pred,
        "prediction_label": "Machine Failure (1)" if pred == 1 else "Normal Operation (0)",
        "probabilities": {"Normal (0)": float(probs[0]), "Failure (1)": float(probs[1])},
        "confidence": conf,
        "flag_human_review": alert_item.flag_human_review,
        "review_reason": alert_item.reason,
    }


# -----------------------------------------------------------------------------
# Main Dashboard UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Industrial Fault Detection | Quantum–Classical ML",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            border-left: 5px solid #5B3BA8;
            margin-bottom: 12px;
        }
        .alert-box-failure {
            background-color: #fde8e8;
            color: #9b1c1c;
            padding: 12px 16px;
            border-radius: 6px;
            font-weight: 600;
            border: 1px solid #f8b4b4;
        }
        .alert-box-warning {
            background-color: #fdf6b2;
            color: #723b13;
            padding: 12px 16px;
            border-radius: 6px;
            font-weight: 600;
            border: 1px solid #fce96a;
        }
        .alert-box-normal {
            background-color: #def7ec;
            color: #03543f;
            padding: 12px 16px;
            border-radius: 6px;
            font-weight: 600;
            border: 1px solid #bcf0da;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚙️ Drift-Aware Industrial Fault Detection")
    st.caption(
        "Hybrid Quantum–Classical Machine Learning & Statistical Drift Monitoring | "
        "**Team Phantoms** (VTU-QFF-HC-0005) | Qiskit Fall Fest 2026"
    )

    # Sidebar Navigation & Settings
    st.sidebar.markdown("## ⚛️ Telemetry Controls")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Live Fault Prediction",
            "Sensor Drift Monitoring",
            "Quantum vs. Classical Benchmark",
            "Unified Alert Center",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Safety Thresholds")
    confidence_threshold = st.sidebar.slider(
        "Uncertainty Alert Cutoff (Confidence)",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Predictions with confidence below this threshold are flagged for human operator review.",
    )

    drift_alpha = st.sidebar.slider(
        "Drift Significance Alpha (p-value)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Kolmogorov-Smirnov test significance threshold for flagging distribution shift.",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**System Safeguards:**\n"
        "• Failure cause indicators (TWF, HDF, PWF, OSF, RNF) are strictly excluded from input features.\n"
        "• Preprocessing is fitted exclusively on reference training data.\n"
        "• No quantum advantage is claimed."
    )

    # Load shared baseline data with robust error handling
    try:
        pipeline, X_train, X_test, y_train, y_test = get_trained_classical_pipeline()
    except Exception as e:
        st.error(
            f"❌ Critical Error: Unable to load or initialize the dataset and baseline pipeline. "
            f"Details: {e}"
        )
        st.stop()

    # -------------------------------------------------------------------------
    # PAGE 1: Overview
    # -------------------------------------------------------------------------
    if page == "Overview":
        st.subheader("📋 System Architecture & Operational Overview")
        st.markdown(
            "Welcome to the **Phantoms Industrial Telemetry & Fault Detection System**. "
            "This platform provides early drift warnings and uncertainty-aware alerts "
            "for predictive maintenance, benchmarking classical algorithms against simulated quantum kernels."
        )

        st.markdown("#### Key Telemetry Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ingested Records", "10,000", "AI4I 2020 Benchmark")
        with col2:
            st.metric("Continuous Sensors", "5 Channels", "Thermal, Speed, Torque, Wear")
        with col3:
            st.metric("Failure Imbalance", "3.39%", "339 Failure / 9,661 Normal")
        with col4:
            st.metric("Classical Baseline Model", "RBF-SVM", "Balanced Class Weights")

        st.markdown("---")
        overview_col1, overview_col2 = st.columns([1, 1])

        with overview_col1:
            st.markdown(
                """
                ### Core Problem & Design Objectives
                - **The Challenge:** Real-world sensors drift over time due to mechanical wear, calibration decay,
                  and ambient environmental variations. Classical ML models silently degrade when inputs deviate from training distributions.
                - **Our Approach:**
                  1. **Clean, Leakage-Free Preprocessing:** Standardized sensor inputs with failure cause labels strictly isolated.
                  2. **Balanced Baseline:** Full-scale RBF-SVM calibrated for severe class imbalance.
                  3. **Quantum-Kernel Classifier:** Evaluated on a fair stratified subset to explore non-linear Hilbert space mappings.
                  4. **Drift Detection:** Two-sample Kolmogorov–Smirnov test flagging covariate shifts before catastrophic failures occur.
                  5. **Human-in-the-Loop Review:** Automatic escalation when prediction confidence drops below threshold.
                """
            )

        with overview_col2:
            st.markdown("### Operational Pipeline Status")
            st.success("✅ **Dataset Pipeline:** Active (10,000 samples, UCI AI4I 2020)")
            st.success("✅ **Classical Model:** Operational (Full-scale Balanced RBF-SVM)")
            st.success("✅ **Quantum Kernel:** Validated (Qiskit ZZFeatureMap, Aer/Statevector)")
            st.success("✅ **Drift Monitor:** Active (Unsupervised Two-Sample KS Test)")
            st.success("✅ **Uncertainty Alert:** Active (Confidence-based Human Escalation)")

        st.markdown("---")
        st.markdown("#### Reference Baseline Telemetry Sample (Healthy Training Data)")
        st.dataframe(X_train.head(6), use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 2: Live Fault Prediction
    # -------------------------------------------------------------------------
    elif page == "Live Fault Prediction":
        st.subheader("🔍 Real-Time Sensor Telemetry & Fault Inference")
        st.write("Adjust sensor telemetry inputs to simulate live machine operational state:")

        col1, col2, col3 = st.columns(3)
        with col1:
            product_type = st.selectbox("Product Quality Variant (Type)", ["L", "M", "H"], index=0)
            air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=315.0, value=298.1, step=0.1)
            process_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=325.0, value=308.6, step=0.1)

        with col2:
            rot_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
            torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
            tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=50, step=5)

        with col3:
            st.markdown("#### Simulation Presets")
            preset = st.radio(
                "Quick-load operational scenario:",
                ["Normal Standard Run", "High Heat & High Torque", "Critical Tool Wear (Failure)"],
            )
            if preset == "High Heat & High Torque":
                air_temp = 304.5
                process_temp = 313.8
                torque = 68.0
            elif preset == "Critical Tool Wear (Failure)":
                tool_wear = 240
                torque = 65.0
                rot_speed = 1320

        reading_dict = {
            "Type": product_type,
            "Air temperature": air_temp,
            "Process temperature": process_temp,
            "Rotational speed": rot_speed,
            "Torque": torque,
            "Tool wear": tool_wear,
        }

        st.markdown("---")
        if st.button("🚀 Run Fault Inference", type="primary"):
            try:
                eval_res = evaluate_single_sensor_reading(
                    pipeline, reading_dict, confidence_threshold=confidence_threshold
                )
            except Exception as e:
                st.error(f"Prediction Error: Unable to evaluate reading. Details: {e}")
                eval_res = None

            if eval_res is not None:
                st.markdown("### Inference Results")
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    if eval_res["prediction"] == 1:
                        st.markdown(
                            f"<div class='alert-box-failure'>⚠️ PREDICTED STATUS: {eval_res['prediction_label']}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='alert-box-normal'>✅ PREDICTED STATUS: {eval_res['prediction_label']}</div>",
                            unsafe_allow_html=True,
                        )

                    st.write("")
                    st.metric("Model Decision Confidence", f"{eval_res['confidence']:.1%}")

                    if eval_res["flag_human_review"]:
                        st.markdown(
                            f"<div class='alert-box-warning'>🔔 HUMAN REVIEW ESCALATION REQUIRED:<br>{eval_res['review_reason']}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.success(f"Confidence satisfies automated criteria ({eval_res['confidence']:.1%} >= {confidence_threshold:.1%})")

                with res_col2:
                    st.markdown("#### Probability Distribution")
                    prob_df = pd.DataFrame(
                        list(eval_res["probabilities"].items()),
                        columns=["Class", "Probability"],
                    )
                    st.bar_chart(prob_df.set_index("Class"))
                    st.caption("Note: SVM probabilities are heuristic Platt-scaled estimates and not Bayesian posteriors.")

    # -------------------------------------------------------------------------
    # PAGE 3: Sensor Drift Monitoring
    # -------------------------------------------------------------------------
    elif page == "Sensor Drift Monitoring":
        st.subheader("📡 Statistical Sensor Drift Detection (Two-Sample KS Test)")
        st.write(
            "Monitors whether incoming sensor batches deviate from reference healthy operations "
            "using unsupervised non-parametric Kolmogorov–Smirnov hypothesis testing."
        )

        detector = SensorDriftDetector(alpha=drift_alpha)
        detector.fit_reference(X_train)

        col1, col2 = st.columns([1, 1])
        with col1:
            batch_choice = st.radio(
                "Select Incoming Sensor Batch to Inspect:",
                [
                    "Nominal Operational Test Batch (Unperturbed)",
                    "Controlled Synthetic Environmental Thermal Shift (+2.5σ)",
                    "Controlled Synthetic Severe Tool Wear (+3.0σ)",
                    "Multi-Channel Sensor Drift (Thermal + Tool Wear)",
                ],
            )

        with col2:
            st.info(
                "**Testing Methodology:**\n"
                "• Compares reference empirical CDF with incoming batch empirical CDF.\n"
                "• A feature flags drift when p-value < α.\n"
                "• Drift indicates covariate shift, not definitive equipment failure."
            )

        if batch_choice == "Nominal Operational Test Batch (Unperturbed)":
            test_batch = X_test.copy()
            batch_label = "Nominal Batch"
        elif batch_choice == "Controlled Synthetic Environmental Thermal Shift (+2.5σ)":
            test_batch = inject_synthetic_drift(
                X_test, {"Air temperature": 2.5, "Process temperature": 2.5}
            )
            batch_label = "Synthetic Thermal Shift (+2.5σ)"
        elif batch_choice == "Controlled Synthetic Severe Tool Wear (+3.0σ)":
            test_batch = inject_synthetic_drift(X_test, {"Tool wear": 3.0})
            batch_label = "Synthetic Tool Wear Shift (+3.0σ)"
        else:
            test_batch = inject_synthetic_drift(
                X_test,
                {"Air temperature": 2.5, "Process temperature": 2.5, "Tool wear": 2.0},
            )
            batch_label = "Synthetic Multi-Channel Drift"

        report: DriftReport = detector.detect_drift(test_batch)

        st.markdown("---")
        if report.drift_detected:
            st.markdown(
                f"<div class='alert-box-warning'>⚠️ {report.summary_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='alert-box-normal'>✅ {report.summary_text}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"#### Per-Feature Hypothesis Test Results ({batch_label})")
        table_rows = []
        for feat, res in report.feature_results.items():
            table_rows.append(
                {
                    "Sensor Feature": feat,
                    "KS Statistic": f"{res.statistic:.4f}",
                    "p-value": f"{res.p_value:.2e}",
                    "Reference Mean": f"{res.reference_mean:.2f}",
                    "Batch Mean": f"{res.test_mean:.2f}",
                    "Drift Flag (p < α)": "⚠️ DRIFT" if res.drift_detected else "✅ Normal",
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 4: Quantum vs. Classical Benchmark
    # -------------------------------------------------------------------------
    elif page == "Quantum vs. Classical Benchmark":
        st.subheader("⚛️ Model Comparison: Classical RBF-SVM vs. Quantum-Kernel SVM")
        st.write(
            "Empirical comparison between the full classical baseline and a controlled quantum simulation experiment."
        )

        st.warning(
            "⚠️ **Evaluation & Sample-Size Notice:**\n\n"
            "• **Controlled Sample Size:** The head-to-head quantum comparison is conducted on a reproducible subset of **only 80 samples** "
            "(60 training / 20 testing), because computing an $N \\times N$ quantum kernel matrix scales quadratically as $O(N^2)$ pairwise circuit simulations.\n"
            "• **Failure Detection Observation:** Due to severe industrial class imbalance (~3.4% failure rate), the 20-sample test partition contained only 1 failure instance. "
            "**Neither the Classical SVM nor the Quantum-Kernel SVM detected this failure on the small test partition (both produced 0.00% failure recall and precision).**\n"
            "• **No Quantum Superiority:** The higher overall accuracy of the quantum model (95.0% vs 90.0%) is purely an artifact of classifying all normal samples correctly in a 19-normal/1-failure test set. "
            "**This sample size is far too small to establish statistical superiority or claim quantum advantage.**"
        )

        tab1, tab2 = st.tabs(["Controlled Head-to-Head Comparison", "Full Classical Baseline (10,000 Samples)"])

        with tab1:
            st.markdown("#### Controlled Stratified Subset Benchmark (80 Samples: 60 Train / 20 Test)")
            st.write(
                "Both models evaluated on the **exact same stratified partition and PCA-compressed sensor features**:"
            )

            comp_df = pd.DataFrame(
                {
                    "Metric": [
                        "Train Samples",
                        "Test Samples (Normal / Failure)",
                        "Input Dimension",
                        "Kernel Architecture",
                        "Accuracy",
                        "Failure Recall",
                        "Failure Precision",
                        "Macro Avg F1",
                        "Training / Kernel Time",
                    ],
                    "Classical SVM (Subset)": [
                        "60",
                        "20 (19 Normal / 1 Failure)",
                        "4 (PCA)",
                        "RBF Kernel",
                        "90.00%",
                        "0.00% (0 / 1)",
                        "0.00%",
                        "0.4737",
                        "0.001s",
                    ],
                    "Quantum-Kernel SVM": [
                        "60",
                        "20 (19 Normal / 1 Failure)",
                        "4 Qubits",
                        "ZZFeatureMap (reps=1)",
                        "95.00%",
                        "0.00% (0 / 1)",
                        "0.00%",
                        "0.4872",
                        "6.528s",
                    ],
                }
            )
            st.dataframe(comp_df.set_index("Metric"), use_container_width=True)

            st.markdown("##### On-Demand Quantum Simulation")
            st.caption("Trigger an explicit re-evaluation of the Qiskit quantum kernel simulation on a small batch:")
            if st.button("Run Live Quantum Kernel Simulation (N=40 samples)"):
                with st.spinner("Computing Quantum Fidelity Gram matrices on Aer/Statevector simulator..."):
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score

                    X_sub, y_sub = get_stratified_quantum_subset(X_train, y_train, n_samples=40, random_state=42)
                    X_tr_q, X_te_q, y_tr_q, y_te_q = train_test_split(
                        X_sub, y_sub, test_size=0.25, random_state=42, stratify=y_sub
                    )
                    q_pre = build_quantum_preprocessor(n_components=4, random_state=42)
                    q_pre.fit(X_tr_q)
                    X_tr_enc = q_pre.transform(X_tr_q)
                    X_te_enc = q_pre.transform(X_te_q)

                    q_clf = QuantumKernelClassifier(n_qubits=4, reps=1, class_weight="balanced", random_state=42)
                    q_clf.fit(X_tr_enc, y_tr_q.values)
                    q_preds = q_clf.predict(X_te_enc)
                    q_acc = accuracy_score(y_te_q, q_preds)

                    st.success(
                        f"Live Quantum Kernel Simulation complete! "
                        f"Kernel evaluation time: {q_clf.train_time_:.2f}s | Test Accuracy: {q_acc:.1%}"
                    )

        with tab2:
            st.markdown("#### Full Scale Baseline (10,000 Operational Records)")
            base_col1, base_col2, base_col3, base_col4 = st.columns(4)
            base_col1.metric("Overall Accuracy", "91.50%")
            base_col2.metric("Failure Recall (Sensitivity)", "89.71%", "61 / 68 Test Failures Caught")
            base_col3.metric("Failure Precision", "27.23%", "Expected Imbalance Trade-off")
            base_col4.metric("Training Time", "2.91s", "Full Scale Pipeline")

            st.write("##### Confusion Matrix (2,000 Test Records):")
            cm_df = pd.DataFrame(
                [[1769, 163], [7, 61]],
                columns=["Pred Normal (0)", "Pred Failure (1)"],
                index=["Actual Normal (0)", "Actual Failure (1)"],
            )
            st.dataframe(cm_df)

    # -------------------------------------------------------------------------
    # PAGE 5: Unified Alert Center
    # -------------------------------------------------------------------------
    elif page == "Unified Alert Center":
        st.subheader("🚨 Unified Industrial Alert Management Center")
        st.write(
            "Consolidates critical machine events, categorizing alerts across three distinct functional tiers:"
        )

        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown(
                """
                <div class='alert-box-failure'>
                🔴 TIER 1: EQUIPMENT FAULT ALERTS<br>
                Trigger: Model predicts machine failure.<br>
                Action: Immediate maintenance dispatch.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with colB:
            st.markdown(
                """
                <div class='alert-box-warning'>
                🟡 TIER 2: SENSOR DRIFT WARNINGS<br>
                Trigger: Distribution shift detected via KS-test.<br>
                Action: Sensor recalibration inspection.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with colC:
            st.markdown(
                """
                <div class='alert-box-warning' style='background-color:#ebf5ff; color:#1e429f; border-color:#a4cafe;'>
                🔵 TIER 3: UNCERTAINTY ESCALATIONS<br>
                Trigger: Confidence < threshold.<br>
                Action: Expert operator manual verification.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### Test Set Uncertainty Audit")
        y_pred = pipeline.predict(X_test)
        y_probs = pipeline.predict_proba(X_test)

        alert_layer = UncertaintyAlertLayer(confidence_threshold=confidence_threshold)
        alert_rep: UncertaintyAlertReport = alert_layer.evaluate_predictions(y_pred, probabilities=y_probs)

        st.write(f"**Audit Result:** {alert_rep.summary_text}")

        flagged_items = [item for item in alert_rep.items if item.flag_human_review]
        if flagged_items:
            audit_rows = []
            for item in flagged_items[:15]:
                audit_rows.append(
                    {
                        "Sample ID": item.index,
                        "Predicted Class": "Failure (1)" if item.prediction == 1 else "Normal (0)",
                        "Confidence Score": f"{item.confidence:.1%}",
                        "Escalation Reason": item.reason,
                    }
                )
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)


if __name__ == "__main__":
    main()
