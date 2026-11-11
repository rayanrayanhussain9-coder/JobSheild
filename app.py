"""
JobShield — Explainable Job Advertisement Fraud Screening
Streamlit Application
"""
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.home import render_about_intro
from src.particle_home import render_particle_home
from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    EVAL_METRICS_PATH,
    BASELINE_METRICS_PATH,
    CALIBRATION_CURVE_PATH,
    CONFUSION_MATRIX_PATH,
    SPLIT_MANIFEST_PATH,
    DATA_QUALITY_REPORT_PATH,
    MIN_INPUT_CHARS,
    MAX_INPUT_CHARS,
)
from src.predictor import (
    JobShieldPredictor,
    JobAdvertisementInput,
    JobShieldResult,
)
from src.report_generator import generate_json_report, generate_html_report

logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="JobShield — Explainable Fraud Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_predictor() -> Tuple[Optional[JobShieldPredictor], Optional[str]]:
    """Loads and caches the JobShield predictor. Returns (predictor, error_message)."""
    if not MODEL_PATH.exists():
        return None, (
            f"Trained model artifact not found at `{MODEL_PATH}`. "
            "Please run the training pipeline first: `python src/train.py`."
        )
    try:
        predictor = JobShieldPredictor(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
        return predictor, None
    except Exception as e:
        return None, f"Failed to initialize JobShield model artifact: {e}"


def load_json_artifact(path: Path) -> dict:
    """Helper to load JSON artifacts safely."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error reading %s: %s", path, e)
    return {}


def render_feature_chart(explanation: dict):
    """Renders a Plotly horizontal bar chart of local linear feature attributions."""
    top_susp = explanation.get("top_suspicious", [])
    top_legit = explanation.get("top_legitimate", [])

    items = []
    for f in top_legit:
        items.append({
            "Feature": f["feature"],
            "Contribution": f["contribution"],
            "Direction": "Pushes Legitimate",
            "Color": "#10b981",  # Emerald Green
        })
    for f in top_susp:
        items.append({
            "Feature": f["feature"],
            "Contribution": f["contribution"],
            "Direction": "Pushes Suspicious",
            "Color": "#ef4444",  # Crimson Red
        })

    if not items:
        st.info("No influential vocabulary features identified for this text.")
        return

    df_chart = pd.DataFrame(items)
    df_chart = df_chart.sort_values(by="Contribution", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_chart["Contribution"],
        y=df_chart["Feature"],
        orientation="h",
        marker=dict(color=df_chart["Color"]),
        text=[f"{v:+.4f}" for v in df_chart["Contribution"]],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Linear Weight Contribution: %{x:+.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Exact Local Feature Attributions (x_i · w_i)",
        xaxis_title="Contribution to Suspicion Logit (<0 Legitimate, >0 Suspicious)",
        yaxis_title="Word / N-gram Feature",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")


def render_screening_results(result: JobShieldResult, ad_input: JobAdvertisementInput, key_prefix: str = "main"):
    """Displays structured screening output: prediction badge, score, rules, and download."""
    if result.status == "INADEQUATE_INPUT":
        st.warning(f"⚠️ **Input Adequacy Notice:** {result.input_adequacy['message']}")
        st.caption(
            "JobShield enforces strict input adequacy guards. A model score will not be fabricated "
            "for empty, nonsensical, or overly brief inputs."
        )
        return

    # Status & Score Banner
    st.markdown("---")
    col_pred, col_score, col_thresh = st.columns([1.5, 1.5, 1.2])

    with col_pred:
        if result.model_prediction == "SUSPICIOUS_PATTERN":
            st.error("🚨 **Prediction: Suspicious Pattern**")
        elif result.model_prediction == "NEEDS_REVIEW":
            st.warning("⚠️ **Prediction: Needs Review (Borderline)**")
        else:
            st.success("✅ **Prediction: Low Risk Pattern**")

    with col_score:
        score_val = result.suspicion_score if result.suspicion_score is not None else 0.0
        st.metric(
            label="Model Suspicion Score",
            value=f"{score_val:.4f}",
            help="Calibrated probability of matching historical fraud patterns from training data.",
        )

    with col_thresh:
        st.metric(
            label="Operating Threshold",
            value=f"{result.operating_threshold:.2f}",
            help="Tuned on validation partition to optimize fraud F1 score.",
        )

    # Synthesis statement
    st.info(f"**Assessment Summary:** {result.synthesis_summary}")

    # Disagreement callout if model & rules conflict
    if result.disagreement_notice:
        st.warning(f"⚡ {result.disagreement_notice}")

    # Rule-Based Warning Section
    st.markdown("### 🔍 Rule-Based Warning Engine Checks")
    rules = result.rule_results or {}
    warnings = rules.get("warnings", [])

    if warnings:
        for idx, w in enumerate(warnings):
            with st.expander(f"🚩 Red Flag: {w['category']} (Severity: {w['severity']})", expanded=True):
                st.markdown(f"**Matched text:** `\"{w['matched_text']}\"`")
                st.markdown(f"**Explanation:** {w['explanation']}")
                st.markdown(f"💡 **Guidance:** {w['guidance']}")
    else:
        st.success("No explicit rule-based warning signs detected (no upfront fee requests or urgent wire transfer demands).")

    # Local Explanations
    if result.explanation:
        st.markdown("### 🔬 Faithful Model Feature Explanations")
        st.caption(
            f"Method: {result.explanation['method']}. Intercept base rate: `{result.explanation['intercept']:+.4f}`. "
            "These weights reflect empirical statistical associations learned during training, not proof of intent."
        )
        render_feature_chart(result.explanation)

    # Missing Information Section
    if result.missing_info_observations:
        with st.expander("ℹ️ Missing Information Observations", expanded=False):
            for obs in result.missing_info_observations:
                st.markdown(f"- **{obs['field']}** ({obs['severity']} Priority): {obs['observation']}")

    # Download Report Options
    st.markdown("### 📥 Download Screening Report")
    json_data = generate_json_report(ad_input, result)
    html_data = generate_html_report(ad_input, result)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📄 Download Full Report (HTML)",
            data=html_data,
            file_name=f"jobshield_report_{key_prefix}.html",
            mime="text/html",
            key=f"dl_html_{key_prefix}",
        )
    with col_dl2:
        st.download_button(
            label="📊 Download Raw Audit Data (JSON)",
            data=json_data,
            file_name=f"jobshield_audit_{key_prefix}.json",
            mime="application/json",
            key=f"dl_json_{key_prefix}",
        )


def history_status(result: JobShieldResult) -> str:
    if (result.rule_results or {}).get("warnings") or result.model_prediction == "SUSPICIOUS_PATTERN":
        return "❌ Flagged"
    if result.model_prediction == "NEEDS_REVIEW":
        return "🟡 Needs review"
    return "✅ Low risk"


def remember_screening(ad: JobAdvertisementInput, result: JobShieldResult):
    if result.status == "ANALYSIS_COMPLETE":
        history = st.session_state.screening_history
        history[:] = [(old_ad, old_result) for old_ad, old_result in history if old_ad != ad]
        history.insert(0, (ad, result))


@st.dialog("Saved screening", width="large")
def open_screening(ad: JobAdvertisementInput, result: JobShieldResult):
    st.subheader(history_status(result))
    st.text(ad.build_full_text())
    render_screening_results(result, ad, key_prefix="history")


# Sidebar Setup
st.session_state.setdefault("screening_history", [])
predictor, err = get_predictor()
st.html("""
<style>
[data-testid="stMainBlockContainer"] {max-width:none;padding:14px 0 0;width:100%}
.st-key-main_nav {--page-gutter:clamp(16px,2.4vw,44px)}
.st-key-main_nav [role="tablist"] {
  box-sizing:border-box;position:relative;isolation:isolate;gap:2px;
  width:max-content;max-width:calc(100% - 20px);margin:0 auto;padding:4px;
  border:1px solid #e2e0e8;border-radius:999px;background:#faf9fc;
  box-shadow:0 2px 5px #23174308,0 6px 20px #291c3908;
  overflow-x:auto;scrollbar-width:none;
}
.st-key-main_nav [role="tab"] {
  flex:0 0 auto;white-space:nowrap;height:38px;min-height:0;padding:0 17px;
  border:0;border-radius:999px;color:#514d5b;transition:background .16s,color .16s;
}
.st-key-main_nav [role="tab"] p {margin:0;font:550 14px/1.25 system-ui;letter-spacing:-.02em}
.st-key-main_nav [role="tab"]:hover {background:#efedf4;color:#251d3c}
.st-key-main_nav [role="tab"][aria-selected="true"] {color:white;background:#282237}
.st-key-main_nav [role="tab"]:focus-visible {outline:2px solid #7964d1;outline-offset:-2px}
.st-key-main_nav .react-aria-SelectionIndicator {display:none}
.st-key-main_nav [role="tabpanel"] {border:0;padding:clamp(20px,3vw,44px) var(--page-gutter)}
.st-key-main_nav [role="tabpanel"][id$="-tabpanel-0"] {padding:0}
.js-home-end {padding-inline:var(--page-gutter,24px)}
@media (max-width:620px) {
  .st-key-main_nav [role="tablist"] {gap:0;max-width:calc(100% - 12px);padding:3px}
  .st-key-main_nav [role="tab"] {padding:0 clamp(5px,1.45vw,9px);height:36px}
  .st-key-main_nav [role="tab"] p {font-size:clamp(10.5px,2.9vw,13px)}
}
[data-testid="stAppDeployButton"],[data-testid="stMainMenu"] {display:none}
[data-testid="stHeader"] {background:transparent;pointer-events:none;height:0}
[data-testid="stHeader"] [data-testid="stToolbar"],
[data-testid="stHeader"] [data-testid="stToolbar"] * {pointer-events:none}
[data-testid="stHeader"] [data-testid="stToolbar"] [data-testid="stExpandSidebarButton"] {pointer-events:auto;background:white}
[data-testid="stSidebarCollapseButton"] {opacity:1;visibility:visible}
[data-testid="stSidebarUserContent"] {padding-top:0}
[data-testid="stSidebarUserContent"] h2 {font-size:1.45rem;letter-spacing:-.04em}
@media (max-width:760px) {
  body:has([data-testid="stExpandSidebarButton"]) .st-key-main_nav [role="tablist"] {
    max-width:calc(100% - 52px);margin-left:46px
  }
}
@media (prefers-reduced-motion:no-preference) {
  /* Slide the new panel and its layout space together when entering the checker. */
  @media (min-width:769px) {
    [data-testid="stSidebar"] {
      transition:margin-inline-start 320ms cubic-bezier(.22,.75,.24,1),transform .3s,min-width .3s,max-width .3s;
    }
    @starting-style {
      [data-testid="stSidebar"][data-collapsed="false"] {margin-inline-start:-300px}
    }
  }
  @media (max-width:768px) {
    [data-testid="stSidebar"] {transition:translate .28s ease-out,transform .3s,min-width .3s,max-width .3s}
    @starting-style {
      [data-testid="stSidebar"][data-collapsed="false"] {translate:-100% 0}
    }
  }
}
@media (prefers-reduced-motion:reduce) {
  [data-testid="stSidebar"] {transition:none!important}
}
[data-testid="stSidebar"] {border-right:1px solid #dadae2}
[data-testid="stForm"], [data-testid="stExpander"] {background:#fff;border-radius:12px}
</style>
""")
tab_home, tab_analyze, tab_compare, tab_performance, tab_about = st.tabs(
    ["Home", "Check a job", "Compare jobs", "Model performance", "About"],
    key="main_nav", on_change="rerun",
)
with tab_home:
    if tab_home.open:
        render_particle_home()
        st.html(Path(__file__).parent / "src" / "home_footer.html")

history_panel = None
if tab_analyze.open or tab_compare.open:
    with st.sidebar:
        st.header("Screening history")
        history_panel = st.container()
        st.divider()
        if err:
            st.error(err)
        else:
            with st.expander("About this screening"):
                st.caption("JobShield checks the wording of a job posting. Verify the employer independently before sharing personal information.")
                meta = predictor.metadata
                st.caption(f"Model {meta.get('model_version', '1.0.0')} · Decision threshold {meta.get('decision_threshold', 0.38)}")


# ==========================================
# TAB 1: ANALYZE JOB
# ==========================================
with tab_analyze:
    st.header("Screen a Job Advertisement")
    st.markdown(
        "Paste the details of a job posting below to screen for known fraud indicators, "
        "suspicious fee demands, and linguistic patterns."
    )

    if predictor is None:
        st.error(f"Cannot run analysis: {err}")
    else:
        # Pre-fill sample button for quick testing
        col_sm1, col_sm2, col_sm3 = st.columns([1, 1, 3])
        from_home = st.session_state.pop("hero_sample", False)
        sample_choice = "SCAM" if from_home else None
        with col_sm1:
            if st.button("Load Legitimate Sample"):
                sample_choice = "LEGIT"
        with col_sm2:
            if st.button("Load Suspect Fee Sample"):
                sample_choice = "SCAM"

        default_title = ""
        default_company = ""
        default_desc = ""
        default_reqs = ""
        default_benefits = ""

        if sample_choice == "LEGIT":
            default_title = "Junior Python Software Engineer"
            default_company = "NovaTech Digital"
            default_desc = (
                "NovaTech Digital is seeking an entry-level Python developer to join our backend platform team. "
                "You will write modular code, write unit tests, review pull requests, and collaborate with engineers. "
                "We offer structured mentorship and career growth."
            )
            default_reqs = "Bachelor's degree in CS/IT or equivalent project portfolio. Solid understanding of Python, Git, and SQL."
            default_benefits = "Health insurance, 401(k) matching, 20 days paid time off, education stipend."
        elif sample_choice == "SCAM":
            default_title = "Remote Data Entry Assistant"
            default_company = "Global FastCareers Inc."
            default_desc = (
                "Urgent requirement! Start immediately from home and earn $3,800 weekly. No experience or interview required. "
                "Direct hiring guaranteed upon receipt of application. Selected applicants must pay a refundable registration "
                "and onboarding fee of $120 via wire transfer to confirm their kit and software license."
            )
            default_reqs = "Basic typing, smartphone or computer, immediate availability."
            default_benefits = "High daily compensation, flexible hours, work in pajamas."

        with st.form("single_ad_form"):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                input_title = st.text_input("Job Title", value=default_title, help="Title of the posted position.")
            with col_t2:
                input_company = st.text_input("Company Name (Context)", value=default_company, help="Employer name.")

            input_desc = st.text_area(
                "Job Description * (Main advertisement text)",
                value=default_desc,
                height=180,
                help=f"Must contain between {MIN_INPUT_CHARS} and {MAX_INPUT_CHARS:,} characters.",
            )

            with st.expander("Optional Supporting Fields (Requirements, Benefits)", expanded=False):
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    input_reqs = st.text_area("Requirements & Qualifications", value=default_reqs, height=90)
                with col_s2:
                    input_benefits = st.text_area("Benefits & Compensation", value=default_benefits, height=90)

            submitted = st.form_submit_button("🛡️ Screen Advertisement", type="primary", width="stretch")

        if submitted or from_home:
            ad_obj = JobAdvertisementInput(
                title=input_title,
                company_name=input_company,
                description=input_desc,
                requirements=input_reqs,
                benefits=input_benefits,
            )
            with st.spinner("Analyzing text patterns and evaluating safety rules..."):
                analysis_res = predictor.predict(ad_obj)
            remember_screening(ad_obj, analysis_res)
            render_screening_results(analysis_res, ad_obj, key_prefix="single")


# ==========================================
# TAB 2: COMPARE JOBS
# ==========================================
with tab_compare:
    st.header("Side-by-Side Job Comparison")
    st.markdown(
        "Compare two advertisements simultaneously to inspect differing risk profiles, "
        "rule warnings, and feature attributions."
    )

    if predictor is None:
        st.error(f"Cannot run comparison: {err}")
    else:
        col_ad_a, col_ad_b = st.columns(2)

        with col_ad_a:
            st.subheader("Job Posting A")
            title_a = st.text_input("Title (A)", value="Associate Backend Engineer", key="cmp_title_a")
            company_a = st.text_input("Company (A)", value="Strata Systems", key="cmp_comp_a")
            desc_a = st.text_area(
                "Description (A)",
                value=(
                    "Strata Systems is hiring an Associate Backend Engineer to develop internal APIs in Python. "
                    "You will work alongside senior engineers, participate in agile standups, and write test suites. "
                    "Requirements include Python familiarity, computer science fundamentals, and good communication. "
                    "Benefits: Competitive salary, health insurance, paid leave."
                ),
                height=220,
                key="cmp_desc_a",
            )

        with col_ad_b:
            st.subheader("Job Posting B")
            title_b = st.text_input("Title (B)", value="Remote Executive Typist", key="cmp_title_b")
            company_b = st.text_input("Company (B)", value="Express Online Careers", key="cmp_comp_b")
            desc_b = st.text_area(
                "Description (B)",
                value=(
                    "Urgent vacancy! Work from home as an executive typist and earn $4,000 per week. "
                    "No previous experience or resume required. Guaranteed hiring. Candidates must pay a mandatory "
                    "refundable training fee of $150 via wire transfer to confirm their workstation package."
                ),
                height=220,
                key="cmp_desc_b",
            )

        btn_compare = st.button("⚖️ Compare Both Advertisements", type="primary", width="stretch")

        if btn_compare:
            ad_a = JobAdvertisementInput(title=title_a, company_name=company_a, description=desc_a)
            ad_b = JobAdvertisementInput(title=title_b, company_name=company_b, description=desc_b)

            res_a = predictor.predict(ad_a)
            res_b = predictor.predict(ad_b)
            remember_screening(ad_a, res_a)
            remember_screening(ad_b, res_b)

            st.markdown("---")
            st.subheader("Comparative Results")
            col_res_a, col_res_b = st.columns(2)

            with col_res_a:
                st.markdown(f"#### Results for Job A: *{title_a}*")
                render_screening_results(res_a, ad_a, key_prefix="cmp_a")

            with col_res_b:
                st.markdown(f"#### Results for Job B: *{title_b}*")
                render_screening_results(res_b, ad_b, key_prefix="cmp_b")


# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab_performance:
    st.header("Rigorous Model Evaluation & Provenance")
    st.markdown(
        "All metrics below reflect genuine evaluations executed on held-out partitions "
        "from the Employment Scam Aegean Dataset (EMSCAD). The final test set was strictly locked "
        "and untouched until model selection was frozen."
    )

    eval_data = load_json_artifact(EVAL_METRICS_PATH)
    baseline_data = load_json_artifact(BASELINE_METRICS_PATH)
    manifest_data = load_json_artifact(SPLIT_MANIFEST_PATH)
    calibration_data = load_json_artifact(CALIBRATION_CURVE_PATH)

    if not eval_data:
        st.warning("Evaluation artifacts not found. Please run `python src/train.py`.")
    else:
        # Split Summary Cards
        st.subheader("1. Partitioning & Group-Aware Separation")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("Total Records", f"{manifest_data.get('total_records', 17880):,}")
        with col_m2:
            tr = manifest_data.get("train", {})
            st.metric("Train Split (70%)", f"{tr.get('total_records', 0):,}", f"Fraud: {tr.get('fraud_rate_pct', 0)}%")
        with col_m3:
            va = manifest_data.get("val", {})
            st.metric("Validation Split (15%)", f"{va.get('total_records', 0):,}", f"Fraud: {va.get('fraud_rate_pct', 0)}%")
        with col_m4:
            te = manifest_data.get("test", {})
            st.metric("Final Test Split (15%)", f"{te.get('total_records', 0):,}", f"Fraud: {te.get('fraud_rate_pct', 0)}%")

        st.caption(
            "🔒 **Data Leakage Control:** All duplicate and template ad copies share identical group IDs "
            "and were constrained to the exact same partition. Cross-partition group overlap is strictly 0."
        )

        st.markdown("---")
        # Validation vs Locked Test Metrics Table
        st.subheader("2. Performance on Held-Out Partitions")
        val_m = eval_data.get("validation_metrics", {})
        test_m = eval_data.get("final_test_metrics", {})

        metrics_comparison = [
            {
                "Partition": "Validation Set (Tuning & Selection)",
                "Decision Threshold": val_m.get("threshold", 0.38),
                "Fraud Precision": f"{val_m.get('precision', 0):.4f}",
                "Fraud Recall": f"{val_m.get('recall', 0):.4f}",
                "Fraud F1 Score": f"{val_m.get('f1', 0):.4f}",
                "Average Precision (PR-AUC)": f"{val_m.get('average_precision', 0):.4f}",
                "ROC-AUC": f"{val_m.get('roc_auc', 0):.4f}",
            },
            {
                "Partition": "Final Locked Test Set (Untouched)",
                "Decision Threshold": test_m.get("threshold", 0.38),
                "Fraud Precision": f"{test_m.get('precision', 0):.4f}",
                "Fraud Recall": f"{test_m.get('recall', 0):.4f}",
                "Fraud F1 Score": f"{test_m.get('f1', 0):.4f}",
                "Average Precision (PR-AUC)": f"{test_m.get('average_precision', 0):.4f}",
                "ROC-AUC": f"{test_m.get('roc_auc', 0):.4f}",
            },
        ]
        st.table(pd.DataFrame(metrics_comparison))

        # Confusion Matrices
        st.subheader("3. Confusion Matrix Breakdown")
        col_cm_v, col_cm_t = st.columns(2)

        def make_cm_fig(cm_dict, title):
            tn = cm_dict.get("tn", 0)
            fp = cm_dict.get("fp", 0)
            fn = cm_dict.get("fn", 0)
            tp = cm_dict.get("tp", 0)
            z = [[tn, fp], [fn, tp]]
            fig = px.imshow(
                z,
                labels=dict(x="Predicted Class", y="True Ground Truth", color="Count"),
                x=["Legitimate (0)", "Fraudulent (1)"],
                y=["Legitimate (0)", "Fraudulent (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
            )
            fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=35, b=10))
            return fig

        with col_cm_v:
            st.plotly_chart(make_cm_fig(val_m.get("confusion_matrix", {}), "Validation Confusion Matrix"), width="stretch")

        with col_cm_t:
            st.plotly_chart(make_cm_fig(test_m.get("confusion_matrix", {}), "Final Test Confusion Matrix"), width="stretch")

        # Baseline Comparison
        if baseline_data and isinstance(baseline_data, list):
            st.subheader("4. Baseline vs Candidate Comparison (Validation)")
            records = []
            for b in baseline_data:
                name = b.get("candidate_name", "candidate")
                records.append({
                    "Candidate Pipeline": name,
                    "Precision": b.get("precision", 0.0),
                    "Recall": b.get("recall", 0.0),
                    "F1 Score": b.get("f1", 0.0),
                    "PR-AUC": b.get("average_precision", 0.0),
                })
            st.dataframe(pd.DataFrame(records), width="stretch")

        # Calibration Curve
        if calibration_data and "binned_true_probabilities" in calibration_data:
            st.subheader("5. Probability Calibration Curve")
            st.caption(calibration_data.get("interpretation", ""))
            y_true_p = calibration_data["binned_true_probabilities"]
            y_pred_p = calibration_data["binned_mean_predicted_probabilities"]

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect Calibration (y = x)",
            ))
            fig_cal.add_trace(go.Scatter(
                x=y_pred_p, y=y_true_p,
                mode="lines+markers",
                marker=dict(color="#2563eb", size=8),
                name="Calibrated Model",
            ))
            fig_cal.update_layout(
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Empirical Fraction of Positives (Fraud)",
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_cal, width="stretch")


# ==========================================
# TAB 4: ABOUT & METHODOLOGY
# ==========================================
with tab_about:
    if tab_about.open:
        render_about_intro()
    with st.expander("Project background and methodology"):
        st.markdown(
            r"""
            ### 🎯 Problem Statement & Academic Context
            Online recruitment fraud (fake job postings) has surged with automated job aggregators. 
            Deceptive listings inflict direct financial damage through upfront registration and training fees, 
            and compromise identity data. 
            **JobShield** is an AI Applications Lab project designed to provide **faithful, explainable screening** 
            of job advertisement text without black-box hallucinations.

            ---

            ### 🏗️ Technical Architecture & Design Decisions
            1. **Dual-Engine Architecture:**
               - **Rule Engine:** High-precision deterministic checks for upfront fees, wire transfers, and identity requests. 
                 Handles negation explicitly (e.g. *"We never charge recruitment fees"* is suppressed).
               - **Statistical Linear Model:** Word and subword TF-IDF n-grams (1–2) paired with a Calibrated Linear Classifier 
                 optimized for imbalanced text classification.
            2. **Explainability by Design:**
               - Uses exact mathematical local attributions ($x_i \cdot w_i$).
               - Separates vocabulary into features that push toward suspicion and features that push toward legitimacy.
               - Explicitly displays the model's base intercept rate.
            3. **Data Hygiene & Anti-Leakage Protocol:**
               - Group-stratified partitioning using template text hashing ensures duplicate and repeated postings 
                 never leak across training, validation, and test sets.
               - Final test set was locked prior to hyperparameter selection and threshold tuning.

            ---

            ### 📊 Dataset Provenance
            - **Source:** University of the Aegean Employment Scam Aegean Dataset (EMSCAD).
            - **Reference:** S. Vidros, C. Kolias, G. Kambourakis, L. Akoglu, *"Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset,"* Future Internet, 2017.
            - **Size:** 17,880 verified job postings (17,014 legitimate, 866 fraudulent ~ 4.84% positive class).

            ---

            ### 🛡️ Product Boundaries & Limitations
            - **Text Screening Only:** JobShield evaluates language patterns. It does not certify corporate registration numbers, verify recruiter emails against DNS records, or contact company HR.
            - **Adversarial Scams:** A carefully crafted scam listing that mimics standard corporate jargon without mentioning upfront payments may receive a low text suspicion score.
            - **Honest Scoring:** The score is a calibrated measure of similarity to historical scam patterns, NOT a real-world certainty certificate.

            ---

            ### 🎓 Quick Viva Q&A
            - **Q: Why not use a Deep Learning Transformer (BERT/LLM)?**  
              *A:* For this domain, linear models on n-gram TF-IDF deliver competitive F1 (>0.83) with sub-10ms CPU inference, zero API costs, and exact, un-approximated feature explanations without surrogate approximations.
            - **Q: How did you handle the 95:5 class imbalance?**  
              *A:* We evaluated class weighting schemes and decision threshold optimization on the PR curve, tuning the threshold to maximize fraud-class F1 rather than chasing misleading 95% raw accuracy.
            - **Q: Why separate the rule engine from the ML model?**  
              *A:* Machine learning captures diffuse statistical vocabulary, while critical red flags (e.g., asking for money via wire transfer) require deterministic detection. Blending them into an opaque composite probability hides conflicting signals.
            """
        )


if history_panel is not None:
    with history_panel:
        history = st.session_state.screening_history
        if history:
            st.caption("This session · newest first")
            for index, (ad, result) in enumerate(history):
                title = ad.title.strip() or " ".join(ad.description.split())[:60]
                label = f"{history_status(result)} · {title}"
                if st.button(label, key=f"history_entry_{index}", width="stretch"):
                    open_screening(ad, result)
        else:
            st.caption("Jobs you screen or compare will appear here during this session.")
