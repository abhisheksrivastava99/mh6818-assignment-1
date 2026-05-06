from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from credit_decisioning import (
    ESCALATE_DTI,
    HIGH_REVIEW_DTI,
    MANUAL_REVIEW,
    RECOMMEND_APPROVE,
    RECOMMEND_REJECT,
    build_credit_decision,
)


st.set_page_config(
    page_title="Credit Decision Support",
    page_icon="🏦",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent
MODEL_BUNDLE_PATH = APP_DIR / "credit_default_threshold_xgb_bundle.joblib"
NEW_APPLICATIONS_PATH = APP_DIR / "data" / "new-applications.csv"
FAIRNESS_SUMMARY_PATH = APP_DIR / "fairness_summary.json"

RAW_COLUMNS = [
    "Id",
    "Home Ownership",
    "Annual Income",
    "Years in current job",
    "Tax Liens",
    "Number of Open Accounts",
    "Years of Credit History",
    "Maximum Open Credit",
    "Number of Credit Problems",
    "Months since last delinquent",
    "Bankruptcies",
    "Purpose",
    "Term",
    "Current Loan Amount",
    "Current Credit Balance",
    "Monthly Debt",
    "Credit Score",
]

CATEGORICAL_COLUMNS = [
    "Home Ownership",
    "Years in current job",
    "Purpose",
    "Term",
]

ROLE_APPLICANT = "Applicant"
ROLE_EMPLOYEE = "Bank Employee"

FIELD_WIDGET_KEYS = {column: f"field_{column.lower().replace(' ', '_')}" for column in RAW_COLUMNS}

RECOMMENDATION_STYLES = {
    RECOMMEND_APPROVE: {
        "background": "#e8f7ee",
        "border": "#1f7a4d",
        "text": "#14532d",
    },
    MANUAL_REVIEW: {
        "background": "#fff7e6",
        "border": "#b7791f",
        "text": "#8a5a13",
    },
    RECOMMEND_REJECT: {
        "background": "#fdecec",
        "border": "#c53030",
        "text": "#7f1d1d",
    },
}


@st.cache_resource
def load_model_bundle() -> Dict[str, object]:
    return joblib.load(MODEL_BUNDLE_PATH)


@st.cache_data
def load_raw_application_data() -> pd.DataFrame:
    return pd.read_csv(NEW_APPLICATIONS_PATH)


def load_fairness_summary() -> Optional[Dict[str, object]]:
    if not FAIRNESS_SUMMARY_PATH.exists():
        return None
    return json.loads(FAIRNESS_SUMMARY_PATH.read_text())


def ordered_unique(series: pd.Series) -> List[str]:
    values = []
    seen = set()
    for value in series.dropna():
        string_value = str(value)
        if string_value not in seen:
            seen.add(string_value)
            values.append(string_value)
    return values


def build_field_metadata(raw_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    for column in RAW_COLUMNS:
        if column in CATEGORICAL_COLUMNS:
            metadata[column] = {
                "type": "categorical",
                "options": ordered_unique(raw_df[column]),
                "default": ordered_unique(raw_df[column])[0],
            }
        else:
            if column == "Id":
                default_value = int(raw_df[column].max() + 1)
                step = 1
            else:
                default_value = float(raw_df[column].median())
                step = 1.0
            metadata[column] = {
                "type": "numeric",
                "default": default_value,
                "step": step,
            }
    return metadata


def job_tenure_to_years(value: object) -> float:
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == "< 1 year":
        return 0.5
    if value == "10+ years":
        return 10.0
    if value and value[0].isdigit():
        return float(value.split()[0])
    return np.nan


def apply_deterministic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    working_df["income_missing_flag"] = working_df["Annual Income"].isna().astype(int)
    working_df["credit_score_missing_flag"] = working_df["Credit Score"].isna().astype(int)
    working_df["job_tenure_missing_flag"] = working_df["Years in current job"].isna().astype(int)
    working_df["delinquency_missing_flag"] = working_df["Months since last delinquent"].isna().astype(int)

    working_df["loan_amount_placeholder_flag"] = working_df["Current Loan Amount"].eq(99999999).astype(int)
    working_df["Home Ownership"] = working_df["Home Ownership"].replace({"Have Mortgage": "Home Mortgage"})
    working_df["Current Loan Amount"] = working_df["Current Loan Amount"].mask(
        working_df["loan_amount_placeholder_flag"].eq(1)
    )
    working_df["Credit Score"] = working_df["Credit Score"].where(
        working_df["Credit Score"] <= 850,
        working_df["Credit Score"] / 10,
    )
    working_df["job_tenure_years"] = working_df["Years in current job"].apply(job_tenure_to_years)

    working_df = working_df.drop(columns=["Id", "Years in current job"])
    return working_df


def transform_application(
    raw_df: pd.DataFrame, preprocessor_artifacts: Dict[str, object]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    working_df = apply_deterministic_cleaning(raw_df)

    for col, median_value in preprocessor_artifacts["imputation_medians"].items():
        working_df[col] = working_df[col].fillna(median_value)

    working_df["monthly_dti"] = working_df["Monthly Debt"] / (working_df["Annual Income"] / 12)
    working_df["credit_utilization_proxy"] = (
        working_df["Current Credit Balance"] / working_df["Maximum Open Credit"].replace(0, np.nan)
    )
    working_df["loan_to_income_ratio"] = working_df["Current Loan Amount"] / working_df["Annual Income"]
    working_df["credit_utilization_proxy"] = working_df["credit_utilization_proxy"].fillna(
        preprocessor_artifacts["credit_utilization_median"]
    )

    prepared_df = working_df[preprocessor_artifacts["feature_cols"]].copy()
    encoded_df = pd.get_dummies(
        prepared_df,
        columns=preprocessor_artifacts["categorical_cols"],
        drop_first=False,
        dtype=int,
    ).sort_index(axis=1)
    encoded_df = encoded_df.reindex(columns=preprocessor_artifacts["encoded_columns"], fill_value=0)
    return prepared_df, encoded_df


def initialize_session_state(field_metadata: Dict[str, Dict[str, object]]) -> None:
    if "selected_role" not in st.session_state:
        st.session_state["selected_role"] = ROLE_APPLICANT
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None
    if "last_scored_values" not in st.session_state:
        st.session_state["last_scored_values"] = None

    for column, meta in field_metadata.items():
        widget_key = FIELD_WIDGET_KEYS[column]
        if widget_key not in st.session_state:
            st.session_state[widget_key] = meta["default"]


def get_current_form_values() -> Dict[str, object]:
    values = {}
    for column in RAW_COLUMNS:
        values[column] = st.session_state[FIELD_WIDGET_KEYS[column]]
    return values


def set_form_values(values: Dict[str, object]) -> None:
    for column in RAW_COLUMNS:
        if column in values:
            st.session_state[FIELD_WIDGET_KEYS[column]] = values[column]


def build_mock_profiles() -> Dict[str, Dict[str, object]]:
    return {
        "Stable Salaried Borrower": {
            "Id": 900001,
            "Home Ownership": "Home Mortgage",
            "Annual Income": 120000.0,
            "Years in current job": "10+ years",
            "Tax Liens": 0.0,
            "Number of Open Accounts": 12.0,
            "Years of Credit History": 14.0,
            "Maximum Open Credit": 95000.0,
            "Number of Credit Problems": 0.0,
            "Months since last delinquent": 48.0,
            "Bankruptcies": 0.0,
            "Purpose": "debt consolidation",
            "Term": "Short Term",
            "Current Loan Amount": 12000.0,
            "Current Credit Balance": 15000.0,
            "Monthly Debt": 1800.0,
            "Credit Score": 742.0,
        },
        "Young Renter Review Case": {
            "Id": 900002,
            "Home Ownership": "Rent",
            "Annual Income": 52000.0,
            "Years in current job": "2 years",
            "Tax Liens": 0.0,
            "Number of Open Accounts": 8.0,
            "Years of Credit History": 5.0,
            "Maximum Open Credit": 24000.0,
            "Number of Credit Problems": 0.0,
            "Months since last delinquent": 18.0,
            "Bankruptcies": 0.0,
            "Purpose": "major purchase",
            "Term": "Long Term",
            "Current Loan Amount": 14000.0,
            "Current Credit Balance": 8200.0,
            "Monthly Debt": 1650.0,
            "Credit Score": 664.0,
        },
        "Business Loan Stretch": {
            "Id": 900003,
            "Home Ownership": "Own Home",
            "Annual Income": 68000.0,
            "Years in current job": "4 years",
            "Tax Liens": 0.0,
            "Number of Open Accounts": 10.0,
            "Years of Credit History": 9.0,
            "Maximum Open Credit": 30000.0,
            "Number of Credit Problems": 1.0,
            "Months since last delinquent": 12.0,
            "Bankruptcies": 0.0,
            "Purpose": "business loan",
            "Term": "Long Term",
            "Current Loan Amount": 25000.0,
            "Current Credit Balance": 19000.0,
            "Monthly Debt": 2850.0,
            "Credit Score": 621.0,
        },
        "Travel Loan Watchlist": {
            "Id": 900004,
            "Home Ownership": "Have Mortgage",
            "Annual Income": 76000.0,
            "Years in current job": "6 years",
            "Tax Liens": 0.0,
            "Number of Open Accounts": 11.0,
            "Years of Credit History": 11.0,
            "Maximum Open Credit": 42000.0,
            "Number of Credit Problems": 0.0,
            "Months since last delinquent": 22.0,
            "Bankruptcies": 0.0,
            "Purpose": "take a trip",
            "Term": "Long Term",
            "Current Loan Amount": 18000.0,
            "Current Credit Balance": 13000.0,
            "Monthly Debt": 2400.0,
            "Credit Score": 646.0,
        },
        "Escalation Case": {
            "Id": 900005,
            "Home Ownership": "Rent",
            "Annual Income": 43000.0,
            "Years in current job": "1 year",
            "Tax Liens": 1.0,
            "Number of Open Accounts": 7.0,
            "Years of Credit History": 4.0,
            "Maximum Open Credit": 16000.0,
            "Number of Credit Problems": 2.0,
            "Months since last delinquent": 6.0,
            "Bankruptcies": 1.0,
            "Purpose": "other",
            "Term": "Long Term",
            "Current Loan Amount": 22000.0,
            "Current Credit Balance": 14500.0,
            "Monthly Debt": 2550.0,
            "Credit Score": 592.0,
        },
    }


def validate_form(values: Dict[str, object]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    field_errors: Dict[str, List[str]] = {}
    field_warnings: Dict[str, List[str]] = {}
    summary_warnings: List[str] = []

    def add_error(field: str, message: str) -> None:
        field_errors.setdefault(field, []).append(message)

    def add_warning(field: str, message: str) -> None:
        field_warnings.setdefault(field, []).append(message)

    if values["Credit Score"] < 0 or values["Credit Score"] > 850:
        add_error("Credit Score", "Please check the credit score. Valid range is 0 to 850.")
    if values["Annual Income"] <= 0:
        add_error("Annual Income", "Annual Income must be greater than 0.")
    if values["Current Loan Amount"] <= 0:
        add_error("Current Loan Amount", "Current Loan Amount must be greater than 0.")
    if values["Current Credit Balance"] < 0:
        add_error("Current Credit Balance", "Current Credit Balance cannot be negative.")
    if values["Monthly Debt"] < 0:
        add_error("Monthly Debt", "Monthly Debt cannot be negative.")
    if values["Maximum Open Credit"] <= 0:
        add_error("Maximum Open Credit", "Maximum Open Credit must be greater than 0.")
    if values["Number of Open Accounts"] < 0:
        add_error("Number of Open Accounts", "Number of Open Accounts cannot be negative.")
    if values["Years of Credit History"] < 0:
        add_error("Years of Credit History", "Years of Credit History cannot be negative.")
    if values["Tax Liens"] < 0:
        add_error("Tax Liens", "Tax Liens cannot be negative.")
    if values["Number of Credit Problems"] < 0:
        add_error("Number of Credit Problems", "Number of Credit Problems cannot be negative.")
    if values["Bankruptcies"] < 0:
        add_error("Bankruptcies", "Bankruptcies cannot be negative.")
    if values["Months since last delinquent"] < 0:
        add_error("Months since last delinquent", "Months since last delinquent cannot be negative.")

    annual_income = values["Annual Income"]
    monthly_debt = values["Monthly Debt"]
    current_credit_balance = values["Current Credit Balance"]
    max_open_credit = values["Maximum Open Credit"]

    if annual_income > 0:
        estimated_dti = monthly_debt / (annual_income / 12)
        if estimated_dti >= 0.65:
            add_warning(
                "Monthly Debt",
                "Monthly debt implies very high debt-to-income pressure. Expect tighter review.",
            )
        elif estimated_dti >= 0.50:
            add_warning(
                "Monthly Debt",
                "Monthly debt implies elevated affordability pressure and may trigger manual review.",
            )

        if monthly_debt * 12 > annual_income:
            summary_warnings.append(
                "Annualized monthly debt is higher than annual income. Please re-check income or debt values."
            )

    if max_open_credit > 0 and current_credit_balance > max_open_credit:
        add_warning(
            "Current Credit Balance",
            "Current Credit Balance is above Maximum Open Credit. Please double-check these values.",
        )

    return field_errors, field_warnings, summary_warnings


def build_raw_input_df(values: Dict[str, object]) -> pd.DataFrame:
    row = {}
    for column in RAW_COLUMNS:
        value = values[column]
        if column in CATEGORICAL_COLUMNS:
            row[column] = value
        elif column == "Id":
            row[column] = int(value)
        else:
            row[column] = float(value)
    return pd.DataFrame([row])


def build_employee_decision(probability: float, prepared_row: pd.Series) -> Dict[str, object]:
    return build_credit_decision(probability, prepared_row)


def score_application(
    bundle: Dict[str, object], raw_input_df: pd.DataFrame
) -> Dict[str, object]:
    prepared_df, encoded_df = transform_application(raw_input_df, bundle["preprocessor_artifacts"])
    probability = float(bundle["model"].predict_proba(encoded_df[bundle["feature_columns"]])[:, 1][0])
    model_default_label = int(probability >= float(bundle["threshold"]))
    prepared_row = prepared_df.iloc[0]
    employee_decision = build_employee_decision(probability, prepared_row)

    return {
        "raw_input_df": raw_input_df,
        "prepared_df": prepared_df,
        "encoded_df": encoded_df,
        "predicted_default_probability": probability,
        "model_default_label": model_default_label,
        "employee_decision": employee_decision,
    }


def render_field_messages(field: str, field_errors: Dict[str, List[str]], field_warnings: Dict[str, List[str]]) -> None:
    for error in field_errors.get(field, []):
        st.error(error)
    for warning in field_warnings.get(field, []):
        st.warning(warning)


def render_role_selector() -> None:
    st.markdown("## Choose Your View")
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Applicant",
            use_container_width=True,
            type="primary" if st.session_state["selected_role"] == ROLE_APPLICANT else "secondary",
        ):
            st.session_state["selected_role"] = ROLE_APPLICANT

    with col2:
        if st.button(
            "Bank Employee",
            use_container_width=True,
            type="primary" if st.session_state["selected_role"] == ROLE_EMPLOYEE else "secondary",
        ):
            st.session_state["selected_role"] = ROLE_EMPLOYEE


def render_mock_profile_sidebar(mock_profiles: Dict[str, Dict[str, object]]) -> None:
    with st.sidebar:
        st.markdown("## Mock Profiles")
        st.caption("Pick a profile to prefill the form, then edit any field you want.")

        for profile_name, profile_values in mock_profiles.items():
            if st.button(profile_name, key=f"profile_{profile_name}", use_container_width=True):
                set_form_values(profile_values)
                st.session_state["latest_result"] = None
                st.session_state["last_scored_values"] = None

        st.markdown("---")
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state["latest_result"] = None
            st.session_state["last_scored_values"] = None
            raw_df = load_raw_application_data()
            metadata = build_field_metadata(raw_df)
            for column, meta in metadata.items():
                st.session_state[FIELD_WIDGET_KEYS[column]] = meta["default"]


def render_numeric_input(label: str, widget_key: str, step: float) -> None:
    format_string = "%d" if step == 1 else "%.2f"
    st.number_input(label, key=widget_key, step=step, format=format_string)


def render_application_form(
    field_metadata: Dict[str, Dict[str, object]],
    field_errors: Dict[str, List[str]],
    field_warnings: Dict[str, List[str]],
) -> None:
    st.markdown("## New Application Entry")
    st.caption("Only raw columns from the original application file are entered here. Engineered fields are calculated after scoring.")

    left_col, right_col = st.columns(2)
    layout = [
        ("Id", left_col),
        ("Home Ownership", right_col),
        ("Annual Income", left_col),
        ("Years in current job", right_col),
        ("Tax Liens", left_col),
        ("Number of Open Accounts", right_col),
        ("Years of Credit History", left_col),
        ("Maximum Open Credit", right_col),
        ("Number of Credit Problems", left_col),
        ("Months since last delinquent", right_col),
        ("Bankruptcies", left_col),
        ("Purpose", right_col),
        ("Term", left_col),
        ("Current Loan Amount", right_col),
        ("Current Credit Balance", left_col),
        ("Monthly Debt", right_col),
        ("Credit Score", left_col),
    ]

    for column, container in layout:
        meta = field_metadata[column]
        with container:
            if meta["type"] == "categorical":
                options = meta["options"]
                current_value = st.session_state[FIELD_WIDGET_KEYS[column]]
                if current_value not in options:
                    options = [current_value] + options
                st.selectbox(column, options=options, key=FIELD_WIDGET_KEYS[column])
            else:
                render_numeric_input(column, FIELD_WIDGET_KEYS[column], meta["step"])

            render_field_messages(column, field_errors, field_warnings)


def render_applicant_results(result: Dict[str, object]) -> None:
    decision_band = result["employee_decision"]["decision_band"]
    st.markdown("## Current Outcome")
    st.metric("Current Outcome", decision_band)

    if decision_band == RECOMMEND_APPROVE:
        message = "Your application currently looks suitable for approval, subject to the bank's normal checks."
    elif decision_band == MANUAL_REVIEW:
        message = "Your application would likely need additional review before a final decision is made."
    else:
        message = "Your application currently appears higher risk and may be escalated or declined."

    st.info(message)
    st.caption("This is a simplified decision-support outcome for the assignment, not a final lending offer.")


def render_employee_results(result: Dict[str, object], bundle: Dict[str, object]) -> None:
    decision = result["employee_decision"]
    probability = float(result["predicted_default_probability"])

    st.markdown("## Decision Summary")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Predicted Default Probability", f"{probability:.1%}")
    with metric_col2:
        st.metric("EAD", f"${float(decision['ead']):,.0f}")
    with metric_col3:
        st.metric("LGD Assumption", f"{float(decision['lgd_assumption']):.0%}")
    with metric_col4:
        st.metric("Expected Loss", f"${float(decision['expected_loss']):,.0f}")

    style = RECOMMENDATION_STYLES[decision["decision_band"]]
    st.markdown(
        f"""
        <div style="
            margin-top: 1rem;
            padding: 1rem 1.25rem;
            border-left: 8px solid {style['border']};
            background: {style['background']};
            border-radius: 0.75rem;
        ">
            <div style="
                font-size: 0.9rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                text-transform: uppercase;
                color: {style['text']};
                margin-bottom: 0.35rem;
            ">Recommendation Band</div>
            <div style="
                font-size: 1.8rem;
                font-weight: 800;
                line-height: 1.2;
                color: {style['text']};
            ">{decision['decision_band']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Model threshold: {float(bundle['threshold']):.4f} | "
        f"Model default label: {result['model_default_label']} | "
        f"Model: {bundle['model_name']}"
    )

    st.markdown("## Why This Result")
    if decision["policy_override_triggered"]:
        st.warning(f"Policy override triggered: {decision['policy_override_reason']}")
    else:
        st.info("No hard policy override was triggered beyond the base risk band.")
    for reason in decision["reasons"]:
        st.info(reason)
    for note in decision["caution_notes"]:
        st.info(note)

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("## Original Application")
        raw_display_df = result["raw_input_df"][RAW_COLUMNS].T.reset_index()
        raw_display_df.columns = ["Field", "Value"]
        raw_display_df["Value"] = raw_display_df["Value"].astype(str)
        st.dataframe(raw_display_df, use_container_width=True, hide_index=True)

    with detail_col2:
        st.markdown("## Engineered Analysis")
        engineered_df = pd.DataFrame(
            {
                "Metric": [
                    "monthly_dti",
                    "loan_to_income_ratio",
                    "credit_utilization_proxy",
                    "ead",
                    "lgd_assumption",
                    "expected_loss",
                ],
                "Value": [
                    decision["monthly_dti"],
                    decision["loan_to_income_ratio"],
                    decision["credit_utilization_proxy"],
                    f"${float(decision['ead']):,.0f}",
                    f"{float(decision['lgd_assumption']):.0%}",
                    f"${float(decision['expected_loss']):,.0f}",
                ],
            }
        )
        for metric_name in ["monthly_dti", "loan_to_income_ratio", "credit_utilization_proxy"]:
            engineered_df.loc[engineered_df["Metric"] == metric_name, "Value"] = engineered_df.loc[
                engineered_df["Metric"] == metric_name, "Value"
            ].map(lambda x: f"{float(x):.3f}")
        st.dataframe(engineered_df, use_container_width=True, hide_index=True)

    fairness_summary = load_fairness_summary()
    if fairness_summary:
        st.markdown("## Fairness Summary")
        st.caption(
            "This summary comes from held-out historical test-set evaluation and is meant for governance review, "
            "not for deciding the current applicant on its own."
        )
        st.markdown(f"**Evaluated group:** {fairness_summary['evaluated_group']}")

        gap_col1, gap_col2, gap_col3 = st.columns(3)
        with gap_col1:
            st.metric("Approval-rate gap", f"{float(fairness_summary['approval_rate_gap']):.1%}")
        with gap_col2:
            st.metric("False-positive-rate gap", f"{float(fairness_summary['false_positive_rate_gap']):.1%}")
        with gap_col3:
            st.metric("False-negative-rate gap", f"{float(fairness_summary['false_negative_rate_gap']):.1%}")

        fairness_table = pd.DataFrame(fairness_summary.get("group_summary", []))
        if not fairness_table.empty:
            fairness_table = fairness_table.rename(
                columns={
                    "Home Ownership": "Group",
                    "sample_count": "Sample Count",
                    "approval_rate": "Approval Rate",
                    "false_positive_rate": "False Positive Rate",
                    "false_negative_rate": "False Negative Rate",
                }
            )
            fairness_table["Approval Rate"] = fairness_table["Approval Rate"].map(lambda x: f"{float(x):.1%}")
            fairness_table["False Positive Rate"] = fairness_table["False Positive Rate"].map(
                lambda x: f"{float(x):.1%}"
            )
            fairness_table["False Negative Rate"] = fairness_table["False Negative Rate"].map(
                lambda x: f"{float(x):.1%}"
            )
            st.dataframe(fairness_table, use_container_width=True, hide_index=True)

        st.info(fairness_summary["mitigation_note"])
        st.caption(fairness_summary["disclaimer"])


def main() -> None:
    st.title("Credit Default Decision Support")
    st.markdown(
        "Enter a new credit application using the original application fields and review the decision-support outcome. "
        "Applicant view keeps the response simple, while bank-employee view shows the internal risk and economics details."
    )

    bundle = load_model_bundle()
    raw_applications_df = load_raw_application_data()
    field_metadata = build_field_metadata(raw_applications_df)
    initialize_session_state(field_metadata)

    mock_profiles = build_mock_profiles()
    render_mock_profile_sidebar(mock_profiles)
    render_role_selector()

    current_values = get_current_form_values()
    field_errors, field_warnings, summary_warnings = validate_form(current_values)

    if field_errors:
        error_count = sum(len(messages) for messages in field_errors.values())
        st.error(f"Please fix {error_count} field issue(s) before running prediction.")
    for warning in summary_warnings:
        st.warning(warning)

    render_application_form(field_metadata, field_errors, field_warnings)

    if st.button("Evaluate Application", type="primary", use_container_width=True):
        if field_errors:
            st.session_state["latest_result"] = None
            st.session_state["last_scored_values"] = None
        else:
            latest_values = get_current_form_values()
            raw_input_df = build_raw_input_df(latest_values)
            st.session_state["latest_result"] = score_application(bundle, raw_input_df)
            st.session_state["last_scored_values"] = latest_values.copy()

    if st.session_state["latest_result"] is not None:
        if st.session_state["last_scored_values"] != get_current_form_values():
            st.info("Inputs changed after the last prediction. Run the evaluation again to refresh the result.")
        st.markdown("---")
        if st.session_state["selected_role"] == ROLE_APPLICANT:
            render_applicant_results(st.session_state["latest_result"])
        else:
            render_employee_results(st.session_state["latest_result"], bundle)


if __name__ == "__main__":
    main()
