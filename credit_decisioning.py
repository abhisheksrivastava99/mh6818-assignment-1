from __future__ import annotations

from typing import Dict, List

import pandas as pd


LOW_RISK_CUTOFF = 0.10
HIGH_RISK_CUTOFF = 0.25
HIGH_REVIEW_DTI = 0.50
ESCALATE_DTI = 0.65
HIGH_LOAN_TO_INCOME = 0.50
HIGH_UTILIZATION = 0.80
LGD_ASSUMPTION = 0.45

RECOMMEND_APPROVE = "Recommend Approve"
MANUAL_REVIEW = "Manual Review"
RECOMMEND_REJECT = "Recommend Reject / Escalate"

DECISION_ORDER = {
    RECOMMEND_APPROVE: 0,
    MANUAL_REVIEW: 1,
    RECOMMEND_REJECT: 2,
}

DECISION_BAND_ORDER = [RECOMMEND_APPROVE, MANUAL_REVIEW, RECOMMEND_REJECT]
DECISION_EXPORT_COLUMNS = [
    "decision_band",
    "policy_override_triggered",
    "policy_override_reason",
    "monthly_dti",
    "loan_to_income_ratio",
    "credit_utilization_proxy",
    "ead",
    "lgd_assumption",
    "expected_loss",
]


def upgrade_decision(current: str, candidate: str) -> str:
    return candidate if DECISION_ORDER[candidate] > DECISION_ORDER[current] else current


def base_decision_band(probability: float) -> str:
    if probability < LOW_RISK_CUTOFF:
        return RECOMMEND_APPROVE
    if probability < HIGH_RISK_CUTOFF:
        return MANUAL_REVIEW
    return RECOMMEND_REJECT


def build_credit_decision(
    probability: float,
    prepared_row: pd.Series,
    lgd_assumption: float = LGD_ASSUMPTION,
) -> Dict[str, object]:
    decision_band = base_decision_band(probability)
    override_reasons: List[str] = []
    caution_notes: List[str] = []

    monthly_dti = float(prepared_row["monthly_dti"])
    loan_to_income_ratio = float(prepared_row["loan_to_income_ratio"])
    credit_utilization_proxy = float(prepared_row["credit_utilization_proxy"])
    ead = float(prepared_row["Current Loan Amount"])
    expected_loss = float(probability * lgd_assumption * ead)

    if monthly_dti >= ESCALATE_DTI:
        decision_band = RECOMMEND_REJECT
        override_reasons.append("Debt-to-income ratio is extremely high and triggers escalation.")
    elif monthly_dti >= HIGH_REVIEW_DTI:
        decision_band = upgrade_decision(decision_band, MANUAL_REVIEW)
        override_reasons.append("Debt-to-income ratio is elevated and triggers manual review.")
    elif monthly_dti >= 0.35:
        caution_notes.append("Debt-to-income ratio is above the healthier affordability range.")
    else:
        caution_notes.append("Debt-to-income ratio is within a healthier affordability range.")

    if loan_to_income_ratio >= HIGH_LOAN_TO_INCOME:
        decision_band = upgrade_decision(decision_band, MANUAL_REVIEW)
        override_reasons.append("Loan amount is high relative to reported income.")
    elif loan_to_income_ratio >= 0.30:
        caution_notes.append("Loan amount is moderately stretched relative to income.")

    if credit_utilization_proxy >= HIGH_UTILIZATION:
        decision_band = upgrade_decision(decision_band, MANUAL_REVIEW)
        override_reasons.append("Credit utilization is high and suggests leverage stress.")
    elif credit_utilization_proxy >= 0.50:
        caution_notes.append("Credit utilization is moderate to high.")

    reasons = [
        f"Predicted default probability of {probability:.1%} maps to {base_decision_band(probability)} under the base policy."
    ]
    if override_reasons:
        reasons.extend(override_reasons)
    else:
        reasons.append("No hard policy override was triggered beyond the base risk band.")

    return {
        "decision_band": decision_band,
        "policy_override_triggered": bool(override_reasons),
        "policy_override_reason": "; ".join(override_reasons) if override_reasons else "",
        "monthly_dti": monthly_dti,
        "loan_to_income_ratio": loan_to_income_ratio,
        "credit_utilization_proxy": credit_utilization_proxy,
        "ead": ead,
        "lgd_assumption": float(lgd_assumption),
        "expected_loss": expected_loss,
        "reasons": reasons,
        "caution_notes": caution_notes,
    }


def build_decision_frame(
    prepared_df: pd.DataFrame,
    predicted_default_probability: pd.Series | List[float],
    lgd_assumption: float = LGD_ASSUMPTION,
) -> pd.DataFrame:
    probability_series = pd.Series(predicted_default_probability, index=prepared_df.index, dtype=float)
    decision_rows = [
        build_credit_decision(float(probability_series.loc[idx]), prepared_df.loc[idx], lgd_assumption=lgd_assumption)
        for idx in prepared_df.index
    ]
    return pd.DataFrame(decision_rows, index=prepared_df.index)


def build_decision_export_frame(
    prepared_df: pd.DataFrame,
    predicted_default_probability: pd.Series | List[float],
    lgd_assumption: float = LGD_ASSUMPTION,
) -> pd.DataFrame:
    decision_df = build_decision_frame(prepared_df, predicted_default_probability, lgd_assumption=lgd_assumption)
    return decision_df[DECISION_EXPORT_COLUMNS].copy()


def summarize_decisions(scored_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    total_count = int(scored_df.shape[0])

    for decision_band in DECISION_BAND_ORDER:
        band_df = scored_df[scored_df["decision_band"] == decision_band]
        summary_rows.append(
            {
                "decision_band": decision_band,
                "application_count": int(band_df.shape[0]),
                "share_of_applications": float(band_df.shape[0] / total_count) if total_count else 0.0,
                "avg_predicted_default_probability": float(band_df["predicted_default_probability"].mean())
                if not band_df.empty
                else 0.0,
                "total_ead": float(band_df["ead"].sum()) if not band_df.empty else 0.0,
                "total_expected_loss": float(band_df["expected_loss"].sum()) if not band_df.empty else 0.0,
            }
        )

    summary_rows.append(
        {
            "decision_band": "Overall",
            "application_count": total_count,
            "share_of_applications": 1.0 if total_count else 0.0,
            "avg_predicted_default_probability": float(scored_df["predicted_default_probability"].mean())
            if total_count
            else 0.0,
            "total_ead": float(scored_df["ead"].sum()) if total_count else 0.0,
            "total_expected_loss": float(scored_df["expected_loss"].sum()) if total_count else 0.0,
        }
    )
    return pd.DataFrame(summary_rows)
