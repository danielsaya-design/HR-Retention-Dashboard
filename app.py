"""
HR Retention Analytics — Streamlit dashboard for business audiences.
Focus: attrition patterns, retention risk drivers, and actionable recommendations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Paths (same folder as this script)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
HR_PATH = BASE_DIR / "hr_data.csv"


# -----------------------------------------------------------------------------
# Data loading & cleaning
# -----------------------------------------------------------------------------
@st.cache_data
def load_hr_data(path: str) -> pd.DataFrame:
    """Load HR CSV; coerce types and handle missing values safely."""
    df = pd.read_csv(Path(path))
    numeric_cols = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "left",
        "promotion_last_5years",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Department" in df.columns:
        df["Department"] = df["Department"].astype(str).replace("nan", np.nan)
    if "salary" in df.columns:
        df["salary"] = df["salary"].astype(str).replace("nan", np.nan)
    for c in numeric_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    if "Department" in df.columns:
        df["Department"] = df["Department"].fillna(
            df["Department"].mode().iloc[0] if len(df["Department"].mode()) else "Unknown"
        )
    if "salary" in df.columns:
        df["salary"] = df["salary"].fillna(
            df["salary"].mode().iloc[0] if len(df["salary"].mode()) else "Unknown"
        )
    # Synthetic employee ID for employee-level tables.
    if "employee_id" not in df.columns:
        df.insert(0, "employee_id", [f"EMP-{i:05d}" for i in range(1, len(df) + 1)])
    return df


def outcome_label(s: pd.Series) -> pd.Series:
    """Map attrition flag to business-friendly labels."""
    return s.astype(int).map({0: "Stayed", 1: "Left"})


def attrition_rate_pct(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate attrition rate (%) and headcount by group."""
    g = df.groupby(group_cols, observed=False).agg(
        headcount=("left", "count"),
        attrition_rate=("left", "mean"),
    )
    g = g.reset_index()
    g["attrition_rate_pct"] = 100.0 * g["attrition_rate"]
    return g


def risk_level_from_score(score: float) -> str:
    """Map numeric Risk_Score to business buckets (sidebar filter labels)."""
    if score < 20:
        return "Low"
    if score < 40:
        return "Medium"
    if score <= 70:
        return "High"
    return "Critical"


# Labels for drilldown “main risk reasons” (must match compute_employee_risk_scores columns)
RISK_REASON_DEFINITIONS: list[tuple[str, str]] = [
    ("reason_boredom", "Boredom gap (3+ yrs, ≤2 projects)"),
    ("reason_burnout", "Burnout risk (4–5 yrs, 6+ projects)"),
    ("reason_high_hours", "Heavy workload (>260 monthly hours)"),
    ("reason_star_poaching", "Star poaching profile (high eval, no promo, 4+ yrs)"),
    ("reason_toxic_dept", "HR / Accounting department factor"),
    ("reason_low_salary", "Low salary band"),
    ("reason_low_satisfaction", "Very low satisfaction (<0.2)"),
]


def compute_employee_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based Risk_Score (0–190+ before exemptions). Weights:
    - Boredom gap: tenure ≥3 yrs and ≤2 projects (+35)
    - Burnout: tenure 4–5 yrs and ≥6 projects (+40)
    - Heavy workload: average_montly_hours > 260 (+20)
    - Star poaching risk: high evaluation, no recent promo, tenure ≥4 yrs (+30)
    - Toxic dept: HR or Accounting (+10)
    - Low salary band (+15)
    - Very low satisfaction <0.2 (+40)

    Score is forced to 0 if tenure ≥7 years OR promoted in last 5 years.

    Also adds boolean reason_* columns (True only when the rule applies and the employee is not exempt).
    """
    out = df.copy()
    tenure = out["time_spend_company"].astype(float)
    proj = out["number_project"].astype(float)
    hours = out["average_montly_hours"].astype(float)
    eval_ = out["last_evaluation"].astype(float)
    promo = out["promotion_last_5years"].astype(int)
    dept = out["Department"].astype(str).str.strip().str.lower()
    sal = out["salary"].astype(str).str.strip().str.lower()
    sat = out["satisfaction_level"].astype(float)

    # Exempt: long-tenure veterans or recently promoted — treat as stable / low flight risk
    exempt = (tenure >= 7) | (promo == 1)
    active = ~exempt

    boredom = (tenure >= 3) & (proj <= 2)
    burnout = (tenure >= 4) & (tenure <= 5) & (proj >= 6)
    high_hours = hours > 260
    # "High" evaluation: strong performer at risk of external offers if under-leveled
    star_poaching = (eval_ >= 0.85) & (promo == 0) & (tenure >= 4)
    toxic_dept = dept.isin(["hr", "accounting"])
    low_salary = sal == "low"
    low_satisfaction = sat < 0.2

    raw = (
        boredom.astype(int) * 35
        + burnout.astype(int) * 40
        + high_hours.astype(int) * 20
        + star_poaching.astype(int) * 30
        + toxic_dept.astype(int) * 10
        + low_salary.astype(int) * 15
        + low_satisfaction.astype(int) * 40
    )
    out["Risk_Score"] = np.where(exempt, 0.0, raw.astype(float))
    out["Risk_Level"] = out["Risk_Score"].apply(risk_level_from_score)

    # Reasons apply to score logic only when not exempt (used for department drilldown)
    out["reason_boredom"] = boredom & active
    out["reason_burnout"] = burnout & active
    out["reason_high_hours"] = high_hours & active
    out["reason_star_poaching"] = star_poaching & active
    out["reason_toxic_dept"] = toxic_dept & active
    out["reason_low_salary"] = low_salary & active
    out["reason_low_satisfaction"] = low_satisfaction & active

    return out


def summarize_department_risk_reasons(dept_df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each risk reason appears among employees in a department."""
    rows = []
    for col, label in RISK_REASON_DEFINITIONS:
        if col in dept_df.columns:
            rows.append({"Reason": label, "Employees": int(dept_df[col].sum())})
    out = pd.DataFrame(rows)
    out = out[out["Employees"] > 0].sort_values("Employees", ascending=False)
    return out


def matching_reason_labels(row: pd.Series) -> str:
    """Comma-separated list of human-readable risk reasons that apply to one employee."""
    parts = []
    for col, label in RISK_REASON_DEFINITIONS:
        if col in row.index and bool(row[col]):
            parts.append(label)
    return "; ".join(parts) if parts else "—"


def slice_employees_by_risk(
    dept_df: pd.DataFrame, risk_level: str, reason_column: str | None
) -> pd.DataFrame:
    """
    Filter to employees in this department with the given Risk_Level.
    If reason_column is set, require that reason flag to be True.
    If reason_column is None, include everyone in that risk band.
    """
    m = dept_df["Risk_Level"] == risk_level
    if reason_column:
        m = m & dept_df[reason_column].astype(bool)
    return dept_df.loc[m].copy()


def employee_risk_drilldown_table(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Columns suitable for displaying in the app (no raw boolean matrix)."""
    if slice_df.empty:
        return slice_df
    out = slice_df.copy()
    # Reference to source row in hr_data.csv (line 1 = header; first employee = line 2)
    out.insert(0, "CSV_line", out.index.astype(int) + 2)
    out["Matching_reasons"] = out.apply(matching_reason_labels, axis=1)
    want = [
        "CSV_line",
        "salary",
        "satisfaction_level",
        "time_spend_company",
        "number_project",
        "average_montly_hours",
        "last_evaluation",
        "Risk_Score",
        "Risk_Level",
        "Matching_reasons",
    ]
    cols = [c for c in want if c in out.columns]
    disp = out[cols].copy()
    rename = {
        "CSV_line": "CSV line",
        "salary": "Salary",
        "satisfaction_level": "Satisfaction",
        "time_spend_company": "Tenure (yrs)",
        "number_project": "Projects",
        "average_montly_hours": "Monthly hours",
        "last_evaluation": "Last evaluation",
        "Risk_Score": "Risk score",
        "Risk_Level": "Risk level",
        "Matching_reasons": "Matching risk reasons",
    }
    disp = disp.rename(columns={k: v for k, v in rename.items() if k in disp.columns})
    for num_col in ("Satisfaction", "Last evaluation"):
        if num_col in disp.columns:
            disp[num_col] = disp[num_col].round(2)
    if "Risk score" in disp.columns:
        disp["Risk score"] = disp["Risk score"].round(1)
    return disp


def risk_level_sidebar_multiselect() -> list[str]:
    """Which risk bands to show on department bar charts (current employees only)."""
    st.sidebar.markdown("### Predictive Risk Monitor")
    opts = ["Low", "Medium", "High", "Critical"]
    return st.sidebar.multiselect(
        "Risk levels to show",
        options=opts,
        default=opts,
        help="Choose which risk categories appear on each department’s bar chart.",
    )


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters; optional advanced filters collapsed to reduce clutter."""
    d = df.copy()
    st.sidebar.markdown("### Filter the data")
    depts = sorted(d["Department"].dropna().unique().tolist())
    salaries = sorted(d["salary"].dropna().unique().tolist())
    sel_dept = st.sidebar.multiselect(
        "Department",
        options=depts,
        default=depts,
        help="Narrow to one or more departments.",
    )
    sel_sal = st.sidebar.multiselect(
        "Salary level",
        options=salaries,
        default=salaries,
        help="Low, medium, or high pay bands.",
    )
    with st.sidebar.expander("More filters", expanded=False):
        prom_opts = sorted(d["promotion_last_5years"].dropna().unique().tolist())
        wa_opts = sorted(d["Work_accident"].dropna().unique().tolist())
        sel_prom = st.multiselect(
            "Promoted in last 5 years",
            options=prom_opts,
            default=prom_opts,
        )
        sel_wa = st.multiselect("Had a work accident", options=wa_opts, default=wa_opts)
    if sel_dept:
        d = d[d["Department"].isin(sel_dept)]
    if sel_sal:
        d = d[d["salary"].isin(sel_sal)]
    if sel_prom:
        d = d[d["promotion_last_5years"].isin(sel_prom)]
    if sel_wa:
        d = d[d["Work_accident"].isin(sel_wa)]
    return d


def build_recommendations(df: pd.DataFrame) -> list[str]:
    """
    Generate 3–5 plain-language recommendations from patterns in the data.
    Uses simple comparisons (rates, averages) — no statistical jargon.
    """
    if len(df) < 30:
        return [
            "Use a broader filter (or the full dataset) so recommendations reflect enough employees."
        ]
    recs: list[str] = []
    overall = 100.0 * df["left"].mean()

    # Salary bands: where is attrition worst?
    sal_grp = attrition_rate_pct(df, ["salary"]).sort_values("attrition_rate_pct", ascending=False)
    if len(sal_grp) >= 2:
        top = sal_grp.iloc[0]
        bottom = sal_grp.iloc[-1]
        if top["attrition_rate_pct"] > bottom["attrition_rate_pct"] + 5:
            recs.append(
                f"**Close the retention gap across pay bands.** "
                f"Attrition is about **{top['attrition_rate_pct']:.0f}%** in the **{top['salary']}** group "
                f"vs **{bottom['attrition_rate_pct']:.0f}%** in the **{bottom['salary']}** group — "
                f"review compensation, recognition, and growth visibility for lower-paid roles."
            )

    # Satisfaction gap
    stayed = df[df["left"] == 0]["satisfaction_level"].mean()
    left = df[df["left"] == 1]["satisfaction_level"].mean()
    if not (np.isnan(stayed) or np.isnan(left)) and (stayed - left) >= 0.08:
        recs.append(
            f"**Act on engagement before people disengage.** "
            f"Employees who left averaged **{left:.2f}** on satisfaction vs **{stayed:.2f}** for those who stayed — "
            f"use stay interviews and pulse surveys in high-churn teams."
        )

    # Workload (hours)
    h_stay = df[df["left"] == 0]["average_montly_hours"].mean()
    h_left = df[df["left"] == 1]["average_montly_hours"].mean()
    if h_left > h_stay + 15:
        recs.append(
            f"**Reduce burnout risk from long hours.** "
            f"People who left worked about **{h_left:.0f}** hours/month on average vs **{h_stay:.0f}** for those who stayed — "
            f"prioritize workload balancing and realistic staffing targets."
        )

    # Tenure / projects (overworked + many projects)
    t_stay = df[df["left"] == 0]["time_spend_company"].mean()
    t_left = df[df["left"] == 1]["time_spend_company"].mean()
    p_stay = df[df["left"] == 0]["number_project"].mean()
    p_left = df[df["left"] == 1]["number_project"].mean()
    if p_left > p_stay + 0.4:
        recs.append(
            f"**Watch project load.** "
            f"Employees who left carried about **{p_left:.1f}** projects on average vs **{p_stay:.1f}** for those who stayed — "
            f"clarify priorities and say no to non-essential work."
        )
    elif t_left < t_stay - 0.5 and overall > 15:
        recs.append(
            f"**Support newer employees.** "
            f"Average tenure is lower among people who left (**{t_left:.1f}** years) than those who stayed (**{t_stay:.1f}** years) — "
            f"strengthen onboarding and first-year check-ins."
        )

    # Department hotspot
    dept_grp = attrition_rate_pct(df, ["Department"]).sort_values(
        "attrition_rate_pct", ascending=False
    )
    if len(dept_grp) >= 2:
        worst_d = dept_grp.iloc[0]
        comp_avg = overall
        if worst_d["attrition_rate_pct"] > comp_avg + 8 and worst_d["headcount"] >= 20:
            recs.append(
                f"**Partner with department leadership.** "
                f"**{worst_d['Department']}** shows about **{worst_d['attrition_rate_pct']:.0f}%** attrition "
                f"in this view (company-wide in view: **{comp_avg:.0f}%**) — schedule a focused retention review."
            )

    # Promotions
    pr = df.groupby("promotion_last_5years")["left"].mean()
    if 0 in pr.index and 1 in pr.index and pr[0] > pr[1] + 0.05:
        recs.append(
            "**Make career growth visible.** "
            "Employees with no recent promotion show higher attrition in this dataset — "
            "map internal mobility and communicate promotion criteria clearly."
        )

    # De-duplicate and cap length
    seen = set()
    out = []
    for r in recs:
        key = r[:40]
        if key not in seen:
            seen.add(key)
            out.append(r)
    if not out:
        out.append(
            "**Keep monitoring.** "
            "Patterns are relatively flat in the current filter — continue tracking satisfaction, workload, and pay equity over time."
        )
    return out[:5]


# -----------------------------------------------------------------------------
# Plot styling (consistent, presentation-friendly)
# -----------------------------------------------------------------------------
CHART_HEIGHT = 380
ATTRITION_LABEL = "Attrition rate (%)"

# Predictive Risk Monitor — consistent category order and colors
RISK_LEVEL_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_LEVEL_COLORS = {
    "Low": "#2E7D32",
    "Medium": "#F9A825",
    "High": "#EF6C00",
    "Critical": "#C62828",
}


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Workforce Retention & Attrition",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Workforce Retention & Attrition")
    st.markdown(
        "**How this dashboard helps:** See where turnover concentrates, how stayers and leavers differ, "
        "and where to focus retention efforts — using your filtered employee population."
    )

    try:
        hr_raw = load_hr_data(HR_PATH)
    except Exception as e:
        st.error(f"Could not load hr_data.csv: {e}")
        st.stop()

    st.sidebar.header("Scope")
    filtered = apply_sidebar_filters(hr_raw)
    risk_level_selection = risk_level_sidebar_multiselect()

    tab_exec, tab_risk, tab_dept, tab_pred, tab_cap_stab, tab_rec = st.tabs(
        [
            "Executive Overview",
            "Risk Drivers",
            "Department Drilldown",
            "Predictive Risk Monitor",
            "Capacity & Stability",
            "Recommendations",
        ]
    )

    # =========================================================================
    # Executive Overview
    # =========================================================================
    with tab_exec:
        st.subheader("Executive overview")
        st.caption(
            "High-level picture of turnover and risk in the population you selected in the sidebar."
        )
        n = len(filtered)
        if n == 0:
            st.warning("No employees match the current filters. Adjust filters in the sidebar.")
        else:
            attr_pct = 100.0 * filtered["left"].mean()
            avg_sat = filtered["satisfaction_level"].mean()
            avg_h = filtered["average_montly_hours"].mean()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Employees in view", f"{n:,}")
            with c2:
                st.metric("Attrition rate", f"{attr_pct:.1f}%", help="Share of employees who left.")
            with c3:
                st.metric("Avg satisfaction (0–1)", f"{avg_sat:.2f}")
            with c4:
                st.metric("Avg monthly hours", f"{avg_h:.0f}")

        st.divider()

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("#### Where is turnover highest by department?")
            if len(filtered):
                dept_df = attrition_rate_pct(filtered, ["Department"]).sort_values(
                    "attrition_rate_pct", ascending=False
                )
                fig_d = px.bar(
                    dept_df,
                    x="Department",
                    y="attrition_rate_pct",
                    color="attrition_rate_pct",
                    color_continuous_scale="Reds",
                    text=dept_df["attrition_rate_pct"].round(1),
                    labels={"attrition_rate_pct": ATTRITION_LABEL},
                )
                fig_d.update_traces(textposition="outside", cliponaxis=False)
                fig_d.update_layout(
                    height=CHART_HEIGHT,
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis_title=ATTRITION_LABEL,
                    xaxis_title="Department",
                )
                st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(
                "*Reading:* Departments farther right show a **larger share of employees who left**. "
                "Use this to prioritize leadership conversations and HR business partner time."
            )

        with r1c2:
            st.markdown("#### How does turnover differ by salary level?")
            if len(filtered):
                sal_df = attrition_rate_pct(filtered, ["salary"])
                _order_map = {"low": 0, "medium": 1, "high": 2}
                order = sorted(
                    sal_df["salary"].unique(),
                    key=lambda x: _order_map.get(str(x).lower(), 99),
                )
                sal_df["salary"] = pd.Categorical(sal_df["salary"], categories=order, ordered=True)
                sal_df = sal_df.sort_values("salary")
                fig_s = px.bar(
                    sal_df,
                    x="salary",
                    y="attrition_rate_pct",
                    color="salary",
                    text=sal_df["attrition_rate_pct"].round(1),
                    labels={"attrition_rate_pct": ATTRITION_LABEL},
                )
                fig_s.update_traces(textposition="outside")
                fig_s.update_layout(
                    height=CHART_HEIGHT,
                    showlegend=False,
                    yaxis_title=ATTRITION_LABEL,
                    xaxis_title="Salary level",
                )
                st.plotly_chart(fig_s, use_container_width=True)
            st.markdown(
                "*Reading:* If lower pay bands show higher attrition, employees may be leaving for **better pay elsewhere** "
                "or feel **undervalued** — pair with compensation reviews and career-path clarity."
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("#### Satisfaction: who is more at risk?")
            if len(filtered):
                plot_df = filtered.assign(Outcome=outcome_label(filtered["left"]))
                fig_h = px.histogram(
                    plot_df,
                    x="satisfaction_level",
                    color="Outcome",
                    barmode="overlay",
                    opacity=0.55,
                    nbins=35,
                    category_orders={"Outcome": ["Stayed", "Left"]},
                    color_discrete_map={"Stayed": "#2E7D32", "Left": "#C62828"},
                    labels={"satisfaction_level": "Satisfaction score"},
                )
                fig_h.update_layout(
                    height=CHART_HEIGHT,
                    legend_title_text="",
                    yaxis_title="Number of employees",
                )
                st.plotly_chart(fig_h, use_container_width=True)
            st.markdown(
                "*Reading:* A **left shift** (more leavers at low satisfaction) signals **engagement and manager quality** "
                "as levers — not just pay."
            )

        with r2c2:
            st.markdown("#### Satisfaction vs. workload (monthly hours)")
            if len(filtered):
                sc_df = filtered.assign(Outcome=outcome_label(filtered["left"]))
                fig_sc = px.scatter(
                    sc_df,
                    x="satisfaction_level",
                    y="average_montly_hours",
                    color="Outcome",
                    opacity=0.35,
                    category_orders={"Outcome": ["Stayed", "Left"]},
                    color_discrete_map={"Stayed": "#2E7D32", "Left": "#C62828"},
                    labels={
                        "satisfaction_level": "Satisfaction",
                        "average_montly_hours": "Average monthly hours",
                    },
                )
                fig_sc.update_layout(height=CHART_HEIGHT + 40, legend_title_text="")
                st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown(
                "*Reading:* Many leavers cluster in **high hours + lower satisfaction** — a classic **burnout** pattern. "
                "Use this to target workload and expectations, not only hiring."
            )

    # =========================================================================
    # Risk Drivers — stayed vs left
    # =========================================================================
    with tab_risk:
        st.subheader("What separates employees who stayed from those who left?")
        st.caption(
            "Side-by-side averages help explain **why** turnover might be happening — without statistical models."
        )
        if len(filtered) < 2:
            st.info("Widen your filters to include more employees.")
        else:
            ol = outcome_label(filtered["left"])
            # Numeric comparisons
            metrics = [
                ("satisfaction_level", "Satisfaction (0–1)", "Higher is generally healthier."),
                ("average_montly_hours", "Average monthly hours", "Very high hours can signal overload."),
                ("number_project", "Number of active projects", "Too many projects can split focus."),
                ("time_spend_company", "Years with the company", "Tenure often reflects fit and experience."),
            ]
            comp_rows = []
            for col, label, _hint in metrics:
                g = filtered.groupby(ol, observed=False)[col].mean()
                comp_rows.append(
                    {
                        "Metric": label,
                        "Stayed (avg)": g.get("Stayed", np.nan),
                        "Left (avg)": g.get("Left", np.nan),
                    }
                )
            comp_tbl = pd.DataFrame(comp_rows)
            st.markdown("##### Average profile: key numbers")
            st.dataframe(
                comp_tbl.round(2),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                "*Tip for class:* Point to the **largest gaps** between columns — those are your first stories for HR and leaders."
            )

            # Faceted bars so each metric keeps its own scale (hours vs 0–1 satisfaction cannot share one axis)
            st.markdown("##### Visual comparison (averages)")
            long_df = []
            for col, label, _ in metrics:
                g = filtered.groupby(ol, observed=False)[col].mean().reset_index()
                g.columns = ["Outcome", "value"]
                g["Metric"] = label
                long_df.append(g)
            if long_df:
                plot_long = pd.concat(long_df, ignore_index=True)
                fig_b = px.bar(
                    plot_long,
                    x="Outcome",
                    y="value",
                    facet_col="Metric",
                    facet_col_wrap=2,
                    color="Outcome",
                    color_discrete_map={"Stayed": "#2E7D32", "Left": "#C62828"},
                    category_orders={"Outcome": ["Stayed", "Left"]},
                )
                fig_b.update_layout(
                    height=520,
                    showlegend=False,
                )
                fig_b.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(fig_b, use_container_width=True)

            # Salary: share of stayed vs left in each band (composition)
            st.markdown("##### Salary mix: stayed vs left")
            st.caption("Shows **what share** of each outcome sits in each pay band — not attrition rate.")
            if "salary" in filtered.columns:
                o = outcome_label(filtered["left"])
                ct = pd.crosstab(o, filtered["salary"], normalize="index") * 100.0
                ct.index.name = "Outcome"
                ct = ct.reset_index().melt(
                    id_vars="Outcome", var_name="salary", value_name="Share of group (%)"
                )
                fig_sal = px.bar(
                    ct,
                    x="salary",
                    y="Share of group (%)",
                    color="Outcome",
                    barmode="group",
                    color_discrete_map={"Stayed": "#2E7D32", "Left": "#C62828"},
                    category_orders={"Outcome": ["Stayed", "Left"]},
                )
                fig_sal.update_layout(height=380, legend_title_text="")
                st.plotly_chart(fig_sal, use_container_width=True)
            st.markdown(
                "*Reading:* If **leavers** are concentrated in lower salary bands, combine this with the Executive view on **attrition rate by salary** "
                "to make a clear case for pay and progression."
            )

    # =========================================================================
    # Department drilldown
    # =========================================================================
    with tab_dept:
        st.subheader("Focus on one department")
        st.caption("Pick a department below. KPIs and charts update for **that team only** (still respects sidebar filters).")
        if len(filtered) == 0:
            st.warning("No employees in view — adjust sidebar filters.")
        else:
            dept_list = sorted(filtered["Department"].unique().tolist())
            pick = st.selectbox(
                "Which department are you presenting?",
                options=dept_list,
                index=0,
            )
            sub = filtered[filtered["Department"] == pick]
            st.markdown(f"### {pick}")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Team size", f"{len(sub):,}")
            with k2:
                st.metric(
                    "Attrition rate",
                    f"{100.0 * sub['left'].mean():.1f}%",
                )
            with k3:
                st.metric("Avg satisfaction", f"{sub['satisfaction_level'].mean():.2f}")
            with k4:
                st.metric(
                    "Avg monthly hours",
                    f"{sub['average_montly_hours'].mean():.0f}",
                )

            # Benchmark vs full filtered population
            bench_attr = 100.0 * filtered["left"].mean()
            bench_sat = filtered["satisfaction_level"].mean()
            st.markdown(
                f"**Context (all employees in current filter):** attrition **{bench_attr:.1f}%**, "
                f"avg satisfaction **{bench_sat:.2f}**."
            )

            st.markdown(f"#### Attrition rate by salary — {pick}")
            sal_sub = attrition_rate_pct(sub, ["salary"])
            if len(sal_sub):
                fig_ds = px.bar(
                    sal_sub,
                    x="salary",
                    y="attrition_rate_pct",
                    text=sal_sub["attrition_rate_pct"].round(1),
                    labels={"attrition_rate_pct": ATTRITION_LABEL},
                )
                fig_ds.update_traces(textposition="outside")
                fig_ds.update_layout(height=CHART_HEIGHT, yaxis_title=ATTRITION_LABEL)
                st.plotly_chart(fig_ds, use_container_width=True)
            st.markdown(
                "*Reading:* Compare pay bands inside this department to see if **retention issues are pay-related** or broader."
            )

            st.markdown(f"#### Satisfaction vs. hours — {pick}")
            sub_p = sub.assign(Outcome=outcome_label(sub["left"]))
            fig_sub = px.scatter(
                sub_p,
                x="satisfaction_level",
                y="average_montly_hours",
                color="Outcome",
                opacity=0.4,
                color_discrete_map={"Stayed": "#2E7D32", "Left": "#C62828"},
                labels={
                    "satisfaction_level": "Satisfaction",
                    "average_montly_hours": "Average monthly hours",
                },
            )
            fig_sub.update_layout(height=CHART_HEIGHT + 20, legend_title_text="")
            st.plotly_chart(fig_sub, use_container_width=True)
            st.markdown(
                "*Reading:* Tight clusters of **high hours and low satisfaction** in this department are good candidates for **manager coaching** and **capacity planning**."
            )

    # =========================================================================
    # Predictive Risk Monitor — department drilldown (rule-based score)
    # =========================================================================
    with tab_pred:
        st.subheader("Predictive Risk Monitor")
        st.caption(
            "Drill into **each department**: how many **current employees** fall into each risk band, "
            "and which **risk drivers** show up most often. Scores are **0** for **7+ years** tenure or a **recent promotion**."
        )
        current = filtered[filtered["left"] == 0].copy()
        if len(current) == 0:
            st.warning("No current employees in the filtered population (or everyone has left). Adjust sidebar filters.")
        elif not risk_level_selection:
            st.info("Select at least one **Risk levels to show** option in the sidebar to build the charts.")
        else:
            scored = compute_employee_risk_scores(current)
            departments = sorted(scored["Department"].astype(str).unique().tolist())
            focus = st.selectbox(
                "Jump to department",
                options=["(Show all below)"] + departments,
                index=0,
            )

            for dep in departments:
                if focus != "(Show all below)" and dep != focus:
                    continue
                sub = scored[scored["Department"] == dep]
                n_emp = len(sub)
                exp_label = f"{dep} — {n_emp:,} current employee(s)"
                with st.expander(exp_label, expanded=(focus == dep)):
                    # Headline KPIs for this department
                    c1, c2, c3 = st.columns(3)
                    vc_all = sub["Risk_Level"].value_counts()
                    with c1:
                        st.metric("High + Critical", f"{int(vc_all.get('High', 0) + vc_all.get('Critical', 0)):,}")
                    with c2:
                        st.metric("Critical only", f"{int(vc_all.get('Critical', 0)):,}")
                    with c3:
                        st.metric("Median risk score", f"{sub['Risk_Score'].median():.0f}")

                    # Bar chart: employees by risk category (sidebar controls visible bands)
                    chart_levels = [lv for lv in RISK_LEVEL_ORDER if lv in risk_level_selection]
                    bar_rows = [
                        {"Risk level": lv, "Employees": int(vc_all.get(lv, 0))}
                        for lv in chart_levels
                    ]
                    bar_df = pd.DataFrame(bar_rows)
                    fig_risk = px.bar(
                        bar_df,
                        x="Risk level",
                        y="Employees",
                        color="Risk level",
                        color_discrete_map=RISK_LEVEL_COLORS,
                        category_orders={"Risk level": chart_levels},
                        text="Employees",
                    )
                    fig_risk.update_traces(textposition="outside")
                    fig_risk.update_layout(
                        height=CHART_HEIGHT,
                        showlegend=False,
                        yaxis_title="Number of employees",
                        xaxis_title="",
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                    # Main risk reasons in this department (frequency of each rule among staff)
                    reason_df = summarize_department_risk_reasons(sub)
                    st.markdown("**Main risk drivers in this department**")
                    if reason_df.empty:
                        st.caption(
                            "No rule-based risk flags in this group (e.g. all exempt, or no matching patterns)."
                        )
                    else:
                        fig_rs = px.bar(
                            reason_df,
                            x="Employees",
                            y="Reason",
                            orientation="h",
                            text="Employees",
                            color_discrete_sequence=["#5C6BC0"],
                        )
                        fig_rs.update_traces(textposition="outside")
                        fig_rs.update_layout(
                            height=min(420, 60 + 40 * len(reason_df)),
                            yaxis=dict(autorange="reversed"),
                            xaxis_title="Employees with this factor",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_rs, use_container_width=True)
                        st.caption(
                            "Counts reflect how many **current employees** match each rule (exempt staff are excluded from flags)."
                        )

                    # Nested drilldown: employee rows for risk band × reason
                    st.markdown("---")
                    with st.expander("Employee list (risk category × risk reason)", expanded=False):
                        st.caption(
                            "Pick a **risk category** and optionally a **specific risk reason** to list employees in this department. "
                            "**CSV line** matches the row in `hr_data.csv` (line 1 is the header)."
                        )
                        reason_choices: list[tuple[str, str | None]] = [
                            ("All employees in this risk band", None)
                        ]
                        reason_choices.extend((lab, col) for col, lab in RISK_REASON_DEFINITIONS)
                        r_labels = [x[0] for x in reason_choices]
                        r_map = dict(reason_choices)

                        ec1, ec2 = st.columns(2)
                        with ec1:
                            sel_cat = st.selectbox(
                                "Risk category",
                                options=RISK_LEVEL_ORDER,
                                index=0,
                                key=f"prm_cat_{dep}",
                            )
                        with ec2:
                            sel_reason_label = st.selectbox(
                                "Risk reason",
                                options=r_labels,
                                index=0,
                                key=f"prm_reason_{dep}",
                            )
                        reason_col = r_map[sel_reason_label]
                        sliced = slice_employees_by_risk(sub, sel_cat, reason_col)
                        disp_tbl = employee_risk_drilldown_table(sliced)
                        if disp_tbl.empty:
                            st.info(
                                "No employees match this combination. "
                                "Try **All employees in this risk band** or another reason."
                            )
                        else:
                            st.dataframe(disp_tbl, use_container_width=True, hide_index=True)
                            st.caption(f"**{len(disp_tbl):,}** employee(s) in this slice.")

            if focus != "(Show all below)":
                st.caption("Use the dropdown above to switch departments, or choose **(Show all below)** to open every section.")

    # =========================================================================
    # Capacity & Stability
    # =========================================================================
    with tab_cap_stab:
        st.subheader("Capacity & Stability")
        st.caption(
            "Use this tab to learn three things: where workload is heavy, who may be underused, "
            "and which employees may be reliable team anchors or mentors."
        )

        # ---------------------------------------------------------------------
        # 1) Capacity Planning
        # ---------------------------------------------------------------------
        st.markdown("### 1) Capacity Planning")
        cap_df = filtered.copy()
        over_5_mask = cap_df["number_project"] > 5
        over_230h_mask = cap_df["average_montly_hours"] >= 230

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Employees over 5 projects", f"{int(over_5_mask.sum()):,}")
        with k2:
            st.metric("Employees at or above 230 monthly hours", f"{int(over_230h_mask.sum()):,}")
        with k3:
            st.metric("Average projects per employee", f"{cap_df['number_project'].mean():.2f}")
        with k4:
            st.metric("Average monthly hours", f"{cap_df['average_montly_hours'].mean():.0f}")

        dep_sum = (
            cap_df.groupby("Department", observed=False)
            .agg(
                employee_count=("left", "count"),
                avg_projects=("number_project", "mean"),
                max_projects=("number_project", "max"),
                employees_over_5_projects=("number_project", lambda s: int((s > 5).sum())),
                avg_monthly_hours=("average_montly_hours", "mean"),
            )
            .reset_index()
        )
        dep_sum["avg_projects"] = dep_sum["avg_projects"].round(2)
        dep_sum["avg_monthly_hours"] = dep_sum["avg_monthly_hours"].round(1)
        st.markdown("#### Department workload summary")
        st.dataframe(dep_sum.sort_values("avg_projects", ascending=False), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            avg_proj = (
                cap_df.groupby("Department", observed=False)["number_project"]
                .mean()
                .reset_index(name="avg_number_project")
            )
            fig_ap = px.bar(
                avg_proj,
                x="Department",
                y="avg_number_project",
                title="Average number of projects by department",
                color_discrete_sequence=["#3B82F6"],
            )
            fig_ap.update_layout(height=320, yaxis_title="Average number of projects")
            st.plotly_chart(fig_ap, use_container_width=True)
        with c2:
            over_5_dept = (
                cap_df.assign(over_5=cap_df["number_project"] > 5)
                .groupby("Department", observed=False)["over_5"]
                .sum()
                .reset_index(name="employees_over_5_projects")
            )
            fig_o5 = px.bar(
                over_5_dept,
                x="Department",
                y="employees_over_5_projects",
                title="Employees over 5 projects by department",
                color_discrete_sequence=["#F97316"],
            )
            fig_o5.update_layout(height=320, yaxis_title="Employees")
            st.plotly_chart(fig_o5, use_container_width=True)

        st.markdown("#### Employee workload table")
        cap_cols = [
            "employee_id",
            "Department",
            "salary",
            "time_spend_company",
            "number_project",
            "average_montly_hours",
            "satisfaction_level",
            "last_evaluation",
            "left",
        ]
        st.dataframe(
            cap_df[cap_cols].sort_values(["number_project", "average_montly_hours"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # ---------------------------------------------------------------------
        # 2) Growth Support Candidates
        # ---------------------------------------------------------------------
        st.markdown("### 2) Growth Support Candidates")
        growth_candidates = cap_df[
            (cap_df["time_spend_company"] == 2) & (cap_df["number_project"] < 3)
        ].copy()
        st.metric("Employees flagged for growth support", f"{len(growth_candidates):,}")
        st.dataframe(
            growth_candidates[cap_cols].sort_values(["Department", "number_project"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These employees may be underutilized and could benefit from a steadier 3-4 project workload "
            "to support engagement and trust."
        )

        st.divider()

        # ---------------------------------------------------------------------
        # 3) Stable / Mentor Candidates
        # ---------------------------------------------------------------------
        st.markdown("### 3) Stable / Mentor Candidates")
        mentor_candidates = cap_df[
            (cap_df["time_spend_company"] >= 7) | (cap_df["promotion_last_5years"] == 1)
        ].copy()
        stable_pool = cap_df[
            (
                ((cap_df["time_spend_company"] >= 7) | (cap_df["promotion_last_5years"] == 1))
                & (cap_df["last_evaluation"] < 0.80)
                & (cap_df["number_project"] <= 5)
                & (cap_df["satisfaction_level"] > 0.20)
                & (cap_df["salary"].astype(str).str.lower() != "low")
            )
        ].copy()

        s1, s2 = st.columns(2)
        with s1:
            st.metric("Mentor candidate count", f"{len(mentor_candidates):,}")
        with s2:
            st.metric("Stable employee pool count", f"{len(stable_pool):,}")

        st.markdown("#### Mentor candidate table")
        st.dataframe(
            mentor_candidates[cap_cols + ["promotion_last_5years"]].sort_values(
                ["time_spend_company", "last_evaluation"], ascending=[False, False]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Stable employee pool table")
        st.dataframe(
            stable_pool[cap_cols + ["promotion_last_5years"]].sort_values(
                ["time_spend_company", "number_project"], ascending=[False, True]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These employees may serve as stable team anchors and practical mentors for colleagues "
            "who are showing heavier workload or retention risk signs."
        )

    # =========================================================================
    # Recommendations
    # =========================================================================
    with tab_rec:
        st.subheader("Recommended actions for leadership & HR")
        st.caption(
            "This page turns the dashboard signals into clear actions. "
            "Each section shows the condition and the recommended next step."
        )

        rec_df = filtered.copy()
        dept_attr = attrition_rate_pct(rec_df, ["Department"])
        dept_alerts = dept_attr[dept_attr["attrition_rate_pct"] >= 20].copy()

        mask_over_5 = rec_df["number_project"] > 5
        mask_hours_230 = rec_df["average_montly_hours"] >= 230
        mask_low_sat = rec_df["satisfaction_level"] <= 0.20
        mask_growth = (rec_df["time_spend_company"] == 2) & (rec_df["number_project"] < 3)
        mask_fast_track = (rec_df["last_evaluation"] >= 0.80) & (rec_df["time_spend_company"] == 4)
        mask_low_salary = rec_df["salary"].astype(str).str.lower() == "low"
        mask_mentor = (rec_df["time_spend_company"] >= 7) | (rec_df["promotion_last_5years"] == 1)
        mask_stable = (
            ((rec_df["time_spend_company"] >= 7) | (rec_df["promotion_last_5years"] == 1))
            & (rec_df["last_evaluation"] < 0.80)
            & (rec_df["number_project"] <= 5)
            & (rec_df["satisfaction_level"] > 0.20)
            & (rec_df["salary"].astype(str).str.lower() != "low")
        )

        st.markdown("### Workload Pressure")
        st.markdown(
            f"**Condition:** {int(mask_over_5.sum()):,} employees have more than 5 projects and "
            f"{int(mask_hours_230.sum()):,} are at or above 230 monthly hours."
        )
        st.markdown(
            "**Action:** Rebalance assignments this cycle, cap project load near 4–5 for overloaded roles, "
            "and prioritize manager approvals for any exceptions."
        )

        st.markdown("### Low Satisfaction Risk")
        st.markdown(
            f"**Condition:** {int(mask_low_sat.sum()):,} employees are at or below 0.20 satisfaction."
        )
        st.markdown(
            "**Action:** Schedule stay interviews in the next 2 weeks, document top themes, and assign "
            "owner-led follow-ups for each high-risk employee."
        )

        st.markdown("### Early Tenure Underutilization")
        st.markdown(
            f"**Condition:** {int(mask_growth.sum()):,} employees are at 2 years tenure with low project load."
        )
        st.markdown(
            "**Action:** Move these employees toward a steady 3–4 project workload with clear role goals "
            "to improve engagement and confidence."
        )

        st.markdown("### Promotion Readiness")
        st.markdown(
            f"**Condition:** {int(mask_fast_track.sum()):,} employees have high evaluation scores at 4 years tenure."
        )
        st.markdown(
            "**Action:** Run a fast-track promotion review list with HR and managers, and confirm decisions "
            "within the current performance cycle."
        )

        st.markdown("### Salary Risk")
        st.markdown(
            f"**Condition:** {int(mask_low_salary.sum()):,} employees are in the low salary group."
        )
        st.markdown(
            "**Action:** Launch salary benchmarking for high-turnover roles, then align pay adjustments and "
            "career-path communication for impacted teams."
        )

        st.markdown("### Stable and Mentor Candidates")
        st.markdown(
            f"**Condition:** {int(mask_mentor.sum()):,} potential mentor candidates and {int(mask_stable.sum()):,} "
            "employees in the stable pool."
        )
        st.markdown(
            "**Action:** Pair selected stable employees with at-risk team members for onboarding, coaching, "
            "and weekly check-ins."
        )

        st.markdown("### Department-Level Attrition")
        if len(dept_alerts) == 0:
            st.markdown("**Condition:** No departments are currently at or above 20% attrition.")
            st.markdown(
                "**Action:** Continue monthly monitoring and keep current manager support plans in place."
            )
        else:
            st.markdown(
                f"**Condition:** {len(dept_alerts):,} departments are at or above 20% attrition."
            )
            st.markdown(
                "**Action:** Run a manager effectiveness audit in flagged departments and review workload "
                "planning, coaching quality, and growth opportunities."
            )
            show_cols = ["Department", "headcount", "attrition_rate_pct"]
            dept_show = dept_alerts[show_cols].copy().sort_values("attrition_rate_pct", ascending=False)
            dept_show["attrition_rate_pct"] = dept_show["attrition_rate_pct"].round(1)
            st.dataframe(dept_show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
