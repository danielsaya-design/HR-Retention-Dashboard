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

    tab_exec, tab_risk, tab_dept, tab_rec = st.tabs(
        [
            "Executive Overview",
            "Risk Drivers",
            "Department Drilldown",
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
    # Recommendations
    # =========================================================================
    with tab_rec:
        st.subheader("Recommended actions for leadership & HR")
        st.caption(
            "These suggestions are **generated from the filtered dataset** — same filters as the rest of the dashboard."
        )
        items = build_recommendations(filtered)
        for i, text in enumerate(items, start=1):
            st.markdown(f"{i}. {text}")
        st.divider()
        st.markdown(
            "**Presenting in class:** Walk through *Executive Overview* first, then *Risk Drivers*, "
            "then one department in *Department Drilldown*, and end with this slide — tie each recommendation "
            "back to a chart you showed."
        )


if __name__ == "__main__":
    main()
