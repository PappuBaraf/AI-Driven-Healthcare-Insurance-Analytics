"""
AI-Driven Healthcare & Insurance Analytics Dashboard
IBM SkillsBuild Data Analytics with AI Internship

Run:
    pip install -r requirements.txt
    streamlit run app.py

Pipeline must be run first:
    python pipeline.py

IMPORTANT:
- The dataset is synthetic (CA Hospital Dataset Q1 2025).
- This dashboard is for analytics / educational use only.
- It does not diagnose patients or provide medical advice.
- It does not claim causation from observational patterns.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE   = Path(__file__).parent
DATA   = BASE / "data"
OUT    = BASE / "outputs"
CHARTS = OUT  / "charts"
PREDS  = OUT  / "predictions"
REPS   = OUT  / "reports"

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare & Insurance Analytics | Pappu Ramesh Baraf",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Project Owner Info
# -----------------------------------------------------------------------
OWNER = {
    "name":     "PAPPU RAMESH BARAF",
    "title":    "Software Developer | AI/ML | Generative AI | Data Analytics",
    "location": "Ratnagiri, Maharashtra, India",
    "email":    "pappubaraf1@gmail.com",
    "linkedin": "https://in.linkedin.com/in/pappubarafaiml",
    "github":   "https://github.com/PappuBaraf",
    "college":  "Finolex Academy of Management & Technology (FAMT), Ratnagiri",
    "degree":   "B.E. Computer Science & Engineering (AI/ML)  |  2023–2027",
    "summary": (
        "Final-year B.E. CSE (AI/ML) student at FAMT, Ratnagiri. "
        "Hands-on experience in Python, JavaScript, REST APIs, React, FastAPI, "
        "AWS, and AI-powered application development. "
        "Completed Lenovo LEAP NextGen Scholar Program and "
        "IBM SkillsBuild GenAI & Cloud Computing Internship (Bharat-Cares / AICTE)."
    ),
}

# -----------------------------------------------------------------------
# Loaders (cached)
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_tables():
    tables = {}
    for f in sorted(DATA.glob("*.csv")):
        try:
            tables[f.stem] = pd.read_csv(f, low_memory=False)
        except Exception as exc:
            st.warning(f"Could not read {f.name}: {exc}")
    return tables


@st.cache_data(show_spinner=False)
def load_output(filename):
    p = OUT / filename
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_predictions():
    p = PREDS / "model_predictions.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_model_summary():
    p = OUT / "model_summary.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return None


def load_chart(name):
    p = CHARTS / name
    if p.exists():
        return mpimg.imread(str(p))
    return None


def show_chart(name, caption=""):
    img = load_chart(name)
    if img is not None:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart not found: {name}. Run pipeline.py first.")


def add_download(df, label, filename):
    if df is not None and not df.empty:
        st.download_button(
            label=label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )


# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
tables     = load_raw_tables()
claims_raw = tables.get("claims_and_billing")

model_summary   = load_model_summary()
metrics_df      = load_output("model_metrics.csv")
predictions_df  = load_predictions()
ai_insights_df  = load_output("ai_insights.csv")
eda_insights_df = load_output("eda_insights.csv")
data_dict_df    = load_output("data_dictionary.csv")
dq_df           = load_output("data_quality_report.csv")
feat_df         = load_output("feature_dictionary.csv")
ml_feat_df      = load_output("ml_feature_selection.csv")
prof_df         = load_output("dataset_profile.csv")
cr_df           = load_output("classification_report.csv")

PIPELINE_RUN = model_summary is not None

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("🏥 AI-Driven Healthcare & Insurance Analytics")
st.caption(
    "Patient Data • Insurance Claims • Medical Outcomes • "
    "EDA • Predictive Analytics • AI-Driven Insights  |  "
    "IBM SkillsBuild Data Analytics with AI Internship"
)

if not PIPELINE_RUN:
    st.error(
        "⚠️ Pipeline outputs not found. Run `python pipeline.py` from the project folder "
        "to generate all charts, metrics and predictions before using this dashboard."
    )

st.info(
    "📌 This dashboard uses the **CA Hospital Dataset – Q1 2025** (synthetic data). "
    "All patterns shown are analytical observations. "
    "No medical diagnoses or causal conclusions are claimed."
)

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Navigation")
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

    page = st.radio(
        "Go to page",
        [
            "🏠 Executive Overview",
            "📂 Load Database",
            "💳 Claims & Insurance",
            "📊 Exploratory Data Analysis",
            "🤖 Predictive Analytics",
            "📈 Model Results",
            "💡 AI-Driven Insights",
            "🧾 Data Dictionary",
            "🔍 Data Quality",
            "⬇️ Export",
            "👤 About / Project Owner",
        ],
    )

    st.divider()
    if claims_raw is not None:
        st.metric("Dataset Rows", f"{len(claims_raw):,}")
        st.metric("Tables Loaded", len(tables))
    st.caption("Synthetic dataset — educational use only")

    # ── Project Owner Card ──
    st.divider()
    st.markdown("**👨‍💻 Project Owner**")
    st.markdown(f"**{OWNER['name']}**")
    st.markdown(f"*{OWNER['title']}*")
    st.markdown(f"📍 {OWNER['location']}")
    st.markdown(f"✉️ {OWNER['email']}")
    st.markdown(
        f"[🔗 LinkedIn]({OWNER['linkedin']})  ·  [🐙 GitHub]({OWNER['github']})"
    )
    st.caption(OWNER["degree"])


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Overview":
    st.subheader("Executive Overview")

    if claims_raw is not None:
        cb = claims_raw
        total        = len(cb)
        denied       = (cb["claim_status"] == "Denied").sum()
        paid         = (cb["claim_status"] == "Paid").sum()
        total_billed = cb["billed_amount"].sum()
        total_paid   = cb["paid_amount"].sum()
        avg_billed   = cb["billed_amount"].mean()
        n_providers  = cb["insurance_provider"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Claims", f"{total:,}")
        c2.metric("Paid Claims", f"{paid:,}", f"{paid/total*100:.1f}%")
        c3.metric("Denied Claims", f"{denied:,}", f"-{denied/total*100:.1f}%", delta_color="inverse")
        c4.metric("Insurance Providers", f"{n_providers}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Billed", f"${total_billed:,.0f}")
        c6.metric("Total Paid", f"${total_paid:,.0f}")
        c7.metric("Avg Billed / Claim", f"${avg_billed:,.2f}")
        c8.metric("Revenue Gap", f"${total_billed - total_paid:,.0f}")

    st.divider()
    st.markdown("### Dataset Summary")
    if prof_df is not None:
        st.dataframe(prof_df, use_container_width=True)
    else:
        if claims_raw is not None:
            st.dataframe(
                pd.DataFrame([{
                    "Table": "claims_and_billing",
                    "Rows": len(claims_raw),
                    "Columns": len(claims_raw.columns),
                    "Missing Cells": int(claims_raw.isna().sum().sum()),
                    "Duplicate Rows": int(claims_raw.duplicated().sum()),
                }]),
                use_container_width=True,
            )

    st.divider()
    st.markdown("### Project Analytics Scope")
    a1, a2, a3, a4 = st.columns(4)
    a1.markdown("**💳 Claims & Billing**\n\n70,000 claim records covering payer mix, billing amounts, claim status and denial reasons")
    a2.markdown("**📊 Exploratory Analysis**\n\n10 EDA charts revealing cost distributions, trends, and payer patterns")
    a3.markdown("**🤖 Predictive Model**\n\nLogistic Regression to classify claim approval (Paid / Denied)")
    a4.markdown("**💡 AI Insights**\n\nAutomatically generated operational observations and recommendations")

    st.divider()
    st.markdown("### Important Interpretation Guidance")
    st.warning(
        "This is a **synthetic** dataset. Denial rates, billing amounts, and provider "
        "distributions are generated, not observed from any real healthcare network. "
        "All findings are analytical observations — not causal conclusions."
    )


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — LOAD DATABASE
# ═══════════════════════════════════════════════════════════════════════
elif page == "📂 Load Database":
    st.subheader("Load Database — Manage Data Sources")

    st.info(
        "This page lets you **upload your own CSV files** to add more tables to the dashboard, "
        "or enter individual records manually. "
        "Uploaded files are loaded into memory for this session only — the original `data/` files are not modified."
    )

    # ── Tab 1: CSV Upload  |  Tab 2: Manual Entry  |  Tab 3: Current Data ──
    tab_up, tab_manual, tab_current = st.tabs(
        ["⬆️ Upload CSV Files", "✏️ Manual Data Entry", "📋 Current Loaded Data"]
    )

    # ------------------------------------------------------------------
    # TAB 1 — CSV Upload
    # ------------------------------------------------------------------
    with tab_up:
        st.markdown("### Upload Additional CSV Files")
        st.write(
            "Upload one or more CSV files. They will be added to the current session alongside "
            "the existing `claims_and_billing.csv` from the `data/` folder."
        )

        uploaded_files = st.file_uploader(
            "Choose CSV file(s)",
            type=["csv"],
            accept_multiple_files=True,
            help="Supports any CSV with a header row. Files are loaded into memory for this session.",
        )

        if uploaded_files:
            for uf in uploaded_files:
                try:
                    df_up = pd.read_csv(uf, low_memory=False)
                    st.success(f"✅ Loaded **{uf.name}** — {len(df_up):,} rows × {len(df_up.columns)} columns")
                    st.markdown(f"#### Preview: {uf.name}")
                    st.dataframe(df_up.head(20), use_container_width=True)

                    col_info = []
                    for c in df_up.columns:
                        col_info.append({
                            "Column": c,
                            "Type": str(df_up[c].dtype),
                            "Missing": int(df_up[c].isna().sum()),
                            "Missing %": round(df_up[c].isna().mean() * 100, 2),
                            "Unique": int(df_up[c].nunique()),
                        })
                    st.markdown("##### Column Profile")
                    st.dataframe(pd.DataFrame(col_info), use_container_width=True)

                    add_download(df_up, f"⬇️ Download {uf.name}", uf.name)

                except Exception as exc:
                    st.error(f"Could not read {uf.name}: {exc}")
        else:
            st.markdown(
                "**Expected CSV files for the full dataset:**\n\n"
                "| File | Description |\n"
                "|------|-------------|\n"
                "| `patients.csv` | Patient demographics |\n"
                "| `encounters.csv` | Hospital visit / encounter records |\n"
                "| `diagnoses.csv` | Diagnosis codes per encounter |\n"
                "| `procedures.csv` | Procedure codes per encounter |\n"
                "| `medications.csv` | Medication records |\n"
                "| `lab_tests.csv` | Laboratory test results |\n"
                "| `claims_and_billing.csv` | Claims & billing (**already loaded**) |\n"
                "| `providers.csv` | Healthcare provider details |\n"
                "| `denials.csv` | Denial records |\n\n"
                "Download from [Kaggle — CA Hospital Dataset Q1 2025]"
                "(https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025)"
            )

    # ------------------------------------------------------------------
    # TAB 2 — Manual Data Entry
    # ------------------------------------------------------------------
    with tab_manual:
        st.markdown("### Manual Claim Record Entry")
        st.write(
            "Enter a single claim record manually. "
            "The record will be appended to the current session dataset for review."
        )

        with st.form("manual_entry_form"):
            st.markdown("#### New Claim Record")
            mc1, mc2 = st.columns(2)
            with mc1:
                m_patient_id   = st.text_input("Patient ID",         value="PAT000001", help="e.g. PAT000001")
                m_encounter_id = st.text_input("Encounter ID",       value="ENC000001")
                m_claim_id     = st.text_input("Claim ID",           value="CLM000001")
                m_billing_id   = st.text_input("Billing ID",         value="BILL000001")
                m_billing_date = st.date_input("Claim Billing Date")
            with mc2:
                m_provider     = st.selectbox(
                    "Insurance Provider",
                    ["BCBS", "Aetna", "Cigna", "Humana", "Medicaid", "Medicare", "UHC", "Other"],
                )
                m_method       = st.selectbox("Payment Method", ["Insurance", "Selfpay"])
                m_status       = st.selectbox("Claim Status",   ["Paid", "Denied"])
                m_billed       = st.number_input("Billed Amount ($)", min_value=0.0, value=1500.0, step=50.0)
                m_paid         = st.number_input(
                    "Paid Amount ($)",
                    min_value=0.0,
                    value=0.0 if m_status == "Denied" else 975.0,
                    step=50.0,
                )
                m_denial       = st.text_input(
                    "Denial Reason",
                    value="" if m_status == "Paid" else "Duplicate Claim",
                    disabled=(m_status == "Paid"),
                )

            submitted = st.form_submit_button("➕ Add Record to Session")

        if submitted:
            new_row = {
                "billing_id":          m_billing_id,
                "patient_id":          m_patient_id,
                "encounter_id":        m_encounter_id,
                "insurance_provider":  m_provider,
                "payment_method":      m_method,
                "claim_id":            m_claim_id,
                "claim_billing_date":  m_billing_date.strftime("%d-%m-%Y 00:00"),
                "billed_amount":       m_billed,
                "paid_amount":         m_paid if m_status == "Paid" else 0.0,
                "claim_status":        m_status,
                "denial_reason":       m_denial if m_status == "Denied" else None,
            }
            new_df = pd.DataFrame([new_row])
            st.success("✅ Record added to session preview below.")
            st.markdown("#### New Record Preview")
            st.dataframe(new_df, use_container_width=True)
            add_download(new_df, "⬇️ Download This Record", "manual_claim_entry.csv")

            st.info(
                "To permanently add this record to the dataset, "
                "download the CSV and place it in `data/` or append it to `claims_and_billing.csv`, "
                "then click **Reload Data** in the sidebar."
            )

    # ------------------------------------------------------------------
    # TAB 3 — Current Loaded Data
    # ------------------------------------------------------------------
    with tab_current:
        st.markdown("### Currently Loaded Tables")

        if not tables:
            st.warning("No tables loaded. Place CSV files in the `data/` folder.")
        else:
            for tname, df in tables.items():
                with st.expander(f"**{tname.replace('_',' ').title()}** — {len(df):,} rows × {len(df.columns)} columns"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rows",       f"{len(df):,}")
                    c2.metric("Columns",    f"{len(df.columns)}")
                    c3.metric("Duplicates", f"{df.duplicated().sum():,}")
                    c4.metric("Missing",    f"{df.isna().sum().sum():,}")
                    st.dataframe(df.head(50), use_container_width=True)
                    add_download(df, f"⬇️ Download {tname}", f"{tname}.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — CLAIMS & INSURANCE
# ═══════════════════════════════════════════════════════════════════════
elif page == "💳 Claims & Insurance":
    st.subheader("Claims & Insurance Analytics")

    if claims_raw is None:
        st.error("claims_and_billing.csv not found in data/ folder.")
        st.stop()

    cb = claims_raw.copy()
    cb["claim_billing_date"] = pd.to_datetime(cb["claim_billing_date"], format="%d-%m-%Y %H:%M", errors="coerce")

    total       = len(cb)
    denied      = (cb["claim_status"] == "Denied").sum()
    paid        = (cb["claim_status"] == "Paid").sum()
    avg_billed  = cb["billed_amount"].mean()
    avg_paid    = cb["paid_amount"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{total:,}")
    c2.metric("Paid", f"{paid:,} ({paid/total*100:.1f}%)")
    c3.metric("Denied", f"{denied:,} ({denied/total*100:.1f}%)")
    c4.metric("Avg Billed", f"${avg_billed:,.2f}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Claim Status", "Insurance Providers", "Billing Amounts", "Denial Analysis"]
    )

    with tab1:
        st.markdown("#### Claim Status Distribution")
        show_chart("01_claim_status.png")
        st.markdown("#### Claim Status by Insurance Provider")
        show_chart("08_status_by_provider.png")
        st.markdown("**Observation:** Denial rates are broadly similar across all providers, "
                    "consistent with the synthetic nature of this dataset.")

    with tab2:
        st.markdown("#### Claims by Insurance Provider")
        show_chart("02_insurance_provider.png")
        st.markdown("#### Average Billed vs Paid Amount by Provider")
        show_chart("04_billed_vs_paid_by_provider.png")

        st.markdown("#### Provider Summary Table")
        prov_tbl = cb.groupby("insurance_provider").agg(
            Claims=("billing_id", "count"),
            Paid_Claims=("claim_status", lambda x: (x == "Paid").sum()),
            Denied_Claims=("claim_status", lambda x: (x == "Denied").sum()),
            Avg_Billed=("billed_amount", "mean"),
            Avg_Paid=("paid_amount", "mean"),
            Total_Billed=("billed_amount", "sum"),
        ).round(2).reset_index()
        prov_tbl["Denial_Rate_%"] = (prov_tbl["Denied_Claims"] / prov_tbl["Claims"] * 100).round(2)
        st.dataframe(prov_tbl, use_container_width=True)

    with tab3:
        st.markdown("#### Billed Amount Distribution")
        show_chart("03_billed_amount_dist.png")
        st.markdown("#### Payment Method Distribution")
        show_chart("06_payment_method.png")
        st.markdown("#### Payment Ratio by Claim Status")
        show_chart("10_payment_ratio.png")
        st.markdown("#### High-Cost Claims (>75th Percentile)")
        show_chart("09_high_cost_flag.png")

        st.markdown("#### Billing Statistics")
        stats = cb["billed_amount"].describe().round(2).to_frame("Billed Amount")
        stats2 = cb["paid_amount"].describe().round(2).to_frame("Paid Amount")
        st.dataframe(pd.concat([stats, stats2], axis=1), use_container_width=True)

    with tab4:
        st.markdown("#### Denial Reasons")
        show_chart("05_denial_reasons.png")

        denied_df = cb[cb["claim_status"] == "Denied"].copy()
        dr_counts = denied_df["denial_reason"].value_counts().reset_index()
        dr_counts.columns = ["Denial Reason", "Count"]
        dr_counts["% of Denials"] = (dr_counts["Count"] / len(denied_df) * 100).round(2)
        st.dataframe(dr_counts, use_container_width=True)

        st.info(
            "⚠️ **Leakage Note:** `denial_reason` is only populated **after** a claim is denied. "
            "It was **excluded** from all ML features to prevent target leakage."
        )

    st.divider()
    st.markdown("### Monthly Trends")
    show_chart("07_monthly_trends.png")

    if not cb["claim_billing_date"].isna().all():
        cb["Month"] = cb["claim_billing_date"].dt.to_period("M").astype(str)
        monthly = cb.groupby("Month").agg(
            Claims=("billing_id", "count"),
            Total_Billed=("billed_amount", "sum"),
            Denied=("claim_status", lambda x: (x == "Denied").sum()),
        ).reset_index()
        monthly["Denial_Rate_%"] = (monthly["Denied"] / monthly["Claims"] * 100).round(2)
        st.dataframe(monthly, use_container_width=True)
        st.caption(f"Note: {cb['claim_billing_date'].isna().sum():,} self-pay rows have no billing date and are excluded from this trend.")

    st.divider()
    st.markdown("### Raw Claims Data")
    st.dataframe(cb.head(500), use_container_width=True)
    add_download(cb, "⬇️ Download Claims Data", "claims_and_billing.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    st.write(
        "EDA follows the internship workflow: explore the data, compare groups, "
        "look for patterns, investigate relationships, and clearly distinguish "
        "observations from hypotheses."
    )

    if claims_raw is None:
        st.error("claims_and_billing.csv not found.")
        st.stop()

    cb = claims_raw.copy()

    # Summary stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(cb):,}")
    c2.metric("Unique Patients", f"{cb['patient_id'].nunique():,}")
    c3.metric("Unique Encounters", f"{cb['encounter_id'].nunique():,}")

    st.divider()

    st.markdown("### Chart Gallery")
    charts = [
        ("01_claim_status.png",            "Claim Status Distribution"),
        ("02_insurance_provider.png",       "Claims by Insurance Provider"),
        ("03_billed_amount_dist.png",       "Billed Amount Distribution"),
        ("04_billed_vs_paid_by_provider.png","Avg Billed vs Paid by Provider"),
        ("05_denial_reasons.png",           "Claim Denial Reasons"),
        ("06_payment_method.png",           "Payment Method Distribution"),
        ("08_status_by_provider.png",       "Claim Status % by Provider"),
        ("09_high_cost_flag.png",           "High-Cost Claims Flag"),
        ("10_payment_ratio.png",            "Payment Ratio by Claim Status"),
    ]

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, (fname, cap) in zip(cols, charts[i:i+2]):
            with col:
                st.markdown(f"**{cap}**")
                show_chart(fname)

    st.divider()
    st.markdown("### Monthly Trends")
    show_chart("07_monthly_trends.png")

    st.divider()
    st.markdown("### EDA Insights (from pipeline)")
    if eda_insights_df is not None:
        for _, row in eda_insights_df.iterrows():
            with st.expander(f"📋 {row['Section']} — {row['Question']}"):
                st.write(f"**Observation:** {row['Observation']}")
                st.write(f"**Possible Insight:** {row['Possible Insight']}")
                st.info(f"⚠️ Limitation: {row['Limitation']}")
    else:
        st.info("Run pipeline.py to generate EDA insights.")

    st.divider()
    st.markdown("### Numeric Summary Statistics")
    nums = cb.select_dtypes(include=np.number)
    st.dataframe(nums.describe().T.round(2), use_container_width=True)

    st.markdown("### Missing Value Summary")
    missing = (
        cb.isna().sum()
        .to_frame("Missing Count")
        .assign(**{"Missing %": lambda x: (x["Missing Count"] / len(cb) * 100).round(2)})
        .sort_values("Missing Count", ascending=False)
    )
    st.dataframe(missing, use_container_width=True)
    st.caption("denial_reason is missing for all Paid claims — this is expected (not a data quality issue).")

    st.divider()
    st.markdown("### Interactive Column Explorer")
    col_choice = st.selectbox("Select a column", cb.columns.tolist())
    if pd.api.types.is_numeric_dtype(cb[col_choice]):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(cb[col_choice].dropna(), bins=40, color="#3b82d4", edgecolor="white")
        ax.set_title(f"Distribution: {col_choice}", fontsize=12)
        ax.set_xlabel(col_choice); ax.set_ylabel("Frequency")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        vc = cb[col_choice].astype(str).value_counts().head(20)
        fig, ax = plt.subplots(figsize=(9, 4))
        vc.sort_values().plot(kind="barh", ax=ax, color="#3b82d4")
        ax.set_title(f"Top Categories: {col_choice}", fontsize=12)
        ax.set_xlabel("Count")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Analytics":
    st.subheader("Predictive Analytics — Claim Denial Prediction")

    st.markdown(
        "The pipeline trained a **Logistic Regression** model to predict `claim_status` "
        "(Paid / Denied) using pre-outcome features only."
    )

    st.markdown("### ML Problem Definition")
    with st.expander("🎯 Target Variable", expanded=True):
        st.markdown(
            "- **Target:** `claim_status` (binary: `Paid` or `Denied`)\n"
            "- **Class distribution:** 64,002 Paid (91.4%) / 5,998 Denied (8.6%)\n"
            "- **Imbalance:** Yes — the dataset is highly skewed toward Paid claims\n"
            "- **Primary metric:** ROC-AUC (more informative than accuracy for imbalanced data)"
        )

    with st.expander("🔒 Leakage Prevention"):
        st.markdown(
            "The following columns were **excluded** from model features:\n\n"
            "| Column | Reason |\n"
            "|--------|--------|\n"
            "| `denial_reason` | Post-outcome — only populated after a claim is denied |\n"
            "| `paid_amount` | Post-outcome — only known after claim is processed |\n"
            "| `billing_id` | Identifier — no predictive signal |\n"
            "| `claim_id` | Identifier — no predictive signal |\n"
            "| `patient_id` | Identifier — no predictive signal |\n"
            "| `encounter_id` | Identifier — no predictive signal |\n"
        )

    with st.expander("✅ Features Used"):
        if model_summary:
            for f in model_summary["features_used"]:
                st.write(f"- `{f}`")
        st.caption(
            "Date column (claim_billing_date) was parsed into year, month, "
            "day-of-week and quarter to extract temporal signal without using raw dates."
        )

    st.divider()
    st.markdown("### Feature Selection Table")
    if ml_feat_df is not None:
        st.dataframe(ml_feat_df, use_container_width=True)
        add_download(ml_feat_df, "⬇️ Download Feature Selection", "ml_feature_selection.csv")

    st.divider()
    st.markdown("### Model Architecture")
    st.code(
        "Pipeline:\n"
        "  Numeric features  → MedianImputer → StandardScaler\n"
        "  Categorical feats → MostFreqImputer → OneHotEncoder\n"
        "  Classifier        → LogisticRegression(max_iter=1000, random_state=42)\n\n"
        "  Variant A — default class weights    (primary metrics)\n"
        "  Variant B — class_weight='balanced'  (higher recall for Denied class)",
        language="text",
    )


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📈 Model Results":
    st.subheader("Model Evaluation Results")

    if not PIPELINE_RUN:
        st.error("Run pipeline.py first to generate model results.")
        st.stop()

    ms = model_summary

    # ---- KPI metrics ----
    st.markdown("### Primary Model Metrics (Default Weights)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{ms['accuracy']:.4f}")
    m2.metric("Precision", f"{ms['precision']:.4f}")
    m3.metric("Recall",    f"{ms['recall']:.4f}")
    m4.metric("F1-Score",  f"{ms['f1']:.4f}")
    m5.metric("ROC-AUC",   f"{ms['roc_auc']:.4f}")

    st.info(
        "ℹ️ **Interpretation:** The default-weight model achieves 91.4% accuracy by predicting "
        "nearly all claims as 'Paid' (matching the majority class distribution). "
        "**ROC-AUC of 0.58** is the meaningful discriminative metric — it is modestly above "
        "random (0.50) but confirms that the selected pre-outcome features have limited "
        "predictive power in this synthetic dataset. This is expected and documented."
    )

    st.markdown("### Both Model Variants")
    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True)
        add_download(metrics_df, "⬇️ Download Model Metrics", "model_metrics.csv")

    st.divider()
    st.markdown("### Confusion Matrices")
    p = OUT / "confusion_matrix.png"
    if p.exists():
        st.image(str(p), use_container_width=True)

    st.markdown("#### Default Weights Confusion Matrix (raw numbers)")
    cm_def = ms["confusion_matrix_default"]
    cm_df  = pd.DataFrame(
        cm_def,
        index=["Actual: Denied", "Actual: Paid"],
        columns=["Predicted: Denied", "Predicted: Paid"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### Balanced Weights Confusion Matrix (raw numbers)")
    cm_bal = ms["confusion_matrix_balanced"]
    cm_df2 = pd.DataFrame(
        cm_bal,
        index=["Actual: Denied", "Actual: Paid"],
        columns=["Predicted: Denied", "Predicted: Paid"],
    )
    st.dataframe(cm_df2, use_container_width=True)

    st.divider()
    st.markdown("### Classification Report (Primary Model)")
    if cr_df is not None:
        st.dataframe(cr_df.round(4), use_container_width=True)

    st.divider()
    st.markdown("### Class Imbalance Note")
    st.warning(ms["class_imbalance_note"])
    st.markdown("**Model Note:** " + ms["model_note"])

    st.divider()
    st.markdown("### Prediction Results")
    if predictions_df is not None:
        total_preds  = len(predictions_df)
        correct      = (predictions_df["actual_label"] == predictions_df["predicted_label"]).sum()
        denied_actual = (predictions_df["actual_label"] == "Denied").sum()

        p1, p2, p3 = st.columns(3)
        p1.metric("Test Set Size", f"{total_preds:,}")
        p2.metric("Correct Predictions", f"{correct:,} ({correct/total_preds*100:.1f}%)")
        p3.metric("Denied Cases in Test", f"{denied_actual:,}")

        st.markdown("#### Risk Level Distribution")
        rl_counts = predictions_df["risk_level"].value_counts().reset_index()
        rl_counts.columns = ["Risk Level", "Count"]
        rl_counts["% of Test"] = (rl_counts["Count"] / total_preds * 100).round(2)
        st.dataframe(rl_counts, use_container_width=True)

        st.markdown("#### Prediction Table (sorted by denial risk score)")
        show_cols = [
            "insurance_provider", "payment_method", "billed_amount",
            "actual_label", "predicted_label",
            "prediction_probability", "denial_risk_score", "risk_level",
        ]
        show_cols = [c for c in show_cols if c in predictions_df.columns]
        st.dataframe(
            predictions_df[show_cols]
            .sort_values("denial_risk_score", ascending=False)
            .head(200),
            use_container_width=True,
        )
        add_download(predictions_df, "⬇️ Download Predictions", "model_predictions.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6 — AI-DRIVEN INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "💡 AI-Driven Insights":
    st.subheader("AI-Driven Healthcare & Insurance Insights")

    st.write(
        "The following insights were **automatically generated** from the actual dataset "
        "by the analytics pipeline. They describe data patterns — they do not establish "
        "causation or provide medical advice."
    )

    if ai_insights_df is None:
        st.error("Run pipeline.py to generate AI insights.")
        st.stop()

    categories = ai_insights_df["Category"].unique().tolist()

    for cat in categories:
        subset = ai_insights_df[ai_insights_df["Category"] == cat]
        with st.expander(f"**{cat}** ({len(subset)} items)", expanded=(cat == "Observations")):
            for _, row in subset.iterrows():
                st.write(f"• {row['Insight']}")

    st.divider()
    st.markdown("### Interpretation Guardrails")
    st.warning(
        "These observations describe the available data. They do not establish "
        "that one healthcare, insurance, demographic, or clinical variable causes another. "
        "The dataset is synthetic; patterns may not reflect any real-world network."
    )

    st.divider()
    st.markdown("### Key Operational Questions Addressed")
    questions = [
        ("Which insurance categories have the largest number of claims?",
         "Claims are distributed roughly equally across 7 providers — consistent with synthetic data generation."),
        ("What is the overall claim denial rate?",
         "8.57% of claims (5,998 of 70,000) were denied in Q1 2025."),
        ("What are the most common denial reasons?",
         "Duplicate Claim (472), followed by Prior Authorization Required and Claim Billed to Wrong Payer."),
        ("How are healthcare costs distributed?",
         "Right-skewed: mean $1,612.87 vs median $1,214.11. A small number of high-cost claims drive the mean."),
        ("What is the revenue gap?",
         f"Total billed: $112.9M; Total paid: $72.8M — a gap of ~$40.1M across Q1 2025."),
    ]
    for q, a in questions:
        with st.expander(f"❓ {q}"):
            st.write(f"**Finding:** {a}")

    st.divider()
    add_download(ai_insights_df, "⬇️ Download AI Insights", "ai_insights.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 7 — DATA DICTIONARY
# ═══════════════════════════════════════════════════════════════════════
elif page == "🧾 Data Dictionary":
    st.subheader("Data Dictionary")

    st.markdown("### Column Definitions")
    if data_dict_df is not None:
        st.dataframe(data_dict_df, use_container_width=True)
        add_download(data_dict_df, "⬇️ Download Data Dictionary", "data_dictionary.csv")
    else:
        if claims_raw is not None:
            rows = []
            descs = {
                "billing_id":          "Unique billing record identifier",
                "patient_id":          "Patient identifier — links to patients table",
                "encounter_id":        "Encounter identifier — links to encounters table",
                "insurance_provider":  "Name of the insurance payer",
                "payment_method":      "How the bill was paid (Insurance / Self-pay)",
                "claim_id":            "Insurance claim identifier",
                "claim_billing_date":  "Date the claim was billed (DD-MM-YYYY HH:MM)",
                "billed_amount":       "Total amount billed in USD",
                "paid_amount":         "Amount actually paid in USD",
                "claim_status":        "Outcome of claim: Paid or Denied",
                "denial_reason":       "Reason for denial (null when Paid — post-outcome field)",
            }
            for col in claims_raw.columns:
                rows.append({
                    "Column": col,
                    "Data Type": str(claims_raw[col].dtype),
                    "Description": descs.get(col, ""),
                    "Missing Count": int(claims_raw[col].isna().sum()),
                    "Unique Values": int(claims_raw[col].nunique()),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.markdown("### Feature Dictionary (Engineered Features)")
    if feat_df is not None:
        st.dataframe(feat_df, use_container_width=True)
        add_download(feat_df, "⬇️ Download Feature Dictionary", "feature_dictionary.csv")

    st.divider()
    st.markdown("### Sample Data")
    if claims_raw is not None:
        st.dataframe(claims_raw.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 8 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Quality":
    st.subheader("Data Quality Report")

    if claims_raw is not None:
        cb = claims_raw
        st.markdown("### Overall Quality KPIs")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Rows",          f"{len(cb):,}")
        q2.metric("Total Columns",       f"{len(cb.columns)}")
        q3.metric("Duplicate Rows",      f"{cb.duplicated().sum():,}")
        q4.metric("Total Missing Cells", f"{cb.isna().sum().sum():,}")

        st.markdown("### Missing Values by Column")
        miss = cb.isna().sum().reset_index()
        miss.columns = ["Column", "Missing Count"]
        miss["Missing %"] = (miss["Missing Count"] / len(cb) * 100).round(2)
        miss["Notes"] = miss["Column"].map({
            "denial_reason": "Expected — only populated for Denied claims",
        }).fillna("")
        st.dataframe(miss, use_container_width=True)

        st.markdown("### Data Type Summary")
        dtypes = cb.dtypes.reset_index()
        dtypes.columns = ["Column", "Data Type"]
        st.dataframe(dtypes, use_container_width=True)

    st.divider()
    st.markdown("### Cleaning Pipeline Report")
    if dq_df is not None:
        st.dataframe(dq_df, use_container_width=True)
        add_download(dq_df, "⬇️ Download Data Quality Report", "data_quality_report.csv")
    else:
        st.info("Run pipeline.py to generate the data quality report.")

    st.divider()
    st.markdown("### Duplicate Analysis")
    if claims_raw is not None:
        dupe_count = claims_raw.duplicated().sum()
        st.metric("Duplicate Rows", dupe_count)
        if dupe_count == 0:
            st.success("✅ No duplicate rows detected in claims_and_billing.csv")

    st.divider()
    st.markdown("### Categorical Value Cardinality")
    if claims_raw is not None:
        cat_cols = claims_raw.select_dtypes(include="object").columns
        card_data = []
        for c in cat_cols:
            card_data.append({
                "Column": c,
                "Unique Values": claims_raw[c].nunique(),
                "Top Value": str(claims_raw[c].mode().iloc[0]) if not claims_raw[c].isna().all() else "N/A",
                "Top Value Count": int(claims_raw[c].value_counts().iloc[0]) if not claims_raw[c].isna().all() else 0,
            })
        st.dataframe(pd.DataFrame(card_data), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 9 — EXPORT
# ═══════════════════════════════════════════════════════════════════════
elif page == "⬇️ Export":
    st.subheader("Export Project Outputs")

    st.write(
        "Download all generated outputs for use in your IBM SkillsBuild report and presentation. "
        "The original Kaggle CSV files are never overwritten by this dashboard."
    )

    sections = [
        ("📋 Dataset Profile",         load_output("dataset_profile.csv"),        "dataset_profile.csv"),
        ("📖 Data Dictionary",          load_output("data_dictionary.csv"),         "data_dictionary.csv"),
        ("🔍 Data Quality Report",      load_output("data_quality_report.csv"),     "data_quality_report.csv"),
        ("🔧 Feature Dictionary",       load_output("feature_dictionary.csv"),      "feature_dictionary.csv"),
        ("🔒 ML Feature Selection",     load_output("ml_feature_selection.csv"),    "ml_feature_selection.csv"),
        ("📊 EDA Insights",             load_output("eda_insights.csv"),            "eda_insights.csv"),
        ("💡 AI Insights",              load_output("ai_insights.csv"),             "ai_insights.csv"),
        ("📈 Model Metrics",            load_output("model_metrics.csv"),           "model_metrics.csv"),
        ("📋 Classification Report",    load_output("classification_report.csv"),   "classification_report.csv"),
        ("🎯 Model Predictions",        load_predictions(),                          "model_predictions.csv"),
    ]

    for label, df, filename in sections:
        with st.expander(label):
            if df is not None and not df.empty:
                st.dataframe(df.head(20), use_container_width=True)
                add_download(df, f"⬇️ Download {label}", filename)
            else:
                st.warning(f"Not available yet. Run pipeline.py to generate {filename}.")

    st.divider()
    st.markdown("### EDA Charts")
    chart_files = sorted(CHARTS.glob("*.png")) if CHARTS.exists() else []
    if chart_files:
        cols = st.columns(3)
        for i, cf in enumerate(chart_files):
            with cols[i % 3]:
                st.image(str(cf), caption=cf.stem.replace("_", " "), use_container_width=True)
    else:
        st.info("No charts found. Run pipeline.py first.")

    st.divider()
    st.success(
        "✅ All outputs generated by pipeline.py are ready for inclusion in your "
        "IBM SkillsBuild internship report and presentation."
    )


# ═══════════════════════════════════════════════════════════════════════
# PAGE — ABOUT / PROJECT OWNER
# ═══════════════════════════════════════════════════════════════════════
elif page == "👤 About / Project Owner":
    st.subheader("About This Project")

    # ── Project Owner Card ──
    st.markdown("---")
    oc1, oc2 = st.columns([1, 2])
    with oc1:
        st.markdown(
            """
<div style='background:#1f2328;color:#fff;border-radius:12px;padding:24px 20px;text-align:center'>
<div style='font-size:48px;margin-bottom:8px'>👨‍💻</div>
<div style='font-size:18px;font-weight:800;letter-spacing:.5px'>PAPPU RAMESH BARAF</div>
<div style='font-size:12px;color:#9ca3af;margin-top:4px'>Software Developer | AI/ML | GenAI | Data Analytics</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with oc2:
        st.markdown(f"## {OWNER['name']}")
        st.markdown(f"**{OWNER['title']}**")
        st.markdown(f"📍 {OWNER['location']}")
        st.markdown(f"✉️ [{OWNER['email']}](mailto:{OWNER['email']})")
        st.markdown(
            f"[🔗 LinkedIn]({OWNER['linkedin']})   |   "
            f"[🐙 GitHub]({OWNER['github']})"
        )

    st.markdown("---")

    st.markdown("### Professional Summary")
    st.write(
        "Final-year B.E. Computer Science & Engineering (AI/ML) student at "
        "**Finolex Academy of Management & Technology (FAMT), Ratnagiri**, with "
        "hands-on experience in software development, Python, JavaScript, REST APIs, "
        "React, FastAPI, databases, AWS, and AI-powered application development. "
        "Experienced in building end-to-end solutions integrating **Generative AI, "
        "Machine Learning, NLP, cloud services, and data analytics**. "
        "Completed the **Lenovo LEAP NextGen Scholar Program** and "
        "**IBM SkillsBuild GenAI & Cloud Computing Internship** through Bharat-Cares and AICTE. "
        "Strong foundation in programming, problem-solving, API development, full-stack "
        "architecture, and cloud deployment. Seeking entry-level Software Developer opportunities."
    )

    st.markdown("### Education")
    st.markdown(
        "| Institution | Degree | Period |\n"
        "|-------------|--------|--------|\n"
        "| **Finolex Academy of Management & Technology (FAMT), Ratnagiri** | "
        "B.E. Computer Science & Engineering — Artificial Intelligence & Machine Learning | "
        "2023–2027 |"
    )

    st.markdown("### Skills & Technologies")
    skill_cols = st.columns(4)
    skill_groups = {
        "Languages": ["Python", "JavaScript", "SQL", "HTML/CSS"],
        "Frameworks": ["React", "FastAPI", "REST APIs", "Node.js"],
        "AI / ML": ["Machine Learning", "Generative AI", "NLP", "Data Analytics"],
        "Cloud / Tools": ["AWS", "GitHub", "Streamlit", "Pandas / scikit-learn"],
    }
    for col, (group, skills) in zip(skill_cols, skill_groups.items()):
        with col:
            st.markdown(f"**{group}**")
            for s in skills:
                st.markdown(f"• {s}")

    st.markdown("### Certifications & Programs")
    st.markdown(
        "- 🏅 **IBM SkillsBuild GenAI & Cloud Computing Internship** — Bharat-Cares / AICTE\n"
        "- 🏅 **IBM SkillsBuild Data Analytics with AI Internship** *(this project)*\n"
        "- 🏅 **Lenovo LEAP NextGen Scholar Program**\n"
    )

    st.markdown("---")
    st.markdown("### About This Project")

    pa1, pa2 = st.columns(2)
    with pa1:
        st.markdown(
            "**Project:** AI-Driven Healthcare & Insurance Analytics\n\n"
            "**Dataset:** CA Hospital Dataset Q1 2025 (Synthetic)\n\n"
            "**Internship:** IBM SkillsBuild Data Analytics with AI\n\n"
            "**Domain:** Healthcare & Insurance Analytics"
        )
    with pa2:
        st.markdown(
            "**Pipeline:** 12-phase analytics pipeline (`pipeline.py`)\n\n"
            "**Dashboard:** 10-page interactive Streamlit app (`app.py`)\n\n"
            "**ML Model:** Logistic Regression — Claim Denial Prediction\n\n"
            "**Outputs:** 10 charts, 12 CSVs, predictions, final report & PPT"
        )

    st.info(
        "This project was built as part of the IBM SkillsBuild Data Analytics with AI internship. "
        "The dataset is **synthetic** — for educational and analytical purposes only. "
        "No medical advice is provided. No real patient data was used."
    )


# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
st.divider()
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    st.caption(
        "IBM SkillsBuild Data Analytics with AI Internship Project  •  "
        "AI-Driven Healthcare & Insurance Analytics  •  "
        "CA Hospital Dataset Q1 2025 (Synthetic)  •  "
        "Educational use only — not medical advice"
    )
with col_f2:
    st.caption(
        f"Project Owner: **{OWNER['name']}**  •  "
        f"{OWNER['college']}  •  {OWNER['email']}"
    )
