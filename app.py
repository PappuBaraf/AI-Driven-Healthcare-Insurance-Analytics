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
from io import BytesIO
import json
import os
import re
import unicodedata
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pypdf import PdfReader
from streamlit.errors import StreamlitSecretNotFoundError

from groq_service import GroqAPIError, request_groq_explanation

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
    page_title="Healthcare Claims Intelligence | Pappu Ramesh Baraf",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --ink: #172b4d;
  --muted: #5b6b82;
  --brand: #126b72;
  --brand-dark: #0b4b59;
  --surface: #ffffff;
  --canvas: #f4f7fb;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--canvas); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(244,247,251,.88); }
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1440px;
  padding: clamp(1rem, 3vw, 2.5rem) clamp(.85rem, 3vw, 2.75rem) 3rem;
}
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); letter-spacing: -.025em; }
h1 { font-size: clamp(1.7rem, 4vw, 2.65rem) !important; }
h2, h3 { font-size: clamp(1.15rem, 2.4vw, 1.55rem) !important; }
[data-testid="stMetric"] {
  min-height: 108px;
  padding: 16px 18px;
  border: 1px solid #e2eaf1;
  border-radius: 14px;
  background: var(--surface);
  box-shadow: 0 4px 16px rgba(17, 46, 70, .045);
}
[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--brand-dark); font-family: 'Manrope', sans-serif; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid #e2eaf1;
  border-radius: 12px;
  overflow: hidden;
}
[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e4ebf2; }
div.stButton > button, div.stDownloadButton > button, button[kind="primary"] {
  min-height: 2.65rem;
  border-radius: 9px;
  font-weight: 650;
  transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 5px 14px rgba(18,107,114,.15);
}
@media (max-width: 700px) {
  [data-testid="stAppViewContainer"] .main .block-container { padding: 1rem .8rem 2rem; }
  [data-testid="stMetric"] { min-height: 90px; padding: 12px; }
  [data-testid="stMetricValue"] { font-size: 1.28rem; }
  [data-testid="stHorizontalBlock"] { gap: .55rem; }
  [data-testid="stTabs"] button[role="tab"] { padding: .45rem .55rem; font-size: .82rem; }
  [data-testid="stDataFrame"] { font-size: .82rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
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
            on_click=record_activity,
            args=("download_csv", {"artifact": filename}),
        )


def record_activity(action, details=None):
    """Record privacy-safe events in this visitor's current Streamlit session."""
    events = st.session_state.setdefault("activity_log", [])
    events.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "details": details or {},
    })
    del events[:-100]


def normalize_audit_text(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_document_text(uploaded_file, max_bytes=5_000_000):
    content = uploaded_file.getvalue()
    if len(content) > max_bytes:
        raise ValueError(f"{uploaded_file.name} exceeds the {max_bytes // 1_000_000} MB limit.")
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError(
                "No selectable text was found in this PDF. Scanned-image OCR is not enabled."
            )
    elif suffix == ".txt":
        text = content.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("Choose a PDF or plain-text document.")
    return text[:100_000]


def get_groq_api_key():
    environment_key = os.environ.get("GROQ_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def detect_bill_columns(columns, candidates):
    lowered = {str(column).lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for normalized, original in lowered.items():
        if any(candidate in normalized for candidate in candidates):
            return original
    return None


def parse_numeric_column(values):
    cleaned = values.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def make_policy_matches(policy_text, descriptions, limit=8):
    matches = []
    policy_lower = policy_text.lower()
    common_words = {
        "about", "after", "before", "claim", "charge", "cost", "from",
        "hospital", "medical", "other", "patient", "service", "under",
        "with", "your",
    }
    for description in descriptions:
        terms = [
            term for term in normalize_audit_text(description).split()
            if len(term) >= 4 and term not in common_words
        ]
        if not terms:
            continue
        matched_term = next((term for term in terms if term in policy_lower), None)
        if matched_term:
            position = policy_lower.find(matched_term)
            start = max(0, position - 180)
            end = min(len(policy_text), position + len(matched_term) + 220)
            matches.append({
                "Bill item": str(description)[:120],
                "Matched term": matched_term,
                "Policy excerpt": " ".join(policy_text[start:end].split()),
            })
        if len(matches) >= limit:
            break
    return matches


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

if "activity_log" not in st.session_state:
    st.session_state["activity_log"] = []
    record_activity("session_started")

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("Healthcare Claims Intelligence")
st.caption(
    "Claims, billing and payer performance analytics · IBM SkillsBuild Data Analytics with AI"
)

page_names = [
    "Patient & Provider Hub",
    "Executive Overview",
    "Claims & Insurance",
    "Interactive Explorer",
    "Exploratory Data Analysis",
    "Predictive Analytics",
    "Model Results",
    "AI-Driven Insights",
    "Data Dictionary",
    "Data Quality",
    "Experiments & Testing",
    "Load Data",
    "Exports & Activity",
    "About the Project",
]

if st.session_state.get("last_page") != st.session_state.get("app_page"):
    page_for_event = st.session_state.get("app_page", page_names[0])
    record_activity("page_view", {"page": page_for_event})
    st.session_state["last_page"] = page_for_event

if not PIPELINE_RUN:
    st.error(
        "Pipeline outputs not found. Run `python pipeline.py` from the project folder "
        "to generate all charts, metrics and predictions before using this dashboard."
    )

st.info(
    "This dashboard uses the **CA Hospital Dataset – Q1 2025** (synthetic data). "
    "All patterns shown are analytical observations. "
    "No medical diagnoses or causal conclusions are claimed."
)

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("Explore the analysis")
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

    page = st.radio(
        "Dashboard section",
        page_names,
        key="app_page",
    )

    st.divider()
    if claims_raw is not None:
        st.metric("Dataset Rows", f"{len(claims_raw):,}")
        st.metric("Tables Loaded", len(tables))
    st.caption("Synthetic data · educational use only")

    # ── Project Owner Card ──
    st.divider()
    st.markdown("**Project owner**")
    st.markdown(f"**{OWNER['name']}**")
    st.markdown(f"*{OWNER['title']}*")
    st.markdown(OWNER["location"])
    st.markdown(OWNER["email"])
    st.markdown(
        f"[LinkedIn]({OWNER['linkedin']})  ·  [GitHub]({OWNER['github']})"
    )
    st.caption(OWNER["degree"])


# ═══════════════════════════════════════════════════════════════════════
# PAGE 0 — PATIENT & PROVIDER HUB
# ═══════════════════════════════════════════════════════════════════════
if page == "Patient & Provider Hub":
    st.subheader("Bill & Policy Review")
    st.write(
        "Review de-identified bill line items for repeated charges, unusual amounts, "
        "and arithmetic mismatches. Search an insurance policy for related wording "
        "to help you prepare questions for the provider or insurer."
    )
    st.warning(
        "Privacy and safety: this is an educational review tool, not an insurer, "
        "medical service, fraud detector, or coverage decision. Use synthetic or "
        "de-identified documents only. Do not upload real patient records, names, "
        "dates of birth, member IDs, account numbers, or other protected health information. "
        "Files are processed by this hosted app; do not treat the public demo as a "
        "private or compliant health-record system."
    )

    with st.expander("How this review works", expanded=True):
        st.markdown(
            """
            1. Upload a de-identified bill as a CSV with one charge per row. A PDF or
               text bill can also be reviewed for repeated text lines when its text is selectable.
            2. Optionally upload a policy PDF or text file. The app finds keyword
               excerpts; it does **not** decide whether a service is covered.
            3. Review the checks, download the findings, and verify any question with
               the billing office or insurer before taking action.

            No AI provider or API key is required. No external AI service is called.
            """
        )

    consent = st.checkbox(
        "I will only upload synthetic or de-identified documents and understand this is not a coverage or fraud determination.",
        key="audit_privacy_confirmed",
    )
    if not consent:
        st.info("Confirm the privacy notice above to enable document review.")
    else:
        bill_file = st.file_uploader(
            "Bill line items (CSV recommended; PDF/TXT supported for text review)",
            type=["csv", "pdf", "txt"],
            key="bill_audit_upload",
            help="CSV: one charge per row. Up to 10 MB and 20,000 rows. Documents: up to 5 MB.",
        )
        policy_file = st.file_uploader(
            "Optional insurance policy (PDF or TXT)",
            type=["pdf", "txt"],
            key="policy_audit_upload",
            help="Only policy word-search and short matching excerpts are shown. Up to 5 MB.",
        )

        policy_text = ""
        bill_context = ""
        if policy_file is not None:
            try:
                policy_text = extract_document_text(policy_file)
                st.success(f"Policy text extracted: {len(policy_text):,} characters.")
            except (ValueError, OSError) as exc:
                st.error(f"Could not read policy document: {exc}")
            except Exception as exc:
                st.error(f"Policy document parsing failed: {exc}")

        if bill_file is not None:
            suffix = Path(bill_file.name).suffix.lower()
            try:
                if suffix == ".csv":
                    bill_bytes = bill_file.getvalue()
                    if len(bill_bytes) > 10_000_000:
                        raise ValueError("CSV exceeds the 10 MB limit.")
                    bill_df = pd.read_csv(BytesIO(bill_bytes), low_memory=False)
                    if bill_df.empty:
                        raise ValueError("The CSV contains no bill rows.")
                    if len(bill_df) > 20_000:
                        raise ValueError("CSV exceeds the 20,000-row review limit.")
                    if len(bill_df.columns) > 100:
                        raise ValueError("CSV exceeds the 100-column review limit.")

                    st.markdown("#### Map bill columns")
                    columns = bill_df.columns.tolist()
                    optional = ["(none)"] + columns
                    amount_guess = detect_bill_columns(
                        columns, ["line_amount", "line_total", "amount", "charge", "total"]
                    )
                    description_guess = detect_bill_columns(
                        columns, ["description", "service", "procedure", "item", "charge_description"]
                    )
                    code_guess = detect_bill_columns(
                        columns, ["cpt_code", "hcpcs_code", "procedure_code", "code"]
                    )
                    quantity_guess = detect_bill_columns(columns, ["quantity", "units", "unit_count"])
                    price_guess = detect_bill_columns(
                        columns, ["unit_price", "price_per_unit", "rate", "unit_cost"]
                    )
                    mapping_cols = st.columns(2)
                    with mapping_cols[0]:
                        description_col = st.selectbox(
                            "Service / item description",
                            optional,
                            index=optional.index(description_guess) if description_guess else 0,
                            key="bill_description_col",
                        )
                        amount_col = st.selectbox(
                            "Line charge amount",
                            columns,
                            index=columns.index(amount_guess) if amount_guess else 0,
                            key="bill_amount_col",
                        )
                        code_col = st.selectbox(
                            "Optional billing code",
                            optional,
                            index=optional.index(code_guess) if code_guess else 0,
                            key="bill_code_col",
                        )
                    with mapping_cols[1]:
                        quantity_col = st.selectbox(
                            "Optional quantity / units",
                            optional,
                            index=optional.index(quantity_guess) if quantity_guess else 0,
                            key="bill_quantity_col",
                        )
                        unit_price_col = st.selectbox(
                            "Optional price per unit",
                            optional,
                            index=optional.index(price_guess) if price_guess else 0,
                            key="bill_unit_price_col",
                        )

                    if amount_guess is None:
                        st.warning(
                            "No familiar charge column name was detected. Verify that "
                            "the selected line-charge column contains numeric amounts."
                        )
                    st.caption("Column mapping is used only for this review; the uploaded file is not modified.")
                    context_columns = [
                        column for column in (
                            description_col, amount_col, code_col, quantity_col, unit_price_col
                        )
                        if column != "(none)"
                    ]
                    bill_context = bill_df[context_columns].head(100).to_csv(index=False)[:12_000]
                    amounts = parse_numeric_column(bill_df[amount_col])
                    if not amounts.notna().any():
                        st.error(
                            "The selected line-charge column has no readable numeric values. "
                            "Choose a column containing amounts, such as 125.00 or $1,250.00."
                        )
                        st.stop()
                    if amounts.isna().any():
                        st.warning(
                            f"{int(amounts.isna().sum()):,} rows have an unreadable line charge "
                            "and will be excluded from amount checks."
                        )
                    descriptions = (
                        bill_df[description_col].fillna("").astype(str)
                        if description_col != "(none)"
                        else pd.Series(["Unlabeled charge"] * len(bill_df), index=bill_df.index)
                    )
                    normalized = descriptions.map(normalize_audit_text)
                    duplicate_keys = pd.DataFrame({
                        "normalized_item": normalized,
                        "amount": amounts.round(2),
                    })
                    if code_col != "(none)":
                        duplicate_keys["code"] = bill_df[code_col].fillna("").astype(str).str.strip().str.lower()
                    duplicate_mask = duplicate_keys.duplicated(keep=False) & normalized.ne("") & amounts.notna()
                    duplicate_rows = bill_df.loc[duplicate_mask].copy()
                    duplicate_rows.insert(0, "Source row", duplicate_rows.index + 2)
                    duplicate_rows["Review note"] = "Repeated item and charge; confirm these are separate services."
                    duplicate_total = (
                        duplicate_keys.loc[duplicate_mask]
                        .groupby(list(duplicate_keys.columns), dropna=False)["amount"]
                        .apply(lambda values: values.iloc[0] * (len(values) - 1))
                        .sum()
                    ) if duplicate_mask.any() else 0.0

                    st.markdown("#### Bill review summary")
                    valid_amounts = amounts.dropna()
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Line items", f"{len(bill_df):,}")
                    metric_cols[1].metric("Total listed charges", f"${valid_amounts.sum():,.2f}")
                    metric_cols[2].metric("Repeated-charge candidates", f"{int(duplicate_mask.sum()):,}")
                    metric_cols[3].metric("Possible repeated amount", f"${duplicate_total:,.2f}")

                    if duplicate_mask.any():
                        st.warning(
                            "Repeated descriptions and amounts are candidates for review, "
                            "not proof of duplicate billing. Separate services can legitimately match."
                        )
                        st.dataframe(duplicate_rows, use_container_width=True, hide_index=True)
                    else:
                        st.success("No exact repeated description-and-amount candidates were found.")

                    if len(valid_amounts) >= 4:
                        q1, q3 = valid_amounts.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        if iqr > 0:
                            high_threshold = q3 + 1.5 * iqr
                            high_rows = bill_df.loc[amounts > high_threshold].copy()
                            if not high_rows.empty:
                                high_rows.insert(0, "Source row", high_rows.index + 2)
                                high_rows["Review note"] = (
                                    f"Above this bill's statistical review threshold (${high_threshold:,.2f}); "
                                    "not compared with market prices."
                                )
                                st.markdown("#### Statistically high line amounts")
                                st.dataframe(high_rows, use_container_width=True, hide_index=True)
                            else:
                                st.success("No line amounts exceeded this bill's outlier review threshold.")
                        else:
                            st.info("Outlier review skipped because charge amounts have no interquartile spread.")
                    else:
                        st.info("Outlier review needs at least four valid line amounts.")

                    if quantity_col != "(none)" and unit_price_col != "(none)":
                        quantity = parse_numeric_column(bill_df[quantity_col])
                        unit_price = parse_numeric_column(bill_df[unit_price_col])
                        arithmetic_mask = (
                            quantity.notna() & unit_price.notna() & amounts.notna()
                            & ((quantity * unit_price - amounts).abs() > 0.01)
                        )
                        if arithmetic_mask.any():
                            arithmetic_rows = bill_df.loc[arithmetic_mask].copy()
                            arithmetic_rows.insert(0, "Source row", arithmetic_rows.index + 2)
                            arithmetic_rows["Calculated amount"] = (
                                quantity.loc[arithmetic_mask] * unit_price.loc[arithmetic_mask]
                            ).round(2)
                            st.markdown("#### Quantity × unit-price differences")
                            st.dataframe(arithmetic_rows, use_container_width=True, hide_index=True)
                        else:
                            st.success("Quantity × unit-price arithmetic matches on all comparable rows.")

                    bill_report = bill_df.copy()
                    bill_report["audit_repeated_item_and_amount"] = duplicate_mask
                    bill_report["audit_amount_outlier"] = (
                        amounts > high_threshold if len(valid_amounts) >= 4 and iqr > 0
                        else False
                    )
                    add_download(
                        bill_report,
                        "Download reviewed bill rows (CSV)",
                        "bill_review_findings.csv",
                    )
                    if st.button("Save review summary to session activity", key="save_bill_review"):
                        record_activity(
                            "bill_review",
                            {
                                "rows": int(len(bill_df)),
                                "repeated_charge_candidates": int(duplicate_mask.sum()),
                                "amount_outliers": int(
                                    (amounts > high_threshold).sum()
                                    if len(valid_amounts) >= 4 and iqr > 0 else 0
                                ),
                            },
                        )
                        st.success("Review summary saved to this session's activity.")

                    if policy_text and description_col != "(none)":
                        st.markdown("#### Policy wording related to bill items")
                        policy_matches = make_policy_matches(
                            policy_text, descriptions.dropna().unique()
                        )
                        if policy_matches:
                            st.dataframe(
                                pd.DataFrame(policy_matches),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(
                                "Keyword overlap only. A matching excerpt does not confirm coverage, "
                                "medical necessity, benefits, or reimbursement."
                            )
                        else:
                            st.info(
                                "No clear keyword excerpts found for the selected descriptions. "
                                "This does not mean the policy excludes the services."
                            )
                else:
                    extracted_bill_text = extract_document_text(bill_file)
                    bill_context = extracted_bill_text[:12_000]
                    st.info(
                        "Text was extracted from this document. PDF/text line parsing is limited: "
                        "repeated text may include page headers or other legitimate repeated wording. "
                        "For amount and arithmetic checks, export the bill line items to CSV."
                    )
                    lines = [
                        (index + 1, line.strip())
                        for index, line in enumerate(extracted_bill_text.splitlines())
                        if len(normalize_audit_text(line)) >= 16
                    ]
                    line_frame = pd.DataFrame(lines, columns=["Document line", "Extracted text"])
                    if not line_frame.empty:
                        line_frame["Normalized text"] = line_frame["Extracted text"].map(normalize_audit_text)
                        repeated = line_frame[
                            line_frame["Normalized text"].duplicated(keep=False)
                        ].drop(columns=["Normalized text"])
                        if not repeated.empty:
                            st.markdown("#### Repeated text lines to review")
                            st.dataframe(repeated, use_container_width=True, hide_index=True)
                        else:
                            st.success("No repeated document text lines were found.")
                    st.text_area(
                        "Extracted document text",
                        extracted_bill_text,
                        height=300,
                        disabled=True,
                    )
                    if policy_text:
                        matches = make_policy_matches(policy_text, [extracted_bill_text[:500]])
                        st.markdown("#### Related policy wording")
                        if matches:
                            st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
                        else:
                            st.info("No keyword excerpts matched; this is not a coverage decision.")
                    if st.button("Save document review to session activity", key="save_pdf_review"):
                        record_activity("bill_document_review", {"format": suffix[1:]})
                        st.success("Document review saved to this session's activity.")
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
                st.error(f"Could not review bill document: {exc}")
            except Exception as exc:
                st.error(f"Bill review failed: {exc}")

        st.markdown("#### Optional AI bill and policy explainer")
        st.caption(
            "Powered by Groq. Only use synthetic or de-identified content. The AI can be "
            "wrong; verify its explanation with the billing office or insurer."
        )
        groq_api_key = get_groq_api_key()
        if not groq_api_key:
            st.info(
                "AI explanations are not configured. Add GROQ_API_KEY to Streamlit Cloud "
                "Settings → Secrets or to your local .streamlit/secrets.toml file."
            )
            st.session_state.pop("ai_document_explanation", None)
        elif not bill_context and not policy_text:
            st.info("Upload a bill or policy document above to enable an AI explanation.")
            st.session_state.pop("ai_document_explanation", None)
        else:
            with st.form("groq_document_explainer"):
                ai_consent = st.checkbox(
                    "I consent to send the selected bill/policy text and my question to Groq "
                    "for processing. I have removed identifying and protected health information.",
                    key="groq_document_consent",
                )
                ai_question = st.text_area(
                    "What would you like explained?",
                    placeholder="For example: Summarize the listed charges and suggest neutral questions to ask.",
                    max_chars=2000,
                    key="groq_document_question",
                )
                ask_ai = st.form_submit_button("Ask AI to explain")

            if not ai_consent:
                st.session_state.pop("ai_document_explanation", None)

            if ask_ai:
                if not ai_consent:
                    st.error("Confirm the separate Groq data-sharing consent before continuing.")
                    st.session_state.pop("ai_document_explanation", None)
                else:
                    document_context = (
                        f"Bill excerpt:\n{bill_context[:12_000]}\n\n"
                        f"Policy excerpt:\n{policy_text[:12_000]}"
                    )
                    try:
                        with st.spinner("Requesting an explanation from Groq..."):
                            explanation = request_groq_explanation(
                                groq_api_key, ai_question, document_context
                            )
                        st.session_state["ai_document_explanation"] = explanation
                        record_activity(
                            "ai_document_explanation",
                            {
                                "provider": "groq",
                                "model": "llama-3.3-70b-versatile",
                                "context_characters": len(document_context),
                            },
                        )
                    except (ValueError, GroqAPIError) as exc:
                        st.session_state.pop("ai_document_explanation", None)
                        st.error(f"Could not generate an AI explanation: {exc}")

            if ai_consent and st.session_state.get("ai_document_explanation"):
                st.markdown("##### AI explanation")
                st.markdown(st.session_state["ai_document_explanation"])

    st.divider()
    st.markdown("### What to ask next")
    st.markdown(
        "- Ask the billing office to explain any repeated line, unit count, or charge that you cannot reconcile.\n"
        "- Ask your insurer to identify the exact policy section and benefit rules that apply.\n"
        "- Keep the bill, explanation of benefits, and policy wording together when requesting a review.\n"
        "- Do not treat statistical outliers as proof of error, fraud, or inappropriate care."
    )


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "Executive Overview":
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
    a1.markdown("**Claims & Billing**\n\n70,000 claim records covering payer mix, billing amounts, claim status and denial reasons")
    a2.markdown("**Exploratory Analysis**\n\n10 EDA charts revealing cost distributions, trends, and payer patterns")
    a3.markdown("**Predictive Model**\n\nLogistic Regression to classify claim outcome (Paid / Denied)")
    a4.markdown("**Analytics Insights**\n\nAutomatically generated operational observations and recommendations")

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
elif page == "Load Data":
    st.subheader("Load Database — Manage Data Sources")

    st.info(
        "This page lets you **upload your own CSV files** to add more tables to the dashboard, "
        "or enter individual records manually. "
        "Uploaded files are loaded into memory for this session only — the original `data/` files are not modified."
    )

    # ── Tab 1: CSV Upload  |  Tab 2: Manual Entry  |  Tab 3: Current Data ──
    tab_up, tab_manual, tab_current = st.tabs(
        ["Upload CSV files", "Manual data entry", "Current loaded data"]
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
                    st.success(f"Loaded **{uf.name}** — {len(df_up):,} rows × {len(df_up.columns)} columns")
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

                    add_download(df_up, f"Download {uf.name}", uf.name)

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

            submitted = st.form_submit_button("Add record to session")

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
            st.success("Record added to the session preview below.")
            st.markdown("#### New Record Preview")
            st.dataframe(new_df, use_container_width=True)
            add_download(new_df, "Download this record", "manual_claim_entry.csv")

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
                    add_download(df, f"Download {tname}", f"{tname}.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — CLAIMS & INSURANCE
# ═══════════════════════════════════════════════════════════════════════
elif page == "Claims & Insurance":
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
            "**Leakage note:** `denial_reason` is only populated **after** a claim is denied. "
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
    add_download(cb, "Download claims data", "claims_and_billing.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ═══════════════════════════════════════════════════════════════════════
elif page == "Interactive Explorer":
    st.subheader("Interactive claims explorer")
    st.write(
        "Filter the synthetic claims dataset, compare payer and claim outcomes, "
        "and export the exact rows used in this exploration."
    )
    if claims_raw is None:
        st.error("The claims dataset is not available.")
        st.stop()

    explorer = claims_raw.copy()
    date_col = "claim_billing_date"
    explorer[date_col] = pd.to_datetime(
        explorer[date_col], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    filter_cols = st.columns(3)
    providers = sorted(explorer["insurance_provider"].dropna().astype(str).unique())
    statuses = sorted(explorer["claim_status"].dropna().astype(str).unique())
    with filter_cols[0]:
        selected_providers = st.multiselect(
            "Insurance providers", providers, default=providers, key="explorer_providers"
        )
    with filter_cols[1]:
        selected_statuses = st.multiselect(
            "Claim outcomes", statuses, default=statuses, key="explorer_statuses"
        )
    with filter_cols[2]:
        date_values = explorer[date_col].dropna()
        if date_values.empty:
            start_date, end_date = None, None
            st.caption("No valid billing dates are available to filter.")
        else:
            start_date, end_date = st.date_input(
                "Billing date range",
                value=(date_values.min().date(), date_values.max().date()),
                min_value=date_values.min().date(),
                max_value=date_values.max().date(),
                key="explorer_dates",
            )

    filtered = explorer[
        explorer["insurance_provider"].astype(str).isin(selected_providers)
        & explorer["claim_status"].astype(str).isin(selected_statuses)
    ].copy()
    if start_date is not None and end_date is not None:
        filtered = filtered[
            filtered[date_col].isna()
            | filtered[date_col].dt.date.between(start_date, end_date)
        ]

    if st.button("Save this experiment to session activity", type="primary"):
        record_activity(
            "experiment_saved",
            {
                "rows": int(len(filtered)),
                "providers": selected_providers,
                "statuses": selected_statuses,
                "date_start": start_date.isoformat() if start_date else None,
                "date_end": end_date.isoformat() if end_date else None,
            },
        )
        st.success("Experiment settings recorded in this browser session only.")

    if filtered.empty:
        st.warning("No claims match these filters. Adjust the selections to continue.")
    else:
        total_rows = len(filtered)
        denial_rate = filtered["claim_status"].eq("Denied").mean() * 100
        billed_total = filtered["billed_amount"].sum()
        paid_total = filtered["paid_amount"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Filtered claims", f"{total_rows:,}")
        k2.metric("Denial rate", f"{denial_rate:.2f}%")
        k3.metric("Billed", f"${billed_total:,.0f}")
        k4.metric("Paid", f"${paid_total:,.0f}")

        st.markdown("### Payer comparison")
        payer_summary = filtered.groupby("insurance_provider").agg(
            Claims=("claim_status", "size"),
            Denied=("claim_status", lambda values: values.eq("Denied").sum()),
            Billed=("billed_amount", "sum"),
            Paid=("paid_amount", "sum"),
        )
        payer_summary["Denial rate (%)"] = (
            payer_summary["Denied"] / payer_summary["Claims"] * 100
        ).round(2)
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**Claims by payer**")
            st.bar_chart(payer_summary[["Claims"]], use_container_width=True)
        with chart_right:
            st.markdown("**Denial rate by payer**")
            st.bar_chart(payer_summary[["Denial rate (%)"]], use_container_width=True)

        st.markdown("### Filtered payer metrics")
        st.dataframe(payer_summary.round(2), use_container_width=True)
        add_download(filtered, "Download filtered claims (CSV)", "filtered_claims.csv")
        add_download(
            payer_summary.reset_index(),
            "Download payer comparison (CSV)",
            "payer_comparison.csv",
        )
        st.markdown("### Matching records")
        st.dataframe(
            filtered.drop(columns=[date_col]).head(500),
            use_container_width=True,
            height=420,
        )
        st.caption(
            "The table is capped at 500 preview rows; the CSV download includes every matching row."
        )


elif page == "Exploratory Data Analysis":
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
            with st.expander(f"{row['Section']} — {row['Question']}"):
                st.write(f"**Observation:** {row['Observation']}")
                st.write(f"**Possible Insight:** {row['Possible Insight']}")
                st.info(f"Limitation: {row['Limitation']}")
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
elif page == "Predictive Analytics":
    st.subheader("Predictive Analytics — Claim Denial Prediction")

    st.markdown(
        "The pipeline trained a **Logistic Regression** model to predict `claim_status` "
        "(Paid / Denied) using pre-outcome features only."
    )

    st.markdown("### ML Problem Definition")
    with st.expander("Target variable", expanded=True):
        st.markdown(
            "- **Target:** `claim_status` (binary: `Paid` or `Denied`)\n"
            "- **Class distribution:** 64,002 Paid (91.4%) / 5,998 Denied (8.6%)\n"
            "- **Imbalance:** Yes — the dataset is highly skewed toward Paid claims\n"
            "- **Primary metric:** ROC-AUC (more informative than accuracy for imbalanced data)"
        )

    with st.expander("Leakage prevention"):
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

    with st.expander("Features used"):
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
        add_download(ml_feat_df, "Download feature selection", "ml_feature_selection.csv")

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
elif page == "Model Results":
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
        "**Interpretation:** The default-weight model achieves 91.4% accuracy by predicting "
        "nearly all claims as 'Paid' (matching the majority class distribution). "
        "**ROC-AUC of 0.58** is the meaningful discriminative metric — it is modestly above "
        "random (0.50) but confirms that the selected pre-outcome features have limited "
        "predictive power in this synthetic dataset. This is expected and documented."
    )

    st.markdown("### Both Model Variants")
    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True)
        add_download(metrics_df, "Download model metrics", "model_metrics.csv")

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
        add_download(predictions_df, "Download predictions", "model_predictions.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6 — AI-DRIVEN INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "AI-Driven Insights":
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
    add_download(ai_insights_df, "Download analytics insights", "ai_insights.csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 7 — DATA DICTIONARY
# ═══════════════════════════════════════════════════════════════════════
elif page == "Data Dictionary":
    st.subheader("Data Dictionary")

    st.markdown("### Column Definitions")
    if data_dict_df is not None:
        st.dataframe(data_dict_df, use_container_width=True)
        add_download(data_dict_df, "Download data dictionary", "data_dictionary.csv")
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
        add_download(feat_df, "Download feature dictionary", "feature_dictionary.csv")

    st.divider()
    st.markdown("### Sample Data")
    if claims_raw is not None:
        st.dataframe(claims_raw.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 8 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════
elif page == "Data Quality":
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
        add_download(dq_df, "Download data quality report", "data_quality_report.csv")
    else:
        st.info("Run pipeline.py to generate the data quality report.")

    st.divider()
    st.markdown("### Duplicate Analysis")
    if claims_raw is not None:
        dupe_count = claims_raw.duplicated().sum()
        st.metric("Duplicate Rows", dupe_count)
        if dupe_count == 0:
            st.success("No duplicate rows detected in claims_and_billing.csv.")

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
elif page == "Experiments & Testing":
    st.subheader("Data checks and model validation")
    st.write(
        "Run transparent checks on the loaded dataset and inspect the saved model "
        "evaluation. These checks validate data structure and metrics; they do not "
        "establish clinical validity or support real-world claim decisions."
    )

    if claims_raw is None:
        st.error("The claims dataset is not available.")
        st.stop()

    required_columns = {
        "billing_id", "patient_id", "encounter_id", "insurance_provider",
        "payment_method", "claim_id", "claim_billing_date", "billed_amount",
        "paid_amount", "claim_status", "denial_reason",
    }
    missing_columns = sorted(required_columns - set(claims_raw.columns))
    valid_statuses = {"Paid", "Denied"}
    unexpected_statuses = sorted(
        set(claims_raw["claim_status"].dropna().astype(str)) - valid_statuses
    ) if "claim_status" in claims_raw else ["claim_status column missing"]
    negative_amounts = int(
        (
            claims_raw[["billed_amount", "paid_amount"]]
            .apply(pd.to_numeric, errors="coerce")
            .lt(0)
            .any(axis=1)
        ).sum()
    ) if {"billed_amount", "paid_amount"}.issubset(claims_raw.columns) else None

    checks = [
        {
            "Check": "Required analysis columns",
            "Result": "PASS" if not missing_columns else "FAIL",
            "Details": "All expected fields are present" if not missing_columns else ", ".join(missing_columns),
        },
        {
            "Check": "Claim status values",
            "Result": "PASS" if not unexpected_statuses else "FAIL",
            "Details": "Paid and Denied only" if not unexpected_statuses else ", ".join(unexpected_statuses),
        },
        {
            "Check": "Negative billed/paid amounts",
            "Result": "PASS" if negative_amounts == 0 else "WARN",
            "Details": f"{negative_amounts:,} rows" if negative_amounts is not None else "Amount columns missing",
        },
        {
            "Check": "Duplicate claim rows",
            "Result": "PASS" if not claims_raw.duplicated().any() else "WARN",
            "Details": f"{int(claims_raw.duplicated().sum()):,} duplicate rows",
        },
        {
            "Check": "Saved model evaluation",
            "Result": "PASS" if PIPELINE_RUN and model_summary.get("roc_auc") is not None else "WARN",
            "Details": f"ROC-AUC {model_summary['roc_auc']:.4f}" if PIPELINE_RUN and model_summary.get("roc_auc") is not None else "Model outputs not available",
        },
    ]
    checks_df = pd.DataFrame(checks)
    for _, check in checks_df.iterrows():
        if check["Result"] == "FAIL":
            st.error(f"{check['Check']}: {check['Details']}")
        elif check["Result"] == "WARN":
            st.warning(f"{check['Check']}: {check['Details']}")
        else:
            st.success(f"{check['Check']}: {check['Details']}")
    st.dataframe(checks_df, use_container_width=True, hide_index=True)
    add_download(checks_df, "Download validation checks (CSV)", "validation_checks.csv")

    st.markdown("### Model evaluation evidence")
    if model_summary:
        model_metrics = {
            key: model_summary.get(key)
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
            if model_summary.get(key) is not None
        }
        st.bar_chart(pd.Series(model_metrics, name="Score").to_frame())
        st.caption(
            "Scores are from the included held-out evaluation. The low ROC-AUC and "
            "class imbalance should be considered; accuracy alone is not sufficient."
        )
    else:
        st.info("Model outputs are unavailable; run `python pipeline.py` locally to generate them.")


elif page == "Exports & Activity":
    st.subheader("Downloads and session activity")

    st.write(
        "Download the source dataset, analysis results, and validation outputs. "
        "Session activity is held only in the current Streamlit session and is not "
        "persisted as a visitor log."
    )

    if claims_raw is not None:
        st.markdown("### Source dataset")
        add_download(claims_raw, "Download claims dataset (CSV)", "claims_and_billing.csv")
        st.markdown(
            "[Original Kaggle dataset page](https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025)"
        )

    sections = [
        ("Dataset profile",             load_output("dataset_profile.csv"),        "dataset_profile.csv"),
        ("Data dictionary",             load_output("data_dictionary.csv"),         "data_dictionary.csv"),
        ("Data quality report",         load_output("data_quality_report.csv"),     "data_quality_report.csv"),
        ("Feature dictionary",          load_output("feature_dictionary.csv"),      "feature_dictionary.csv"),
        ("ML feature selection",        load_output("ml_feature_selection.csv"),    "ml_feature_selection.csv"),
        ("EDA insights",                load_output("eda_insights.csv"),            "eda_insights.csv"),
        ("Analytics insights",          load_output("ai_insights.csv"),             "ai_insights.csv"),
        ("Model metrics",               load_output("model_metrics.csv"),           "model_metrics.csv"),
        ("Classification report",       load_output("classification_report.csv"),   "classification_report.csv"),
        ("Model predictions",           load_predictions(),                          "model_predictions.csv"),
    ]

    for label, df, filename in sections:
        with st.expander(label):
            if df is not None and not df.empty:
                st.dataframe(df.head(20), use_container_width=True)
                add_download(df, f"Download {label}", filename)
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
        "Analysis outputs generated by pipeline.py are ready for inclusion in your "
        "IBM SkillsBuild internship report and presentation."
    )

    st.divider()
    st.markdown("### Current-session activity log")
    st.caption(
        "Privacy note: this JSON includes only timestamps and generic page, experiment, "
        "and download actions. It excludes visitor identity, claim identifiers, and uploaded filenames. "
        "It lives in this session's memory and is cleared when the session ends."
    )
    activity = st.session_state.get("activity_log", [])
    st.dataframe(pd.DataFrame(activity), use_container_width=True, hide_index=True)
    st.download_button(
        "Download session activity (JSON)",
        data=json.dumps(activity, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="session_activity.json",
        mime="application/json",
        on_click=record_activity,
        args=("download_activity_json", {}),
    )


# ═══════════════════════════════════════════════════════════════════════
# PAGE — ABOUT / PROJECT OWNER
# ═══════════════════════════════════════════════════════════════════════
elif page == "About the Project":
    st.subheader("About This Project")

    # ── Project Owner Card ──
    st.markdown("---")
    oc1, oc2 = st.columns([1, 2])
    with oc1:
        st.markdown(
            """
<div style='background:#1f2328;color:#fff;border-radius:12px;padding:24px 20px;text-align:center'>
<div style='font-size:18px;font-weight:800;letter-spacing:.5px'>PAPPU RAMESH BARAF</div>
<div style='font-size:12px;color:#9ca3af;margin-top:4px'>Software Developer | AI/ML | GenAI | Data Analytics</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with oc2:
        st.markdown(f"## {OWNER['name']}")
        st.markdown(f"**{OWNER['title']}**")
        st.markdown(OWNER["location"])
        st.markdown(f"[{OWNER['email']}](mailto:{OWNER['email']})")
        st.markdown(
            f"[LinkedIn]({OWNER['linkedin']})   |   "
            f"[GitHub]({OWNER['github']})"
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
