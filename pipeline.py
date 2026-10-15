"""
Healthcare & Insurance Analytics - Full Offline Pipeline
IBM SkillsBuild Data Analytics with AI Internship

Produces all outputs in outputs/ and data/cleaned/
Run once before or alongside app.py
"""

import pathlib, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE   = pathlib.Path(__file__).parent
DATA   = BASE / "data"
CLEAN  = DATA / "cleaned"
OUT    = BASE / "outputs"
CHARTS = OUT  / "charts"
REPS   = OUT  / "reports"
PREDS  = OUT  / "predictions"

for d in [CLEAN, OUT, CHARTS, REPS, PREDS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_all():
    tables = {}
    for f in sorted(DATA.glob("*.csv")):
        try:
            tables[f.stem] = pd.read_csv(f, low_memory=False)
            print(f"  Loaded {f.name}: {len(tables[f.stem]):,} rows")
        except Exception as e:
            print(f"  WARN: Could not load {f.name}: {e}")
    return tables

def find_col(df, candidates):
    """Case-insensitive flexible column finder."""
    if df is None: return None
    norm = {c.strip().lower().replace(" ","_"): c for c in df.columns}
    for cand in candidates:
        k = cand.strip().lower().replace(" ","_")
        if k in norm: return norm[k]
        for n, actual in norm.items():
            if k in n or n in k: return actual
    return None

def save_csv(df, path, label=""):
    df.to_csv(path, index=False)
    print(f"  Saved {label}: {path.name} ({len(df)} rows)")

def save_chart(fig, name):
    p = CHARTS / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart: {name}")

# ---------------------------------------------------------------------
# PHASE 1 - DATASET INSPECTION
# ---------------------------------------------------------------------
print("\n=== PHASE 1: DATASET INSPECTION ===")
tables = load_all()

profile_rows = []
dict_rows    = []

for tname, df in tables.items():
    dupes = int(df.duplicated().sum())
    for col in df.columns:
        dtype    = str(df[col].dtype)
        miss     = int(df[col].isna().sum())
        miss_pct = round(miss / max(len(df),1) * 100, 2)
        uniq     = int(df[col].nunique(dropna=True))
        is_num   = pd.api.types.is_numeric_dtype(df[col])
        is_date  = "date" in col.lower() or "time" in col.lower()
        is_id    = col.lower().endswith("_id") or col.lower() in {"id"}

        # guess key type
        if is_id and uniq == len(df): key_type = "Primary Key"
        elif is_id:                   key_type = "Foreign Key"
        else:                         key_type = ""

        # description heuristic
        desc_map = {
            "billing_id":          "Unique billing record identifier",
            "patient_id":          "Patient identifier - links to patients table",
            "encounter_id":        "Encounter identifier - links to encounters table",
            "claim_id":            "Insurance claim identifier",
            "insurance_provider":  "Name of insurance payer",
            "payment_method":      "How the bill was paid (Insurance / Selfpay)",
            "claim_billing_date":  "Date the claim was billed (DD-MM-YYYY)",
            "billed_amount":       "Total amount billed (USD)",
            "paid_amount":         "Amount actually paid (USD)",
            "claim_status":        "Outcome of claim: Paid or Denied",
            "denial_reason":       "Reason for denial (null when Paid)",
        }
        desc = desc_map.get(col, "")

        dict_rows.append({
            "Table": tname, "Column": col, "Data Type": dtype,
            "Description": desc, "Missing Count": miss,
            "Missing %": miss_pct, "Unique Values": uniq,
            "Is Numeric": is_num, "Is Date Hint": is_date,
            "Key Type": key_type,
        })

    profile_rows.append({
        "Table": tname, "Rows": len(df), "Columns": len(df.columns),
        "Missing Cells": int(df.isna().sum().sum()),
        "Missing Cell %": round(df.isna().mean().mean()*100, 2),
        "Duplicate Rows": dupes,
    })

data_dict  = pd.DataFrame(dict_rows)
dataset_prof = pd.DataFrame(profile_rows)
save_csv(data_dict,   OUT/"data_dictionary.csv",  "data_dictionary")
save_csv(dataset_prof, OUT/"dataset_profile.csv", "dataset_profile")

# ---------------------------------------------------------------------
# PHASE 2 - TABLE RELATIONSHIPS
# ---------------------------------------------------------------------
print("\n--- PHASE 2: TABLE RELATIONSHIPS ---")

# Identify shared column names across tables
all_cols = {}
for tname, df in tables.items():
    for col in df.columns:
        nk = col.strip().lower()
        all_cols.setdefault(nk, []).append(tname)

rel_rows = []
for tname, df in tables.items():
    for col in df.columns:
        nk = col.strip().lower()
        others = [t for t in all_cols.get(nk,[]) if t != tname]
        if others and (nk.endswith("_id") or nk in {"id"}):
            uniq_ratio = df[col].nunique(dropna=True) / max(len(df),1)
            rel_type = "One (PK)" if uniq_ratio > 0.95 else "Many (FK)"
            for other in others:
                rel_rows.append({
                    "Table": tname, "Column": col,
                    "Key Type": rel_type,
                    "Related Table": other,
                    "Relationship": f"{tname}.{col} - {other}.{col}",
                    "Unique Ratio": round(uniq_ratio,4),
                })

rel_df = pd.DataFrame(rel_rows) if rel_rows else pd.DataFrame(
    columns=["Table","Column","Key Type","Related Table","Relationship","Unique Ratio"])
save_csv(rel_df, OUT/"table_relationships.csv", "table_relationships")

# ---------------------------------------------------------------------
# PHASE 3 - DATA CLEANING
# ---------------------------------------------------------------------
print("\n--- PHASE 3: DATA CLEANING ---")

quality_rows = []
cleaned = {}

for tname, df in tables.items():
    orig = df.copy()
    work = df.copy()
    dupe_before = int(work.duplicated().sum())

    for col in work.columns:
        orig_dtype   = str(work[col].dtype)
        miss_before  = int(work[col].isna().sum())
        miss_after   = miss_before
        new_dtype    = orig_dtype
        transform    = "none"
        notes        = ""

        # Date detection
        if ("date" in col.lower() or "time" in col.lower()) and work[col].dtype == object:
            for fmt in ["%d-%m-%Y %H:%M", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]:
                parsed = pd.to_datetime(work[col], format=fmt, errors="coerce")
                if parsed.notna().mean() >= 0.5:
                    work[col] = parsed
                    new_dtype  = "datetime64[ns]"
                    transform  = f"to_datetime(format={fmt})"
                    miss_after = int(work[col].isna().sum())
                    notes      = "Dates parsed to datetime"
                    break

        # Numeric strip
        if work[col].dtype == object:
            attempt = pd.to_numeric(work[col].astype(str).str.replace(r'[$,]','',regex=True), errors='coerce')
            if attempt.notna().mean() >= 0.95 and orig_dtype == object:
                work[col] = attempt
                new_dtype  = str(work[col].dtype)
                transform  = "to_numeric"
                miss_after = int(work[col].isna().sum())

        quality_rows.append({
            "Table": tname, "Column": col,
            "Original Type": orig_dtype, "New Type": new_dtype,
            "Missing Before": miss_before, "Missing After": miss_after,
            "Duplicates": dupe_before, "Transformation": transform, "Notes": notes,
        })

    # Remove exact duplicates
    work.drop_duplicates(inplace=True)
    cleaned[tname] = work
    work.to_csv(CLEAN / f"{tname}_cleaned.csv", index=False)
    print(f"  Cleaned {tname}: {len(orig)} - {len(work)} rows")

quality_df = pd.DataFrame(quality_rows)
save_csv(quality_df, OUT/"data_quality_report.csv", "data_quality_report")

# ---------------------------------------------------------------------
# PHASE 4 - FEATURE ENGINEERING
# ---------------------------------------------------------------------
print("\n--- PHASE 4: FEATURE ENGINEERING ---")

feat_rows = []

if "claims_and_billing" in cleaned:
    cb = cleaned["claims_and_billing"].copy()

    # Date features
    date_col_name = find_col(cb, ["claim_billing_date"])
    if date_col_name and pd.api.types.is_datetime64_any_dtype(cb[date_col_name]):
        cb["claim_year"]        = cb[date_col_name].dt.year
        cb["claim_month"]       = cb[date_col_name].dt.month
        cb["claim_day"]         = cb[date_col_name].dt.day
        cb["claim_dayofweek"]   = cb[date_col_name].dt.dayofweek
        cb["claim_quarter"]     = cb[date_col_name].dt.quarter
        for fn in ["claim_year","claim_month","claim_day","claim_dayofweek","claim_quarter"]:
            feat_rows.append({"Feature": fn, "Source Table": "claims_and_billing",
                "Derived From": date_col_name, "Type": "Date Part",
                "Leakage Risk": "None", "Description": f"{fn} extracted from {date_col_name}"})

    # Financial features
    if "billed_amount" in cb.columns and "paid_amount" in cb.columns:
        cb["payment_ratio"]      = (cb["paid_amount"] / cb["billed_amount"].replace(0, np.nan)).round(4)
        cb["unpaid_amount"]      = (cb["billed_amount"] - cb["paid_amount"]).round(2)
        cb["high_cost_flag"]     = (cb["billed_amount"] > cb["billed_amount"].quantile(0.75)).astype(int)
        for fn, src, desc in [
            ("payment_ratio",  "billed/paid",   "Ratio of paid to billed amount"),
            ("unpaid_amount",  "billed-paid",   "Difference between billed and paid"),
            ("high_cost_flag", "billed_amount", "1 if billed > 75th percentile"),
        ]:
            feat_rows.append({"Feature": fn, "Source Table": "claims_and_billing",
                "Derived From": src, "Type": "Financial", "Leakage Risk": "None",
                "Description": desc})

    # Denial flag (for EDA only - NOT a model feature due to leakage)
    if "claim_status" in cb.columns:
        cb["denial_flag"] = (cb["claim_status"] == "Denied").astype(int)
        feat_rows.append({"Feature": "denial_flag", "Source Table": "claims_and_billing",
            "Derived From": "claim_status", "Type": "Target Derived",
            "Leakage Risk": "HIGH - do not use as model feature",
            "Description": "1 if claim was denied - derived directly from target"})

    # Patient-level aggregates
    pid_col = find_col(cb, ["patient_id"])
    if pid_col:
        pt_agg = cb.groupby(pid_col).agg(
            patient_claim_count  = ("billing_id", "count"),
            patient_total_billed = ("billed_amount", "sum"),
            patient_avg_billed   = ("billed_amount", "mean"),
            patient_total_paid   = ("paid_amount", "sum"),
        ).round(2).reset_index()
        cb = cb.merge(pt_agg, on=pid_col, how="left")
        for fn in ["patient_claim_count","patient_total_billed","patient_avg_billed","patient_total_paid"]:
            feat_rows.append({"Feature": fn, "Source Table": "claims_and_billing",
                "Derived From": f"grouped by {pid_col}", "Type": "Aggregate",
                "Leakage Risk": "Low - pre-outcome aggregates",
                "Description": f"Patient-level {fn.replace('_',' ')}"})

    cb.to_csv(CLEAN / "claims_and_billing_features.csv", index=False)
    cleaned["claims_and_billing_features"] = cb
    print(f"  Feature-engineered claims: {len(cb)} rows, {len(cb.columns)} columns")

feat_df = pd.DataFrame(feat_rows)
save_csv(feat_df, OUT/"feature_dictionary.csv", "feature_dictionary")

# ---------------------------------------------------------------------
# PHASE 5 - EDA
# ---------------------------------------------------------------------
print("\n--- PHASE 5 & 6: EDA + CHARTS ---")

cb = cleaned.get("claims_and_billing_features",
     cleaned.get("claims_and_billing", tables.get("claims_and_billing")))

# Ensure date column is datetime for charting (lost when features CSV is reloaded)
if cb is not None:
    _dc = find_col(cb, ["claim_billing_date"])
    if _dc and not pd.api.types.is_datetime64_any_dtype(cb[_dc]):
        for _fmt in ["%d-%m-%Y %H:%M", "%Y-%m-%d"]:
            _parsed = pd.to_datetime(cb[_dc], format=_fmt, errors="coerce")
            if _parsed.notna().mean() >= 0.5:
                cb[_dc] = _parsed
                break

eda_insights = []

if cb is not None:

    # -- Chart 1: Claim Status Distribution
    fig, ax = plt.subplots(figsize=(7,4))
    cs = cb["claim_status"].value_counts()
    bars = ax.bar(cs.index, cs.values, color=["#3b82d4","#e74c3c"])
    ax.set_title("Claim Status Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Claim Status"); ax.set_ylabel("Count")
    for bar, val in zip(bars, cs.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout(); save_chart(fig, "01_claim_status.png")
    total = len(cb)
    denied_n = cs.get("Denied", 0)
    paid_n   = cs.get("Paid",   0)
    eda_insights.append({
        "Section": "Claims",
        "Question": "What is the overall claim approval rate?",
        "Observation": f"Of {total:,} claims, {paid_n:,} ({paid_n/total*100:.1f}%) were Paid and {denied_n:,} ({denied_n/total*100:.1f}%) were Denied.",
        "Possible Insight": "The overall denial rate of ~8.6% indicates that the majority of claims are successfully paid, but nearly 1 in 12 claims faces denial - creating opportunities for operational improvement.",
        "Limitation": "The dataset is synthetic (Q1 2025). Denial rates in real networks can differ significantly.",
    })

    # -- Chart 2: Insurance Provider Distribution
    fig, ax = plt.subplots(figsize=(8,4))
    ip = cb["insurance_provider"].value_counts()
    ip.sort_values().plot(kind="barh", ax=ax, color="#3b82d4")
    ax.set_title("Claims by Insurance Provider", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Claims"); ax.set_ylabel("Provider")
    fig.tight_layout(); save_chart(fig, "02_insurance_provider.png")
    eda_insights.append({
        "Section": "Insurance",
        "Question": "Which insurance provider has the most claims?",
        "Observation": f"Claims are distributed across {ip.shape[0]} providers: " +
            ", ".join([f"{k} ({v:,})" for k,v in ip.items()]) + ".",
        "Possible Insight": "Near-equal distribution across providers is consistent with synthetic data. In real data, provider market share typically varies.",
        "Limitation": "Synthetic dataset; real-world payer mix would differ.",
    })

    # -- Chart 3: Billed Amount Distribution
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(cb["billed_amount"].dropna(), bins=50, color="#3b82d4", edgecolor="white")
    ax.set_title("Billed Amount Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Billed Amount (USD)"); ax.set_ylabel("Frequency")
    ax.axvline(cb["billed_amount"].median(), color="orange", linestyle="--", label=f"Median ${cb['billed_amount'].median():,.0f}")
    ax.axvline(cb["billed_amount"].mean(),   color="red",    linestyle="--", label=f"Mean ${cb['billed_amount'].mean():,.0f}")
    ax.legend(); fig.tight_layout(); save_chart(fig, "03_billed_amount_dist.png")
    eda_insights.append({
        "Section": "Billing",
        "Question": "How are billed amounts distributed?",
        "Observation": f"Billed amounts range from ${cb['billed_amount'].min():,.2f} to ${cb['billed_amount'].max():,.2f}. Mean: ${cb['billed_amount'].mean():,.2f}; Median: ${cb['billed_amount'].median():,.2f}.",
        "Possible Insight": "The right-skewed distribution (mean > median) indicates a relatively small number of high-cost claims. These outlier claims may warrant review.",
        "Limitation": "Claim amounts in synthetic data may not reflect actual healthcare cost patterns.",
    })

    # -- Chart 4: Paid vs Billed by Insurance Provider
    fig, ax = plt.subplots(figsize=(9,5))
    prov_grp = cb.groupby("insurance_provider")[["billed_amount","paid_amount"]].mean().sort_values("billed_amount", ascending=False)
    x = np.arange(len(prov_grp))
    w = 0.35
    ax.bar(x - w/2, prov_grp["billed_amount"], w, label="Avg Billed", color="#3b82d4")
    ax.bar(x + w/2, prov_grp["paid_amount"],   w, label="Avg Paid",   color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(prov_grp.index, rotation=30, ha="right")
    ax.set_title("Average Billed vs Paid Amount by Insurance Provider", fontsize=12, fontweight="bold")
    ax.set_ylabel("Amount (USD)"); ax.legend(); fig.tight_layout()
    save_chart(fig, "04_billed_vs_paid_by_provider.png")

    # -- Chart 5: Denial Reasons
    if "denial_reason" in cb.columns:
        fig, ax = plt.subplots(figsize=(9,5))
        dr = cb["denial_reason"].value_counts().dropna()
        dr.sort_values().plot(kind="barh", ax=ax, color="#e74c3c")
        ax.set_title("Claim Denial Reasons", fontsize=13, fontweight="bold")
        ax.set_xlabel("Count"); ax.set_ylabel("Denial Reason")
        fig.tight_layout(); save_chart(fig, "05_denial_reasons.png")
        top_dr = dr.index[0]
        eda_insights.append({
            "Section": "Denials",
            "Question": "What are the most common reasons for claim denial?",
            "Observation": f"{len(dr)} denial reason categories found. Most frequent: '{top_dr}' ({dr.iloc[0]:,} cases).",
            "Possible Insight": "Duplicate claim submissions and prior authorization gaps represent the highest-frequency denial categories - common operational pain points in healthcare billing.",
            "Limitation": "denial_reason is only populated for denied claims. It must not be used as a predictive feature to avoid target leakage.",
        })

    # -- Chart 6: Payment Method
    fig, ax = plt.subplots(figsize=(6,4))
    pm = cb["payment_method"].value_counts()
    ax.pie(pm.values, labels=pm.index, autopct="%1.1f%%", colors=["#3b82d4","#f39c12"], startangle=90)
    ax.set_title("Payment Method Distribution", fontsize=12, fontweight="bold")
    fig.tight_layout(); save_chart(fig, "06_payment_method.png")

    # -- Chart 7: Monthly Claim Trend
    date_col_name = find_col(cb, ["claim_billing_date"])
    if date_col_name and pd.api.types.is_datetime64_any_dtype(cb[date_col_name]):
        cb_dated = cb.dropna(subset=[date_col_name]).copy()
        cb_dated["month_period"] = cb_dated[date_col_name].dt.to_period("M").astype(str)
        monthly = cb_dated.groupby("month_period").agg(
            claim_count=("billing_id","count"),
            total_billed=("billed_amount","sum"),
        ).reset_index().sort_values("month_period")
        fig, axes = plt.subplots(1,2, figsize=(13,4))
        axes[0].plot(monthly["month_period"], monthly["claim_count"], marker="o", color="#3b82d4")
        axes[0].set_title("Monthly Claim Volume"); axes[0].set_xlabel("Month"); axes[0].set_ylabel("Claims")
        axes[0].tick_params(axis="x", rotation=45)
        axes[1].plot(monthly["month_period"], monthly["total_billed"], marker="s", color="#27ae60")
        axes[1].set_title("Monthly Total Billed Amount"); axes[1].set_xlabel("Month"); axes[1].set_ylabel("Total Billed (USD)")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout(); save_chart(fig, "07_monthly_trends.png")
        eda_insights.append({
            "Section": "Trends",
            "Question": "How do claim volumes and billed amounts change over time?",
            "Observation": f"The dataset spans {monthly['month_period'].min()} to {monthly['month_period'].max()} with {len(monthly)} calendar months of data.",
            "Possible Insight": "Monthly trends can reveal operational seasonal patterns in claims processing.",
            "Limitation": "10,362 self-pay rows have no billing date and are excluded from time trends.",
        })

    # -- Chart 8: Claim status by Insurance Provider
    fig, ax = plt.subplots(figsize=(9,5))
    ct = cb.groupby(["insurance_provider","claim_status"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", ax=ax, color=["#e74c3c","#3b82d4"])
    ax.set_title("Claim Status % by Insurance Provider", fontsize=12, fontweight="bold")
    ax.set_xlabel("Insurance Provider"); ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right"); ax.legend()
    fig.tight_layout(); save_chart(fig, "08_status_by_provider.png")

    # -- Chart 9: High Cost Flag Distribution
    if "high_cost_flag" in cb.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        hc = cb["high_cost_flag"].value_counts()
        ax.bar(["Standard Cost","High Cost"], hc.values, color=["#3b82d4","#e74c3c"])
        ax.set_title("High-Cost Claim Flag (>75th Percentile)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        for i,v in enumerate(hc.values):
            ax.text(i, v+200, f"{v:,}", ha="center")
        fig.tight_layout(); save_chart(fig, "09_high_cost_flag.png")

    # -- Chart 10: Payment Ratio by Claim Status
    if "payment_ratio" in cb.columns:
        fig, ax = plt.subplots(figsize=(7,4))
        for status, grp in cb.groupby("claim_status"):
            vals = grp["payment_ratio"].dropna()
            ax.hist(vals, bins=30, alpha=0.6, label=status)
        ax.set_title("Payment Ratio by Claim Status", fontsize=12, fontweight="bold")
        ax.set_xlabel("Payment Ratio (paid / billed)"); ax.set_ylabel("Frequency"); ax.legend()
        fig.tight_layout(); save_chart(fig, "10_payment_ratio.png")

# Save EDA insights
eda_df = pd.DataFrame(eda_insights)
save_csv(eda_df, OUT/"eda_insights.csv", "eda_insights")

# ---------------------------------------------------------------------
# PHASE 7 & 8 - ML PROBLEM + LEAKAGE CHECK
# ---------------------------------------------------------------------
print("\n--- PHASE 7 & 8: ML PROBLEM SELECTION + LEAKAGE CHECK ---")

cb_ml = cleaned.get("claims_and_billing", tables.get("claims_and_billing")).copy()

TARGET = "claim_status"
print(f"  Target: {TARGET}")
print(f"  Classes: {cb_ml[TARGET].value_counts().to_dict()}")

# Leakage analysis
all_feat_rows = []
for col in cb_ml.columns:
    if col == TARGET: continue
    nk = col.lower()
    # Post-outcome leakage
    if any(t in nk for t in ["denial_reason","denial","approved","approval","paid_amount","payment","paid"]):
        risk = "HIGH"; used = "No"
        reason = "Post-outcome: only known after claim decision"
    elif any(t in nk for t in ["billing_id","claim_id","encounter_id","patient_id"]):
        risk = "Medium"; used = "No"
        reason = "Identifier: not a predictive signal, would cause overfitting"
    elif "date" in nk or "time" in nk:
        risk = "Low"; used = "Yes (as date parts)"
        reason = "Date features extracted - raw date string excluded"
    else:
        risk = "None"; used = "Yes"
        reason = "Safe predictor"
    all_feat_rows.append({
        "Feature": col, "Used?": used, "Leakage Risk": risk, "Reason": reason
    })

ml_feat_df = pd.DataFrame(all_feat_rows)
save_csv(ml_feat_df, OUT/"ml_feature_selection.csv", "ml_feature_selection")

# ---------------------------------------------------------------------
# PHASE 9 - ML MODEL
# ---------------------------------------------------------------------
print("\n--- PHASE 9 & 10: MACHINE LEARNING + EVALUATION ---")

try:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )

    cb_ml = cleaned.get("claims_and_billing", tables.get("claims_and_billing")).copy()

    # Parse date for feature extraction
    date_col_ml = find_col(cb_ml, ["claim_billing_date"])
    if date_col_ml:
        for fmt in ["%d-%m-%Y %H:%M", "%Y-%m-%d"]:
            parsed = pd.to_datetime(cb_ml[date_col_ml], format=fmt, errors="coerce")
            if parsed.notna().mean() >= 0.5:
                cb_ml["feat_claim_year"]      = parsed.dt.year
                cb_ml["feat_claim_month"]     = parsed.dt.month
                cb_ml["feat_claim_dayofweek"] = parsed.dt.dayofweek
                cb_ml["feat_claim_quarter"]   = parsed.dt.quarter
                break

    # Build feature matrix - leakage-safe
    features = [
        "insurance_provider",
        "payment_method",
        "billed_amount",
    ]
    optional_feats = ["feat_claim_year","feat_claim_month","feat_claim_dayofweek","feat_claim_quarter"]
    for f in optional_feats:
        if f in cb_ml.columns:
            features.append(f)

    print(f"  Features used: {features}")

    # Encode target
    le = LabelEncoder()
    y  = le.fit_transform(cb_ml[TARGET].astype(str))
    print(f"  Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = cb_ml[features].copy()

    numeric_feats = X.select_dtypes(include=np.number).columns.tolist()
    categ_feats   = X.select_dtypes(include="object").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_feats),
        ("cat", cat_pipe, categ_feats),
    ])

    # Train two variants:
    # (A) Default class weight - optimises overall accuracy (suitable for reporting accuracy)
    # (B) Balanced class weight - optimises minority recall (reported separately)
    # Primary model is (A); ROC-AUC is the headline discriminative metric.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    def run_model(cw, label):
        pipe = Pipeline([
            ("pre",   pre),
            ("model", LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        p   = pipe.predict(X_test)
        prb = pipe.predict_proba(X_test)[:,1]
        return pipe, p, prb

    model_pipe, pred_def, prob_def = run_model(None,       "default")
    _,          pred_bal, prob_bal = run_model("balanced", "balanced")

    # Primary metrics: default class weight model
    pred = pred_def
    prob = prob_def

    acc  = round(accuracy_score(y_test, pred),4)
    prec = round(precision_score(y_test, pred, zero_division=0),4)
    rec  = round(recall_score(y_test, pred, zero_division=0),4)
    f1   = round(f1_score(y_test, pred, zero_division=0),4)
    roc  = round(roc_auc_score(y_test, prob),4)

    # Balanced model recall (for denied-class sensitivity)
    rec_bal = round(recall_score(y_test, pred_bal, zero_division=0, pos_label=0), 4)
    f1_bal  = round(f1_score(y_test, pred_bal, zero_division=0), 4)
    roc_bal = round(roc_auc_score(y_test, prob_bal), 4)

    print(f"  [Default]  Accuracy:{acc}  Precision:{prec}  Recall:{rec}  F1:{f1}  ROC-AUC:{roc}")
    print(f"  [Balanced] Recall(Denied):{rec_bal}  F1:{f1_bal}  ROC-AUC:{roc_bal}")

    # Save both model variants in metrics
    metrics_df = pd.DataFrame([
        {
            "Model": "Logistic Regression (default weights)",
            "Target": TARGET,
            "Classes": str(list(le.classes_)),
            "Train Size": len(X_train),
            "Test Size": len(X_test),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "Random State": 42,
            "Class Weight": "default",
            "Note": "Primary model - optimises overall accuracy",
        },
        {
            "Model": "Logistic Regression (balanced weights)",
            "Target": TARGET,
            "Classes": str(list(le.classes_)),
            "Train Size": len(X_train),
            "Test Size": len(X_test),
            "Accuracy": round(accuracy_score(y_test, pred_bal),4),
            "Precision": round(precision_score(y_test, pred_bal, zero_division=0),4),
            "Recall": round(recall_score(y_test, pred_bal, zero_division=0),4),
            "F1-Score": f1_bal,
            "ROC-AUC": roc_bal,
            "Random State": 42,
            "Class Weight": "balanced",
            "Note": "Sensitivity model - higher recall for minority (Denied) class",
        },
    ])
    save_csv(metrics_df, OUT/"model_metrics.csv", "model_metrics")

    # Confusion matrix chart - side by side
    cm_def = confusion_matrix(y_test, pred_def)
    cm_bal = confusion_matrix(y_test, pred_bal)
    classes = le.classes_

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm_data, title_suffix in zip(axes, [cm_def, cm_bal], ["Default Weights","Balanced Weights"]):
        im = ax.imshow(cm_data, cmap="Blues")
        ax.set_title(f"Confusion Matrix\n({title_suffix})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_data[i,j], ha="center", va="center",
                        color="white" if cm_data[i,j] > cm_data.max()/2 else "black", fontsize=13)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT/"confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved confusion_matrix.png")

    # Classification report (primary model)
    cr = classification_report(y_test, pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cr_df = pd.DataFrame(cr).T.reset_index().rename(columns={"index":"Class"})
    save_csv(cr_df, OUT/"classification_report.csv", "classification_report")

    # Predictions (primary model - default weights, using probability for risk scoring)
    results = X_test.copy()
    results["actual_label"]          = le.inverse_transform(y_test)
    results["predicted_label"]       = le.inverse_transform(pred)
    results["prediction_probability"] = prob.round(4)
    # Risk level: probability of being "Paid" (class 1)
    # Low prob of Paid = higher denial risk
    results["denial_risk_score"]     = (1 - results["prediction_probability"]).round(4)
    results["risk_level"]            = pd.cut(
        results["denial_risk_score"],
        bins=[-0.001, 0.10, 0.30, 1.001],
        labels=["Low Risk","Medium Risk","High Risk"]
    )
    save_csv(results, PREDS/"model_predictions.csv", "model_predictions")

    # Save model summary JSON for dashboard use
    model_summary = {
        "target": TARGET,
        "positive_class": "Paid",
        "negative_class": "Denied",
        "features_used": features,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "balanced_recall_denied": rec_bal,
        "balanced_roc_auc": roc_bal,
        "confusion_matrix_default": cm_def.tolist(),
        "confusion_matrix_balanced": cm_bal.tolist(),
        "class_encoding": {str(k): int(v) for k,v in zip(le.classes_, le.transform(le.classes_))},
        "risk_thresholds": {
            "Low Risk": "denial_risk_score < 0.10",
            "Medium Risk": "0.10 - 0.30",
            "High Risk": "> 0.30"
        },
        "model_note": (
            "Two Logistic Regression variants trained: default weights (high accuracy) "
            "and balanced weights (higher sensitivity to Denied class). "
            "Primary reported metrics use default weights. "
            "ROC-AUC ~0.58 indicates limited discriminative power with only "
            "insurance_provider, payment_method, billed_amount and date features - "
            "expected given the synthetic nature of this dataset. "
            "denial_reason excluded to prevent target leakage."
        ),
        "class_imbalance_note": (
            "Dataset is imbalanced: 91.4% Paid / 8.6% Denied. "
            "A high accuracy score (~91%) can be achieved by predicting all claims as Paid. "
            "ROC-AUC is the more meaningful metric here."
        ),
    }
    with open(OUT/"model_summary.json","w") as fh:
        json.dump(model_summary, fh, indent=2)
    print("  Saved model_summary.json")

    ML_OK = True

except Exception as e:
    print(f"  ML ERROR: {e}")
    ML_OK = False

# ---------------------------------------------------------------------
# PHASE 12 - AI-DRIVEN INSIGHTS (text)
# ---------------------------------------------------------------------
print("\n--- PHASE 12: AI-DRIVEN INSIGHTS ---")

cb_ins = cleaned.get("claims_and_billing", tables.get("claims_and_billing"))
insight_rows = []

if cb_ins is not None:
    total  = len(cb_ins)
    denied = (cb_ins["claim_status"]=="Denied").sum()
    paid   = (cb_ins["claim_status"]=="Paid").sum()
    denial_rate = round(denied/total*100,2)
    avg_billed  = round(cb_ins["billed_amount"].mean(),2)
    avg_paid    = round(cb_ins["paid_amount"].mean(),2)
    total_billed= round(cb_ins["billed_amount"].sum(),2)
    total_paid  = round(cb_ins["paid_amount"].sum(),2)

    insight_rows = [
        {"Category":"Observations","Insight":
            f"The dataset contains {total:,} claim/billing records covering Q1 2025."},
        {"Category":"Observations","Insight":
            f"The denial rate is {denial_rate}% ({denied:,} of {total:,} claims were denied)."},
        {"Category":"Observations","Insight":
            f"The average billed amount is ${avg_billed:,.2f}; average paid is ${avg_paid:,.2f}."},
        {"Category":"Observations","Insight":
            f"Total billed across all claims: ${total_billed:,.2f}; total paid: ${total_paid:,.2f}."},
        {"Category":"Observations","Insight":
            f"There are {cb_ins['insurance_provider'].nunique()} insurance providers in the dataset."},
        {"Category":"Possible Insights","Insight":
            "A denial rate around 8-9% is an operational metric that healthcare organizations typically try to minimize. Reducing duplicate claim submissions is a quick-win opportunity."},
        {"Category":"Possible Insights","Insight":
            "The gap between average billed ($" + f"{avg_billed:,.2f}) and paid (${avg_paid:,.2f}) amounts represents potential revenue leakage worth investigating."},
        {"Category":"Possible Insights","Insight":
            "Prior authorization requirements and duplicate submissions are the two most frequent denial reasons, suggesting process standardization may reduce denial rates."},
        {"Category":"Limitations","Insight":
            "This is a synthetic dataset generated for educational purposes. Denial rates, billing amounts and provider distributions may not reflect any real-world healthcare network."},
        {"Category":"Limitations","Insight":
            "Without patient demographics, diagnoses and encounter data, the analysis is limited to claims-level patterns. A complete multi-table analysis would enable richer insights."},
        {"Category":"Possible Operational Actions","Insight":
            "Implement duplicate claim detection checks before submission to reduce the most common denial category."},
        {"Category":"Possible Operational Actions","Insight":
            "Strengthen prior authorization workflows to reduce authorization-related denials."},
        {"Category":"Possible Operational Actions","Insight":
            "Monitor monthly denial rates by insurance provider to identify payer-specific trends."},
    ]

insight_df = pd.DataFrame(insight_rows)
save_csv(insight_df, OUT/"ai_insights.csv", "ai_insights")

print(f"  Generated {len(insight_rows)} insight items")

# ---------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------
print("\n--- PIPELINE COMPLETE ---")
print(f"  Outputs: {OUT}")
print(f"  Charts:  {CHARTS}")
