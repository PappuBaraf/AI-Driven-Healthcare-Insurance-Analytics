# AI-Driven Healthcare & Insurance Analytics
## Final Presentation

**IBM SkillsBuild Data Analytics with AI Internship**
**Format:** Slide-by-slide Markdown (18 slides)

---

## Slide 1 — Title

# AI-Driven Healthcare & Insurance Analytics
### Patient Data • Insurance Claims • Healthcare Costs • Medical Outcomes

**IBM SkillsBuild Data Analytics with AI Internship**

**Dataset:** CA Hospital Dataset – Q1 2025 (Synthetic)

**Tools:** Python • pandas • scikit-learn • Streamlit • Matplotlib

> *Educational project using synthetic healthcare data. Not medical advice.*

---

## Slide 2 — Problem Statement

# Problem Statement

Healthcare organizations face significant operational challenges in claims management:

- **Claim denials** represent lost revenue and increased administrative burden
- **Billing errors** and **duplicate submissions** are common and preventable
- **Prior authorization** requirements create friction in the claims process
- **Manual review** of 70,000+ claims per quarter is not scalable

### The Question:
> *Can data analytics and machine learning help identify patterns in healthcare claims that support operational decision-making?*

---

## Slide 3 — Project Motive

# Project Motive

This project was built to demonstrate **end-to-end data analytics capability** for the IBM SkillsBuild internship program.

### Why Healthcare & Insurance?

| Reason | Detail |
|--------|--------|
| Rich data domain | Claims, billing, clinical, demographics |
| Real business impact | Denial reduction, revenue optimization |
| Analytical diversity | Classification, EDA, time-series, NLP potential |
| IBM alignment | IBM Watson Health, IBM Consulting Healthcare |

### What this project proves:
- Ability to work with real-world messy data
- Rigorous ML methodology with leakage prevention
- Communication of findings to non-technical stakeholders

---

## Slide 4 — Objectives

# Project Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Load, inspect and profile the dataset | ✅ Done |
| 2 | Clean data and document transformations | ✅ Done |
| 3 | Engineer meaningful features | ✅ Done |
| 4 | Perform comprehensive EDA with 10 charts | ✅ Done |
| 5 | Define a valid ML prediction target | ✅ Done |
| 6 | Prevent target leakage (deny_reason excluded) | ✅ Done |
| 7 | Train and evaluate Logistic Regression model | ✅ Done |
| 8 | Generate AI-driven operational insights | ✅ Done |
| 9 | Build 9-page interactive Streamlit dashboard | ✅ Done |
| 10 | Export all outputs for report and presentation | ✅ Done |

---

## Slide 5 — Dataset

# Dataset

**CA Hospital Dataset – Q1 2025**  
Source: Kaggle (rajkumarpadmanabhan/ca-hospital-dataset-q1-2025)

| Property | Value |
|----------|-------|
| Type | **Synthetic** (for educational use) |
| File | claims_and_billing.csv |
| Rows | **70,000** |
| Columns | 11 |
| Period | January – May 2025 |

### Available Columns
`billing_id` • `patient_id` • `encounter_id` • `insurance_provider`  
`payment_method` • `claim_id` • `claim_billing_date`  
`billed_amount` • `paid_amount` • `claim_status` • `denial_reason`

### Dataset Note
Only `claims_and_billing.csv` was available. Multi-table analysis (patients, encounters, diagnoses) was not possible with this download.

---

## Slide 6 — Data Architecture

# Data Architecture

```
┌─────────────────────────────────────────────────────┐
│                  data/                               │
│   claims_and_billing.csv (70,000 rows × 11 cols)   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              pipeline.py (12 phases)                 │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │Inspect   │  │Clean +   │  │ML: Logistic        │ │
│  │Profile   │  │Engineer  │  │Regression (x2)     │ │
│  └──────────┘  └──────────┘  └────────────────────┘ │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                outputs/                              │
│  10 EDA charts  •  model_metrics.csv                │
│  ai_insights    •  model_predictions.csv            │
│  data_dictionary •  classification_report.csv       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│          app.py — Streamlit Dashboard (9 pages)     │
└─────────────────────────────────────────────────────┘
```

---

## Slide 7 — Data Cleaning

# Data Cleaning Results

| Column | Issue Found | Action Taken |
|--------|-------------|-------------|
| claim_billing_date | String format DD-MM-YYYY HH:MM | Parsed to datetime64 |
| denial_reason | 85.7% null values | **Expected** — null = Paid claim |
| All others | No issues | No action needed |

### Quality Summary

| Metric | Value |
|--------|-------|
| Total Rows | 70,000 |
| Duplicate Rows | **0** |
| Columns with issues | 1 (date format) |
| Missing values (unexpected) | 0 |

### Key Finding
`denial_reason` nulls are **not a data quality problem** — they represent the 91.4% of claims that were approved. This distinction is critical for ML (post-outcome variable).

---

## Slide 8 — Exploratory Data Analysis (EDA)

# Exploratory Data Analysis

### 10 EDA Charts Produced

| Chart | Key Finding |
|-------|-------------|
| Claim Status Distribution | 91.4% Paid / 8.6% Denied |
| Insurance Provider | 7 providers, near-equal split (~14% each) |
| Billed Amount Distribution | Right-skewed; mean $1,612 > median $1,214 |
| Billed vs Paid by Provider | Consistent ~35% discount across all providers |
| Denial Reasons | 14 categories; Duplicate Claim most common |
| Payment Method | ~85% Insurance / ~15% Self-Pay |
| Monthly Trends | Stable volumes across Jan–May 2025 |
| Status by Provider | Denial rates consistent across providers |
| High-Cost Flag | 25% of claims exceed $2,178 billed |
| Payment Ratio | Denied claims have payment ratio = 0 |

---

## Slide 9 — Key Findings

# Key Findings from EDA

### Financial KPIs (Actual Values)

| Metric | Value |
|--------|-------|
| Total Claims | 70,000 |
| Total Billed | **$112,900,639.84** |
| Total Paid | **$72,845,650.54** |
| Revenue Gap | **$40,054,989.30** |
| Average Billed / Claim | $1,612.87 |
| Average Paid / Claim | $1,040.65 |
| Overall Payment Rate | 64.5% |

### Claims Quality KPIs

| Metric | Value |
|--------|-------|
| Denial Rate | **8.57%** |
| Denied Claims | 5,998 |
| Paid Claims | 64,002 |
| Top Denial Reason | Duplicate Claim |

---

## Slide 10 — ML Problem

# Machine Learning Problem Definition

### Target Variable: `claim_status`

| Class | Count | % |
|-------|-------|---|
| Paid | 64,002 | **91.4%** |
| Denied | 5,998 | **8.6%** |

**Challenge:** Highly imbalanced dataset (91:9 ratio)

### Leakage Prevention

Variables **excluded** from ML features:

| Excluded | Reason |
|----------|--------|
| `denial_reason` | Post-outcome leakage |
| `paid_amount` | Post-outcome leakage |
| `billing_id` | Identifier only |
| `claim_id` | Identifier only |
| `patient_id` | Identifier only |
| `encounter_id` | Identifier only |

### Features Used (7 pre-outcome features)
insurance_provider • payment_method • billed_amount • claim_year • claim_month • claim_dayofweek • claim_quarter

---

## Slide 11 — ML Methodology

# Machine Learning Methodology

### Algorithm: Logistic Regression

**Why Logistic Regression?**
- Interpretable baseline for binary classification
- Works with mixed feature types after preprocessing
- Fast to train on 70,000 records
- ROC-AUC is a valid comparison metric

### Pipeline

```
Input Features (7)
     │
     ├── Numeric (billed_amount, date parts)
     │       └── MedianImputer → StandardScaler
     │
     └── Categorical (insurance_provider, payment_method)
             └── MostFreqImputer → OneHotEncoder
                          │
                          ▼
              LogisticRegression (max_iter=1000)
```

### Two Variants Trained

| Variant | class_weight | Purpose |
|---------|-------------|---------|
| A – Primary | default | Maximize accuracy |
| B – Sensitivity | balanced | Maximize recall for Denied class |

**Train/Test:** 80% / 20% stratified split (56,000 train / 14,000 test)

---

## Slide 12 — Model Results

# Model Evaluation Results

### Primary Model (Default Weights)

| Metric | Value | Note |
|--------|-------|------|
| Accuracy | **91.43%** | Reflects majority-class prediction |
| Precision | 0.9143 | For Paid class |
| Recall | 1.0000 | All Paid claims identified |
| F1-Score | 0.9552 | Weighted by majority class |
| **ROC-AUC** | **0.5764** | Primary discriminative metric |

### Sensitivity Model (Balanced Weights)

| Metric | Value | Note |
|--------|-------|------|
| Accuracy | 23.4% | Predicts many as Denied |
| Recall (Denied) | **99.9%** | Catches almost all denials |
| ROC-AUC | 0.5767 | Similar discrimination |

### Interpretation
> ROC-AUC of 0.576 is modest but honest. With 7 pre-outcome features in a synthetic dataset, this level of discrimination is expected. The model correctly learns that ~91% of claims are Paid as the base rate.

---

## Slide 13 — Prediction Results

# Prediction Results

**Test set:** 14,000 claims

### Denial Risk Score Distribution

| Risk Level | Count | % |
|------------|-------|---|
| Low Risk (denial risk < 10%) | 8,608 | 61.5% |
| Medium Risk (10–30%) | 5,392 | 38.5% |
| High Risk (>30%) | 0 | 0% |

### Sample Predictions (Top 5 by Denial Risk)

| Provider | Method | Billed | Actual | Predicted | Denial Risk |
|----------|--------|--------|--------|-----------|-------------|
| Cigna | Insurance | $629.94 | Paid | Paid | 10.5% |
| Medicaid | Insurance | $1,485 | Paid | Paid | 10.8% |
| BCBS | Insurance | $1,602 | **Denied** | Paid | 9.9% |

### Note
The primary model predicts all test claims as "Paid" (consistent with 91.4% majority class). The denial risk score provides a relative ranking for claims review prioritization.

---

## Slide 14 — Healthcare & Insurance Insights

# AI-Driven Healthcare & Insurance Insights

### Data-Driven Observations (actual values)

1. **70,000** claim records analyzed for Q1 2025
2. **$40.1M revenue gap** between billed and paid amounts
3. **8.57% denial rate** — ~1 in 12 claims is denied
4. **Duplicate Claim** is the #1 denial reason
5. **7 insurance providers** with nearly equal claim distribution

### Possible Operational Insights

> "A denial rate around 8-9% is an operational metric that healthcare organizations typically try to minimize. Reducing duplicate claim submissions is a quick-win opportunity."

> "The gap between average billed ($1,612.87) and paid ($1,040.65) represents potential revenue leakage worth investigating."

> "Prior authorization requirements and duplicate submissions are the top two denial categories — suggesting process standardization could reduce denial rates."

**Disclaimer:** These are interpretations of synthetic data patterns — not findings from real healthcare operations.

---

## Slide 15 — Recommendations

# Recommendations

Based on analysis of the synthetic dataset:

| # | Recommendation | Addresses |
|---|----------------|-----------|
| 1 | **Implement real-time duplicate claim detection** at submission | #1 denial reason |
| 2 | **Automate prior authorization checks** against payer rules | #2 denial reason |
| 3 | **Flag claims >75th percentile** ($2,178+) for pre-submission review | High-cost outliers |
| 4 | **Build monthly denial rate dashboard** by provider, reason, claim type | Monitoring |
| 5 | **Integrate full multi-table dataset** for patient-level analysis | Richer insights |

### For ML Improvement

- Add patient demographics and diagnosis codes as features
- Use ensemble methods (Random Forest, XGBoost) for better imbalanced classification
- Implement SHAP values for model explainability
- Validate on real (de-identified) data before any production use

---

## Slide 16 — Limitations & Ethics

# Limitations & Ethical Considerations

### Limitations

| Limitation | Impact |
|-----------|--------|
| Synthetic dataset | Patterns don't reflect real healthcare networks |
| Single table only | No patient/clinical context for richer analysis |
| Low ROC-AUC (0.576) | Limited ML discriminative power with available features |
| No temporal validation | Random split, not future-data hold-out |
| Class imbalance | 91:9 ratio complicates ML evaluation |

### Ethical Guardrails

- **Not medical advice:** No output should influence medical or treatment decisions
- **No causation claimed:** All findings are correlational observations
- **Synthetic data disclosed:** Clearly stated throughout all outputs
- **No real PII:** Only synthetic patient identifiers used
- **Fairness not evaluated:** A production model would require demographic fairness audits

---

## Slide 17 — Dashboard

# Interactive Streamlit Dashboard

### 9 Pages Built

| Page | Content |
|------|---------|
| Executive Overview | KPIs: 70K claims, $112.9M billed, 8.57% denial rate |
| Claims & Insurance | Tabbed: status, providers, billing, denials + monthly trends |
| Exploratory Data Analysis | All 10 charts + EDA insights + column explorer |
| Predictive Analytics | ML problem, leakage table, model architecture |
| Model Results | Metrics, confusion matrices, classification report, predictions |
| AI-Driven Insights | 13 auto-generated insights organized by category |
| Data Dictionary | Column definitions + feature dictionary |
| Data Quality | Missing values, data types, cleaning report |
| Export | Download all CSVs and charts |

### Technology Stack
- **Streamlit** — interactive web dashboard
- **pandas** — data loading and analysis
- **Matplotlib** — chart rendering
- **scikit-learn** — ML pipeline (run offline in pipeline.py)
- **Python** — all pipeline and dashboard code

### Run Command
```bash
python pipeline.py      # Run once to generate outputs
streamlit run app.py    # Launch dashboard
```

---

## Slide 18 — Conclusion & Future Scope

# Conclusion & Future Scope

### What Was Achieved

| Deliverable | Status |
|-------------|--------|
| End-to-end analytics pipeline | ✅ pipeline.py (713 lines, 12 phases) |
| 10 EDA visualizations | ✅ All from actual data |
| ML model with leakage prevention | ✅ Logistic Regression, 2 variants |
| 9-page Streamlit dashboard | ✅ Fully interactive |
| AI-driven insights | ✅ 13 observations from actual values |
| Final report | ✅ 22 sections, all actual results |

### Future Scope

1. Integrate full multi-table dataset (patients, encounters, diagnoses)
2. Advanced models: Random Forest, XGBoost, neural networks
3. SHAP explainability for model transparency
4. NLP on denial reason text for pattern mining
5. Real-time prediction API for claim submission workflow
6. Temporal forecasting of claim volumes and costs
7. Validate on real de-identified healthcare data

### Summary Statement

> This project demonstrates end-to-end data analytics proficiency — from raw synthetic data to a production-grade interactive dashboard — using industry-standard Python tools and rigorous ML methodology. All findings are grounded in actual dataset values, with clear ethical guardrails and honest reporting of model limitations.

---

**IBM SkillsBuild Data Analytics with AI Internship**  
**AI-Driven Healthcare & Insurance Analytics**  
**CA Hospital Dataset Q1 2025 (Synthetic) — Educational Use Only**
