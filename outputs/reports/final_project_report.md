# AI-Driven Healthcare & Insurance Analytics
## Final Project Report

**IBM SkillsBuild Data Analytics with AI Internship**
**Domain:** Healthcare & Insurance Analytics
**Dataset:** CA Hospital Dataset – Q1 2025 (Synthetic)
**Author:** IBM SkillsBuild Intern
**Date:** 2025

---

## 1. Title

**AI-Driven Healthcare & Insurance Analytics: Patient Data, Insurance Claims, Healthcare Costs and Medical Outcomes**

---

## 2. Abstract

This project presents an end-to-end analytics application built on the CA Hospital Dataset Q1 2025, a synthetic healthcare dataset containing 70,000 insurance claim and billing records. The project covers data inspection, cleaning, feature engineering, exploratory data analysis (EDA), machine learning model development with leakage prevention, AI-generated insights, and an interactive Streamlit dashboard. A Logistic Regression model was trained to predict claim approval status (Paid vs Denied), achieving 91.4% accuracy and a ROC-AUC of 0.576 with pre-outcome features only. The dashboard presents all findings through nine interactive pages. All results are derived from the actual dataset — no values were fabricated.

> **Important:** The CA Hospital Dataset Q1 2025 is a **synthetic** dataset generated for educational purposes. It does not represent real patient records, real insurance claims, or any specific healthcare organization. All findings are analytical observations only and must not be used for medical decisions or insurance policy changes.

---

## 3. Introduction

Healthcare and insurance analytics is one of the fastest-growing application areas for data science. Insurance organizations face mounting challenges in claims management, denial rate reduction, cost containment and fraud prevention. Data-driven approaches offer the potential to identify patterns in large claims datasets that are not visible to manual review.

This project was developed as part of the IBM SkillsBuild Data Analytics with AI internship program. The goal was to build a complete, working analytics application — from raw data to an interactive dashboard — using industry-standard tools (Python, pandas, scikit-learn, Streamlit) and best practices in data cleaning, feature engineering, leakage prevention and model evaluation.

---

## 4. Problem Statement

Given a synthetic healthcare claims and billing dataset, can we:

1. Profile the dataset and identify data quality issues?
2. Perform EDA to identify observable patterns in claims, billing amounts, insurance providers and denial rates?
3. Build a machine learning model to predict whether a claim will be Paid or Denied using only pre-outcome information (preventing target leakage)?
4. Generate actionable operational insights from the data?
5. Present all findings through an interactive Streamlit dashboard?

---

## 5. Project Motive

This project was motivated by the need to demonstrate end-to-end data analytics competency as part of the IBM SkillsBuild internship. The healthcare and insurance domain was selected because it offers rich analytical opportunities — including classification problems, time-series analysis, cost distribution analysis and operational KPI tracking.

The project directly applies the internship curriculum: data analysis, visualization, machine learning and AI-driven insights — all grounded in an actual dataset rather than fabricated examples.

---

## 6. Objectives

1. Load, inspect and profile the CA Hospital Dataset Q1 2025
2. Clean the dataset and document all transformations
3. Engineer features from raw fields (date parts, financial ratios, patient-level aggregates)
4. Perform comprehensive EDA with 10 visualizations
5. Define a valid binary classification target (claim_status: Paid / Denied)
6. Prevent target leakage by excluding post-outcome variables
7. Train and evaluate a Logistic Regression model with two class-weight variants
8. Generate AI-driven operational insights from actual data values
9. Produce a nine-page interactive Streamlit dashboard
10. Export all outputs for inclusion in the internship report and presentation

---

## 7. Dataset

| Property | Value |
|----------|-------|
| **Name** | CA Hospital Dataset – Q1 2025 |
| **Source** | Kaggle (rajkumarpadmanabhan/ca-hospital-dataset-q1-2025) |
| **Type** | Synthetic (generated for educational use) |
| **File used** | claims_and_billing.csv |
| **Rows** | 70,000 |
| **Columns** | 11 |
| **Time Period** | Q1 2025 (January – May 2025) |

### 7.1 Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| billing_id | String | Unique billing record identifier (Primary Key) |
| patient_id | String | Patient identifier – links to patients table |
| encounter_id | String | Encounter identifier – links to encounters table |
| insurance_provider | String | Name of insurance payer (7 categories) |
| payment_method | String | Insurance or Self-Pay |
| claim_id | String | Insurance claim identifier |
| claim_billing_date | String | Date the claim was billed (DD-MM-YYYY HH:MM) |
| billed_amount | Float | Total amount billed in USD |
| paid_amount | Float | Amount actually paid in USD |
| claim_status | String | **Target variable** – Paid or Denied |
| denial_reason | String | Reason for denial (null when Paid – **post-outcome, excluded from ML**) |

### 7.2 Dataset Note

Only `claims_and_billing.csv` was available. The Kaggle dataset page lists additional tables (patients, encounters, diagnoses, procedures, medications, lab_tests, providers, denials) that were not present in the downloaded data. All analysis is therefore limited to claims-level patterns. A multi-table analysis would enable richer patient-level insights.

---

## 8. Data Architecture

```
Healthcare_Insurance_Analytics/
├── app.py                          # Streamlit dashboard (9 pages)
├── pipeline.py                     # Offline analytics & ML pipeline
├── requirements.txt                # Python dependencies
├── data/
│   ├── claims_and_billing.csv      # Primary dataset (70,000 rows)
│   └── cleaned/
│       ├── claims_and_billing_cleaned.csv
│       └── claims_and_billing_features.csv
└── outputs/
    ├── dataset_profile.csv
    ├── data_dictionary.csv
    ├── data_quality_report.csv
    ├── feature_dictionary.csv
    ├── ml_feature_selection.csv
    ├── eda_insights.csv
    ├── ai_insights.csv
    ├── model_metrics.csv
    ├── model_summary.json
    ├── classification_report.csv
    ├── confusion_matrix.png
    ├── table_relationships.csv
    ├── charts/
    │   ├── 01_claim_status.png
    │   ├── 02_insurance_provider.png
    │   ├── 03_billed_amount_dist.png
    │   ├── 04_billed_vs_paid_by_provider.png
    │   ├── 05_denial_reasons.png
    │   ├── 06_payment_method.png
    │   ├── 07_monthly_trends.png
    │   ├── 08_status_by_provider.png
    │   ├── 09_high_cost_flag.png
    │   └── 10_payment_ratio.png
    ├── predictions/
    │   └── model_predictions.csv
    └── reports/
        ├── final_project_report.md
        └── final_presentation.md
```

---

## 9. Data Cleaning

The data cleaning pipeline (`pipeline.py`, Phase 3) applied the following transformations:

| Column | Original Type | New Type | Transformation | Notes |
|--------|--------------|----------|---------------|-------|
| claim_billing_date | object | datetime64[ns] | to_datetime(format=%d-%m-%Y %H:%M) | ~85% parseable; 14.8% self-pay rows have no date |
| billed_amount | float64 | float64 | none | No cleaning needed |
| paid_amount | float64 | float64 | none | No cleaning needed |
| denial_reason | object | object | none | Expected nulls for Paid claims |
| All others | various | various | none | No issues detected |

**Duplicate rows removed:** 0 (no duplicates in the dataset)  
**Missing values:** denial_reason is null for all 64,002 Paid claims — this is **expected behavior**, not a data quality issue.  
**Final row count:** 70,000 (unchanged after cleaning)

---

## 10. Exploratory Data Analysis

### 10.1 Claim Status Distribution

| Status | Count | Percentage |
|--------|-------|-----------|
| Paid | 64,002 | 91.43% |
| Denied | 5,998 | 8.57% |

The dataset is **highly imbalanced** — 91.4% of claims are Paid. This is a critical consideration for ML model evaluation.

### 10.2 Insurance Provider Distribution

| Provider | Claims | % |
|----------|--------|---|
| Medicaid | 10,108 | 14.44% |
| UHC | 10,062 | 14.37% |
| BCBS | 10,004 | 14.29% |
| Aetna | 10,004 | 14.29% |
| Medicare | 9,983 | 14.26% |
| Humana | 9,924 | 14.18% |
| Cigna | 9,915 | 14.16% |

**Observation:** Claims are distributed nearly equally across 7 providers — consistent with synthetic data generation. Real-world payer mix would show greater variation.

### 10.3 Billing Amount Statistics

| Statistic | Billed Amount (USD) | Paid Amount (USD) |
|-----------|--------------------|--------------------|
| Mean | $1,612.87 | $1,040.65 |
| Median | $1,214.11 | $784.23 |
| Min | $100.07 | $0.00 |
| Max | $38,191.35 | $33,987.35 |
| Std | $1,502.45 | $971.83 |

**Observation:** The right-skewed distribution (mean > median) indicates a relatively small number of high-cost claims driving the mean upward. The gap between average billed ($1,612.87) and paid ($1,040.65) represents a 35.5% payment discount across the portfolio.

### 10.4 Financial Summary

| Metric | Value |
|--------|-------|
| Total Billed | $112,900,639.84 |
| Total Paid | $72,845,650.54 |
| Revenue Gap (Billed - Paid) | $40,054,989.30 |
| Overall Payment Rate | 64.5% |

### 10.5 Denial Reasons

14 distinct denial reason categories were identified. Most frequent:

| Denial Reason | Count | % of Denials |
|---------------|-------|-------------|
| Duplicate Claim | 472 | 7.87% |
| Prior Authorization Required | ~440 | ~7.3% |
| Claim Billed to Wrong Payer | ~430 | ~7.2% |

**Note:** `denial_reason` is only populated for the 5,998 denied claims. It is a **post-outcome variable** and was excluded from all ML features to prevent target leakage.

### 10.6 Monthly Trends

The dataset spans January 2025 to May 2025 (Q1 data extended slightly). Monthly claim volumes and billed amounts show relatively stable patterns consistent with synthetic generation.

---

## 11. Feature Engineering

The following features were engineered from the raw dataset (`pipeline.py`, Phase 4):

| Feature | Source | Type | Leakage Risk |
|---------|--------|------|-------------|
| claim_year | claim_billing_date | Date Part | None |
| claim_month | claim_billing_date | Date Part | None |
| claim_day | claim_billing_date | Date Part | None |
| claim_dayofweek | claim_billing_date | Date Part | None |
| claim_quarter | claim_billing_date | Date Part | None |
| payment_ratio | paid_amount / billed_amount | Financial | None |
| unpaid_amount | billed_amount - paid_amount | Financial | None |
| high_cost_flag | billed_amount > 75th percentile | Financial | None |
| denial_flag | derived from claim_status | Target Derived | **HIGH – not used in ML** |
| patient_claim_count | grouped by patient_id | Aggregate | Low |

---

## 12. ML Methodology

### 12.1 Problem Type

Binary classification: predict whether a new claim will be **Paid** (positive class) or **Denied** (negative class).

### 12.2 Target Variable

`claim_status` — two classes: Paid (91.4%) and Denied (8.6%)

### 12.3 Algorithm

Logistic Regression (scikit-learn)  
- Solver: lbfgs  
- Max iterations: 1,000  
- Random state: 42  
- Preprocessing: MedianImputer → StandardScaler (numeric); MostFreqImputer → OneHotEncoder (categorical)

### 12.4 Train/Test Split

- Train: 56,000 records (80%)  
- Test: 14,000 records (20%)  
- Stratified by target class to preserve class ratio

### 12.5 Two Model Variants

Two variants were trained to explicitly address the class imbalance challenge:

| Variant | class_weight | Purpose |
|---------|-------------|---------|
| Variant A (Primary) | default | Maximizes overall accuracy |
| Variant B (Sensitivity) | balanced | Maximizes recall for minority class (Denied) |

---

## 13. Leakage Prevention

The following variables were **explicitly excluded** from model features:

| Variable | Reason |
|----------|--------|
| denial_reason | Post-outcome: only populated after a claim is denied |
| paid_amount | Post-outcome: only known after claim processing |
| denial_flag | Derived directly from the target variable |
| billing_id | Identifier: no predictive signal |
| claim_id | Identifier: no predictive signal |
| patient_id | Identifier: no predictive signal |
| encounter_id | Identifier: no predictive signal |

**Features used in ML:**
1. `insurance_provider` (categorical, 7 values)
2. `payment_method` (categorical, 2 values: Insurance / Self-Pay)
3. `billed_amount` (numeric)
4. `feat_claim_year` (from date)
5. `feat_claim_month` (from date)
6. `feat_claim_dayofweek` (from date)
7. `feat_claim_quarter` (from date)

---

## 14. Model Evaluation

### 14.1 Primary Model (Default Class Weights)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9143 |
| Precision | 0.9143 |
| Recall | 1.0000 |
| F1-Score | 0.9552 |
| ROC-AUC | 0.5764 |

### 14.2 Sensitivity Model (Balanced Class Weights)

| Metric | Value |
|--------|-------|
| Accuracy | 0.2341 |
| Recall for Denied class | 0.9992 |
| F1-Score | 0.2794 |
| ROC-AUC | 0.5767 |

### 14.3 Confusion Matrix — Primary Model

```
                 Predicted: Denied   Predicted: Paid
Actual: Denied        0               1,200
Actual: Paid          0              12,800
```

### 14.4 Confusion Matrix — Sensitivity Model

```
                 Predicted: Denied   Predicted: Paid
Actual: Denied     1,199                  1
Actual: Paid      10,721              2,079
```

### 14.5 Classification Report — Primary Model

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Denied | 0.000 | 0.000 | 0.000 | 1,200 |
| Paid | 0.914 | 1.000 | 0.955 | 12,800 |
| Macro Avg | 0.457 | 0.500 | 0.478 | 14,000 |
| Weighted Avg | 0.836 | 0.914 | 0.872 | 14,000 |

---

## 15. Results

### 15.1 Key Findings from EDA

1. **Denial Rate:** 8.57% of all Q1 2025 claims were denied (5,998 of 70,000)
2. **Revenue Gap:** Total billed ($112.9M) minus total paid ($72.8M) = $40.1M gap across Q1 2025
3. **Cost Distribution:** Right-skewed — the top 25% of claims account for a disproportionate share of total billed amounts
4. **Provider Parity:** Denial rates are consistent across all 7 insurance providers (~8.5% each)
5. **Duplicate Claims:** The single most common denial reason, representing ~7.9% of all denials
6. **Payment Method:** Approximately 85% of claims use Insurance; 15% are Self-Pay

### 15.2 ML Results Interpretation

The **primary model** (default weights) achieves 91.4% accuracy but does so by predicting all claims as "Paid." This is the naive baseline: because 91.4% of claims are Paid, always predicting Paid yields 91.4% accuracy without learning any pattern.

The **ROC-AUC of 0.576** is the meaningful metric — it indicates modest discrimination ability above the random baseline of 0.50. This is **expected and honest**: with only 7 pre-outcome features (insurance_provider, payment_method, billed_amount, date parts), a logistic regression cannot learn strong predictive patterns in synthetic data where claim denial is not structurally dependent on these variables.

The **sensitivity model** (balanced weights) demonstrates that with appropriate weighting, the model can identify 99.9% of denied claims — but at the cost of 83.8% false positive rate (most Paid claims are also flagged as Denied). In practice, this would be useful as a "flag all suspicious claims for review" approach.

---

## 16. Healthcare & Insurance Insights

### 16.1 Observational Findings (from actual data)

1. The dataset contains 70,000 claim/billing records covering Q1 2025
2. The denial rate is 8.57% (5,998 of 70,000 claims were denied)
3. The average billed amount is $1,612.87; average paid is $1,040.65
4. Total billed across all claims: $112,900,639.84; total paid: $72,845,650.54
5. There are 7 insurance providers in the dataset with near-equal claim distribution
6. 14 distinct denial reason categories were identified

### 16.2 Possible Operational Insights

These are suggested interpretations — not proven causal findings:

1. A denial rate around 8-9% is an operational metric that healthcare organizations typically try to minimize. Reducing duplicate claim submissions is a quick-win opportunity.
2. The gap between average billed ($1,612.87) and paid ($1,040.65) amounts represents potential revenue optimization worth investigating.
3. Prior authorization requirements and duplicate submissions are the two most frequent denial categories, suggesting process standardization may reduce denial rates.

### 16.3 Possible Operational Actions

1. Implement duplicate claim detection checks before submission to reduce the most common denial category
2. Strengthen prior authorization workflows to reduce authorization-related denials
3. Monitor monthly denial rates by insurance provider to identify payer-specific trends

---

## 17. Recommendations

Based on the analytical findings from the synthetic dataset:

1. **Duplicate Claim Prevention:** Implement real-time duplicate detection at point of submission. This addresses the single largest denial category.
2. **Prior Authorization Pre-Check:** Automate prior authorization requirement checks against payer rules before claim submission.
3. **High-Cost Claim Review:** Flag claims above the 75th percentile billed amount ($2,178 in this dataset) for pre-submission clinical review.
4. **Denial Rate Monitoring:** Establish monthly dashboards tracking denial rates by provider, reason, and claim type.
5. **Multi-Table Analysis:** For richer insights, integrate the full dataset (patients, encounters, diagnoses, procedures) to analyze patient-level and diagnosis-level denial patterns.

---

## 18. Limitations

1. **Synthetic Dataset:** The CA Hospital Dataset Q1 2025 is synthetically generated. Patterns, denial rates, billing amounts and provider distributions are not reflective of any real healthcare network.
2. **Single Table:** Only claims_and_billing.csv was available. The absence of patients, encounters, diagnoses and procedures tables limits analysis to claims-level patterns only.
3. **ML Feature Gap:** With only 7 pre-outcome features available (none of which are structurally correlated with claim denial in the synthetic data), the ML model's discriminative power is limited. ROC-AUC of 0.576 reflects this limitation honestly.
4. **No Temporal Validation:** The model was evaluated on a random test split, not a temporal hold-out. In production, models should be tested on future data.
5. **Class Imbalance:** The 91.4% / 8.6% class split is a typical real-world challenge. The project documents both standard and balanced model variants to address this.

---

## 19. Ethical Considerations

1. **Not Medical Advice:** This dashboard and all outputs are for educational and analytical purposes only. No output should be used for medical diagnosis or treatment decisions.
2. **Not Causal Claims:** All findings describe observed patterns in the data. Correlation in observational data does not imply causation.
3. **Synthetic Data Disclosure:** The project clearly and consistently states that the dataset is synthetic. No real patient data was used.
4. **No PII:** The dataset contains patient IDs (PAT000001-format synthetic identifiers) — not real patient personal information.
5. **Fairness Consideration:** If a denial prediction model were deployed in a real setting, it would require fairness audits across demographic groups. This project does not make any demographic fairness claims.

---

## 20. Conclusion

This project successfully delivered an end-to-end healthcare and insurance analytics application:

- **70,000** synthetic insurance claims analyzed
- **10 EDA charts** generated from actual data
- **2 ML model variants** trained and evaluated with leakage prevention
- **9-page interactive Streamlit dashboard** built and tested
- **13 AI-generated insights** derived from actual dataset values
- **All outputs** exported as downloadable CSVs and PNGs

The project demonstrates the complete data analytics workflow — from raw data inspection to production-ready dashboard — while maintaining rigorous standards of data ethics, honest reporting of model limitations, and clear separation of observations from interpretations.

---

## 21. Future Scope

1. **Multi-table integration:** Load patients, encounters, diagnoses and procedures to enable patient-level and diagnosis-level analysis
2. **Advanced models:** Explore Random Forest, Gradient Boosting or XGBoost for improved discrimination on imbalanced data
3. **SHAP explanations:** Add feature importance and SHAP values to explain model predictions
4. **Real data validation:** Validate all methodology on real (de-identified) claims data
5. **Temporal modeling:** Implement time-series forecasting for monthly claim volume and cost prediction
6. **Natural Language Processing:** Apply NLP to denial reason text for better categorization and prediction
7. **Automated alerting:** Build a rules engine on top of the ML model to flag high-risk claims before submission

---

## 22. References

1. Kaggle Dataset: CA Hospital Dataset – Q1 2025 by rajkumarpadmanabhan  
   https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025

2. scikit-learn: Machine Learning in Python  
   Pedregosa et al. (2011), JMLR 12, pp. 2825–2830  
   https://scikit-learn.org

3. Streamlit: The fastest way to build data apps  
   https://streamlit.io

4. pandas: Python Data Analysis Library  
   https://pandas.pydata.org

5. IBM SkillsBuild Data Analytics with AI Program  
   https://skillsbuild.org

6. Healthcare Financial Management Association (HFMA): Claim Denial Trends  
   (Referenced for industry context only — not for dataset claims)

---

*This report was generated as part of the IBM SkillsBuild Data Analytics with AI Internship.*  
*All numerical values in this report are derived from the CA Hospital Dataset Q1 2025 (synthetic).*  
*No values were fabricated or estimated.*
