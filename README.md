# AI-Driven Healthcare & Insurance Analytics

**IBM SkillsBuild Data Analytics with AI Academic Internship Program**  
Conducted by **BharatCares in Association with AICTE**

---

## Project Owner

| Field | Details |
|-------|---------|
| **Name** | Pappu Ramesh Baraf |
| **Email** | pappubaraf1@gmail.com |
| **LinkedIn** | https://in.linkedin.com/in/pappubarafaiml |
| **GitHub** | https://github.com/PappuBaraf |
| **College** | Finolex Academy of Management & Technology (FAMT), Ratnagiri |
| **Degree** | B.E. Computer Science & Engineering – AI/ML (2023–2027) |

---

## Project Description

An end-to-end **healthcare claims intelligence application** built using Python, scikit-learn, pandas, Matplotlib and Streamlit. The project analyses 70,000 synthetic insurance claim and billing records, performs exploratory data analysis, evaluates a Logistic Regression example for Paid/Denied outcomes, and presents results through a responsive interactive dashboard. The Patient & Provider Hub also provides rules-based bill review, policy-text keyword navigation, and session-only experiment/activity exports.

> **Important:** The CA Hospital Dataset Q1 2025 is a **synthetic** dataset generated for educational purposes. It does not represent real patient records, real insurance claims, or any specific healthcare organisation. All findings are analytical observations only and must not be used for medical decisions or insurance policy changes.

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | CA Hospital Dataset – Q1 2025 |
| **Source** | Kaggle |
| **Link** | https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025 |
| **Type** | Synthetic (generated for educational use) |
| **File used** | `claims_and_billing.csv` |
| **Rows** | 70,000 |
| **Columns** | 11 |
| **Period** | January – May 2025 |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| billing_id | String | Primary key |
| patient_id | String | Patient identifier |
| encounter_id | String | Encounter identifier |
| insurance_provider | String | Payer (7 categories: Aetna, BCBS, Cigna, Humana, Medicaid, Medicare, UHC) |
| payment_method | String | Insurance or Selfpay |
| claim_id | String | Claim identifier |
| claim_billing_date | String | Date billed (DD-MM-YYYY HH:MM) |
| billed_amount | Float | Total amount billed (USD) |
| paid_amount | Float | Amount actually paid (USD) |
| claim_status | **String (Target)** | **Paid or Denied** |
| denial_reason | String | Reason for denial (null when Paid — post-outcome, excluded from ML) |

---

## Key Results

| Metric | Value |
|--------|-------|
| Total Claims | 70,000 |
| Paid Claims | 64,002 (91.43%) |
| Denied Claims | 5,998 (8.57%) |
| Total Billed | $112,900,639.84 |
| Total Paid | $72,845,650.54 |
| Revenue Gap | $40,054,989.30 |
| ML Accuracy | 91.43% |
| ML ROC-AUC | 0.5764 |
| ML F1-Score | 0.9552 |

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.13 |
| Data Processing | pandas 3.0.0, numpy 2.4.2 |
| Visualisation | matplotlib 3.10.8 |
| Machine Learning | scikit-learn 1.8.0 |
| Dashboard | Streamlit |
| Notebook | Jupyter Notebook |

---

## Project Structure

```
Healthcare_Insurance_Analytics/
├── PappuRameshBaraf_HealthcareInsuranceAnalytics.ipynb  # Submission notebook
├── app.py                   # Streamlit dashboard and bill/policy review hub
├── pipeline.py              # Offline analytics & ML pipeline (12 phases)
├── generate_notebook.py     # Script to regenerate the .ipynb file
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── data/
│   ├── claims_and_billing.csv          # Primary dataset (70,000 rows)
│   └── cleaned/
│       ├── claims_and_billing_cleaned.csv
│       └── claims_and_billing_features.csv
│
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

## Setup & Run Instructions

### 1. Prerequisites

- Python 3.10 or higher
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `claims_and_billing.csv` from Kaggle:  
https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025

Place it in the `data/` folder:
```
Healthcare_Insurance_Analytics/
└── data/
    └── claims_and_billing.csv
```

### 4. Run the Analytics Pipeline

```bash
python pipeline.py
```

This generates all outputs in `outputs/` — charts, CSVs, model metrics, predictions, and AI insights.

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

### 6. Open the Jupyter Notebook

```bash
jupyter notebook PappuRameshBaraf_HealthcareInsuranceAnalytics.ipynb
```

Run all cells from top to bottom (Kernel → Restart & Run All).

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Patient & Provider Hub | De-identified bill review, repeated-line candidates, statistical outliers, arithmetic checks and policy keyword excerpts |
| Executive Overview | Claims, billed/paid totals, denial rate and provider KPIs |
| Claims & Insurance | Provider analysis, monthly trends, denial reasons and billing statistics |
| Interactive Explorer | Filter by provider, outcome and billing dates; chart and download matching rows |
| Exploratory Data Analysis | EDA charts, insights, numeric summaries and column explorer |
| Predictive Analytics / Model Results | Leakage-aware modeling methodology, metrics, classification report and predictions |
| Data Quality / Data Dictionary | Missing values, duplicates, data types, cleaning reports and field definitions |
| Experiments & Testing | Dataset validation checks and available model evaluation evidence |
| Load Data / Exports & Activity | Session uploads, manual entry, downloads and a privacy-minimized JSON activity export |
| About the Project | Project context, limitations and owner profile |

### Privacy and interpretation limits for bill review

- The public Streamlit demo is educational; **do not upload identifiable patient information or protected health information**.
- Bill checks are deterministic review aids: identical item/amount lines are only candidates, and statistical outliers are relative to the uploaded bill—not evidence of fraud, overcharging or inappropriate care.
- Policy search returns excerpts with matching words; it does not determine coverage, benefits or reimbursement.
- The PDF reader extracts selectable text only; scanned-image OCR is not included.
- Uploaded documents are not intentionally written to project files. Activity export contains generic events and aggregate counts only, remains in the current Streamlit session, and is not a persistent visitor log. Hosting providers may process requests under their own terms.
- No third-party LLM is called and no API key is required. Never put an API key in the repository or a public notebook.

---

## ML Methodology

| Aspect | Details |
|--------|---------|
| **Problem Type** | Binary Classification |
| **Target** | `claim_status` (Paid / Denied) |
| **Algorithm** | Logistic Regression (scikit-learn) |
| **Train/Test** | 80% / 20% stratified (56,000 / 14,000) |
| **Primary Metric** | ROC-AUC (correct for imbalanced data) |
| **Class Imbalance** | 91.4% Paid / 8.6% Denied |
| **Leakage Prevention** | denial_reason, paid_amount, all IDs excluded |

---

## Ethical Considerations

- Synthetic dataset — no real patient data was used.
- No medical diagnoses or treatment recommendations are made.
- All findings are observational — not causal conclusions.
- The ML model is for educational demonstration only.

---

## Submission Files

| File | Description |
|------|-------------|
| `PappuRameshBaraf_HealthcareInsuranceAnalytics.ipynb` | Complete project code (Jupyter Notebook) |
| `requirements.txt` | Python library dependencies |
| `PappuRameshBaraf_ProjectReport.docx` | Full project report (Word document) |
| `README.md` | This file |

---

*IBM SkillsBuild Data Analytics with AI Academic Internship — BharatCares / AICTE*  
*Author: Pappu Ramesh Baraf | pappubaraf1@gmail.com*
