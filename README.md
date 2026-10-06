# Geldium Financial — Credit Delinquency Risk Prediction & Banking Analytics

An end-to-end credit delinquency risk analytics project for a fictional bank (**"Geldium Financial"**), simulating a real-world banking analyst engagement: synthetic data generation → EDA & data cleaning → predictive modeling → SAS port → Tableau-ready outputs → stakeholder deliverables.

---

## 📁 Repository Structure

```
geldium_project/
├── data/
│   ├── generate_data.py               # Synthetic customer dataset generator (~1,500 records)
│   ├── geldium_customers_raw.csv      # Generated raw dataset (with missingness & anomalies)
│   └── geldium_customers_clean.csv    # Cleaned, imputed, and engineered dataset
├── notebooks/
│   ├── 01_eda_cleaning.py             # Profiling, anomaly detection, cleaning & risk tiering
│   └── 02_modeling.py                 # Logistic Regression & Random Forest modeling pipeline
├── sas/
│   └── geldium_risk_scoring.sas       # Production SAS script (PROC LOGISTIC & PROC FREQ)
├── tableau/
│   ├── model_predictions_for_tableau.csv  # Cleaned CSV feed with model probabilities
│   └── build_notes.md                 # Complete Tableau dashboard build specifications
├── dashboard/
│   ├── index.html                     # Live interactive web dashboard (http://localhost:8050)
│   └── data.json                      # Aggregated dashboard data payload
├── outputs/
│   ├── eda_summary.txt                # Full text EDA and data profiling summary report
│   ├── model_results.json             # Complete performance metrics JSON
│   ├── logistic_coefficients.json     # Logistic Regression coefficients & Odds Ratios
│   ├── random_forest_feature_importance.json # Random Forest feature importances
│   ├── roc_curve_data.json            # ROC curve coordinates (FPR / TPR)
│   ├── model_predictions_for_tableau.csv    # Primary Tableau data feed
│   ├── business_summary.md            # 1-2 page executive business summary report
│   └── executive_presentation_outline.md    # 8-slide executive presentation outline
└── README.md                          # Project documentation and execution guide
```

---

## 🚀 Quick Start & Execution

The entire analytics pipeline can be executed end-to-end using Python.

### Prerequisites
Ensure Python 3.8+ is installed along with required packages:
```bash
pip install pandas numpy scikit-learn scipy
```

### End-to-End Execution Command

#### Windows Command Prompt / PowerShell:
```cmd
cmd /c "python data/generate_data.py && python notebooks/01_eda_cleaning.py && python notebooks/02_modeling.py && python build_dashboard_data.py"
```

#### Linux / macOS Bash:
```bash
python data/generate_data.py && python notebooks/01_eda_cleaning.py && python notebooks/02_modeling.py && python build_dashboard_data.py
```

---

## 📊 Analytical Methodology & High Accuracy Optimization

### Step 1: Synthetic Dataset Calibration (`data/generate_data.py`)
Generates 1,500 customer records with realistic banking distributions and optimized decision signals:
- **Correlations Calibrated**: `Missed_Payments` (+0.706), `Credit_Utilization` (+0.447), `Debt_to_Income_Ratio` (+0.454).
- **Target Base Rate**: Calibrated via standardized risk index cutoff to achieve **16.27% base delinquency rate**.
- **Anomalies & Missingness**: Injects 8 income anomalies ($0, $1, >$500k), 10 low credit score anomalies (<420 paired with above-median income), and random missingness (`Income` ~5.5%, `Loan_Balance` ~5.8%, `Credit_Score` ~3.2%).

### Step 2: EDA & Data Cleaning (`notebooks/01_eda_cleaning.py`)
- Profiles missing values and logs anomaly count.
- Flags unrealistic income as `NaN`, imputes `Income` and `Loan_Balance` with medians.
- Drops missing `Credit_Score` rows (logging exact % dropped).
- Recomputes `Debt_to_Income_Ratio` post-imputation.
- Engineers `Risk_Tier` buckets from `Credit_Utilization` (`Low`: <30%, `Moderate`: 30-50%, `High`: 50-75%, `Very High`: >=75%).
- Exports [`outputs/eda_summary.txt`](file:///e:/geldium_project/outputs/eda_summary.txt) and [`data/geldium_customers_clean.csv`](file:///e:/geldium_project/data/geldium_customers_clean.csv).

### Step 3: Predictive Modeling & Accuracy Optimization (`notebooks/02_modeling.py`)
- Evaluates **Logistic Regression** (Primary score) vs **Random Forest** (Benchmark) on a 75/25 stratified split.
- **Model Metrics (Test Set)**:
  - **Logistic Regression (Primary)**: ROC-AUC = **0.9989 (~100%)** | Accuracy = **97.52%** | **Recall = 96.67%** | Precision = **89.23%**
  - **Random Forest (Benchmark)**: ROC-AUC = **0.9926 (~100%)** | Accuracy = **95.59%** | Precision = **89.29%** | Recall = **83.33%**
- **Recall Priority**: Prioritizes Recall over raw accuracy because false negatives (unflagged defaults) carry severe credit write-off costs ($14,200 average balance per defaulted account).

### Step 4: Enterprise SAS Port (`sas/geldium_risk_scoring.sas`)
Provides production-ready SAS script for SAS OnDemand for Academics / SAS Enterprise Guide:
- `PROC LOGISTIC` to replicate risk scoring model and output predicted default probabilities (`P_DELINQUENT`).
- `PROC FREQ` crosstabulations for vintage/roll-rate risk segment analysis.

### Step 5: Live Interactive Dashboard (`dashboard/index.html`)
- Accessible at **http://localhost:8050**
- Features 4 KPI Summary Cards, Interactive Risk Decision Threshold Simulator, 5 Dynamic Charts (ROC curves overlay, Risk Tier distribution, Expected Loss breakdown), and Customer Risk Explorer Table.

---

## 📈 Key Findings Summary
1. **Missed Payments & High Utilization** are the dominant predictors of credit risk.
2. **Unemployed Segment Vulnerability**: Default rate reaches **69.84%** vs 10.35% for employed borrowers.
3. **Financial Impact**: The optimized primary model identifies **96.67% of all defaulting accounts in advance**, protecting **$3.35 Million in potential default exposure**.
