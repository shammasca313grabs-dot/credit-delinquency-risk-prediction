# Geldium Financial — Credit Risk Executive Presentation
## Slide Deck Structure & Content Outline (8 Slides)

---

### SLIDE 1: Title & Executive Summary
- **Title**: Geldium Financial: Credit Delinquency Risk Prediction & Portfolio Analytics
- **Subtitle**: Machine Learning Scorecard & Portfolio Vulnerability Analysis
- **Presenter**: Credit Risk Analytics Engagement Team
- **Key Takeaway**: A logistic regression scorecard achieves **0.795 ROC-AUC** and **67.7% Recall**, providing early warning on **$2.4M in potential default losses** and enabling targeted mitigation across 4 risk tiers.

---

### SLIDE 2: Problem Statement & Portfolio Landscape
- **Context**: Rising macroeconomic pressures require proactive identification of credit default risk before accounts hit 90+ DPD.
- **Portfolio Snapshot**:
  - Total Active Accounts Analyzed: 1,452 customers
  - Total Loan Exposure: $20.7 Million
  - Baseline Portfolio Delinquency Rate: **17.27%**
- **Business Challenge**: Traditional credit scores miss early signals of distress caused by rapid utilization spikes and employment shifts.

---

### SLIDE 3: Top Risk Drivers & Data Insights
- **Key Risk Drivers (Strongest Statistical Impact)**:
  1. **Credit Utilization** (Correlation: **+0.304** | Odds Ratio: **1.79**): Accounts with >75% utilization default at **51.1%** vs 6.8% for <30% utilization.
  2. **Missed Payments** (Correlation: **+0.345** | Odds Ratio: **1.73**): Past payment delinquency increases future default odds by **73%** per incident.
  3. **Employment Vulnerability**: Unemployed borrowers exhibit a **46.03% delinquency rate** vs 15.03% for employed borrowers.
- **Data Quality Integrity**: Cleaned 90 income anomalies and 48 missing credit score records prior to modeling.

---

### SLIDE 4: Predictive Modeling Approach & Recall Focus
- **Methodology**: Evaluated Logistic Regression (Primary) against Random Forest (Benchmark) on a 75/25 stratified split.
- **Why Recall is Priority #1**:
  - **Cost of False Negative** (Unflagged Default): **~$14,200** loan write-off.
  - **Cost of False Positive** (Unnecessary Review): **~$75** operational cost.
  - **Strategy**: Calibrated decision threshold to optimize Recall over raw accuracy.

---

### SLIDE 5: Model Evaluation & Performance Benchmark
- **Model Comparison Table**:
  - **Logistic Regression**: ROC-AUC: **0.7951** | Recall: **67.74%** | Accuracy: **74.38%**
  - **Random Forest**: ROC-AUC: **0.7783** | Recall: **56.45%** | Accuracy: **79.34%**
- **Recommendation**: Adopt **Logistic Regression** as the core scoring model due to superior Recall (67.7% vs 56.5%) and full regulatory auditability / explainability.

---

### SLIDE 6: Risk-Tier Portfolio Segmentation
- **4-Tier Risk Matrix**:
  - **Low Risk (<30% Util)**: 574 accounts (39.5% of portfolio) | Default Rate: **6.79%**
  - **Moderate Risk (30-50% Util)**: 545 accounts (37.5% of portfolio) | Default Rate: **17.80%**
  - **High Risk (50-75% Util)**: 286 accounts (19.7% of portfolio) | Default Rate: **30.77%**
  - **Very High Risk (>=75% Util)**: 47 accounts (3.2% of portfolio) | Default Rate: **51.06%**
- **Expected Financial Loss**: Total portfolio expected loss is calculated at **$3.8 Million**.

---

### SLIDE 7: Recommended Intervention Strategy & Business Impact
- **Tiered Action Plan**:
  - **Low Tier**: Automated credit limit increases & cross-selling.
  - **Moderate Tier**: Automated SMS payment reminders 5 days prior to due dates.
  - **High Tier**: Proactive hardship outreach & temporary line lock.
  - **Very High Tier**: Mandatory workout plans & loan restructuring.
- **Financial Return on Investment**: Projected reduction of annual net write-offs by **$620k - $850k**.

---

### SLIDE 8: Deployment Roadmap & Next Steps
1. **SAS Scoring Integration**: Monthly automated execution via `sas/geldium_risk_scoring.sas`.
2. **Tableau C-Suite Dashboard**: Live tracking using `tableau/build_notes.md` specification.
3. **Quarterly Recalibration**: Monitor feature drift and retrain weights semi-annually.
