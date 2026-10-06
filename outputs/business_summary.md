# Geldium Financial — Credit Delinquency Risk Analytics
## Executive Business Summary & Risk Strategy Report

### 1. Executive Context & Objectives
Geldium Financial commissioned this risk analytics engagement to develop a machine-learning-driven credit scoring framework. The objective is to identify retail and commercial customers at elevated risk of loan default (90+ days past due), quantify financial exposure across risk tiers, and implement targeted intervention protocols to reduce net credit write-offs.

---

### 2. Portfolio Risk Landscape & Key Findings
Analysis of the customer portfolio (1,452 active accounts) revealed an overall base delinquency rate of **17.27%**, representing **$3.58M in total loan exposure at risk**. 

The investigation identified three primary drivers of credit default:
1. **Missed Payments (Correlation: +0.345, Odds Ratio: 1.73)**: Prior payment defaults are the strongest single predictor of future delinquency. Each additional missed payment increases the odds of default by **73.1%**.
2. **Credit Utilization (Correlation: +0.304, Odds Ratio: 1.79)**: Customers utilizing over 75% of available credit lines exhibit a **51.06% default rate**, compared to just **6.79%** for low-utilization accounts (<30%).
3. **Employment & Income Vulnerability**: Unemployed account holders suffer a **46.03% delinquency rate** (nearly 3x the portfolio average). Additionally, high Debt-to-Income (DTI) ratios (>0.45) compound default probability significantly.

#### Data Quality & Anomaly Discovery
The EDA pipeline flagged critical data-entry anomalies:
- 8 records containing zero, $1, or extreme erroneous incomes (>$500k).
- 10 records with severely depressed credit scores (<420) despite above-median incomes, signaling potential identity theft or reporting errors.
All anomalies were cleaned via median imputation and standardized filters prior to model training.

---

### 3. Predictive Model Performance in Business Terms
Two models were evaluated on a 25% holdout test set (363 accounts):
- **Logistic Regression (Primary Scorecard)**: Chosen for full regulatory transparency and auditability.
- **Random Forest (Benchmark Model)**: Used to evaluate non-linear pattern capture.

| Model | Accuracy | Precision | **Recall (Sensitivity)** | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **74.38%** | **36.52%** | **67.74%** | **0.4746** | **0.7951** |
| Random Forest (Benchmark) | 79.34% | 42.17% | 56.45% | 0.4828 | 0.7783 |

#### Why Recall Matters Most in Banking
In credit risk modeling, a **False Negative** (failing to flag a customer who ultimately defaults) causes direct write-off losses of principal ($14,200 average balance per loan). Conversely, a **False Positive** (flagging a safe customer) carries a minor operational cost of manual review ($50-$100). 

Logistic Regression achieved a **67.74% Recall rate**, successfully identifying **over two-thirds of all true delinquent accounts** prior to default. In business terms:
> **"The primary scoring model correctly flags 67.7% of defaulting accounts in advance, enabling Geldium Financial to proactively protect $2.42M in loan principal."**

---

### 4. Strategic Risk-Tier Intervention Framework

Based on model probability outputs (`Prob_Logistic`), customers are segmented into four actionable risk tiers:

```
+-----------------------------------------------------------------------------------+
| RISK TIER   | UTILIZATION | DEFAULT RATE | RECOMMENDED INTERVENTION PROTOCOL     |
+-----------------------------------------------------------------------------------+
| Low         | < 30%       | 6.79%        | Automated line increases & pre-approved|
|             |             |              | cross-sell products.                  |
+-----------------------------------------------------------------------------------+
| Moderate    | 30% - <50%  | 17.80%       | Soft digital alerts, automated SMS   |
|             |             |              | payment reminders 5 days prior to due.|
+-----------------------------------------------------------------------------------+
| High        | 50% - <75%  | 30.77%       | Temporary credit line freeze, proactive|
|             |             |              | hardship outreach, financial counsel. |
+-----------------------------------------------------------------------------------+
| Very High   | >= 75%      | 51.06%       | Immediate credit cap, mandatory debt  |
|             |             |              | restructuring / workout plans.        |
+-----------------------------------------------------------------------------------+
```

---

### 5. Financial ROI & Next Steps
1. **Expected Loss Reduction**: Implementing the 0.30 probability threshold intervention protocol is projected to lower annual credit losses by **$620,000 - $850,000**.
2. **SAS Infrastructure Deployment**: Execute [`sas/geldium_risk_scoring.sas`](file:///e:/geldium_project/sas/geldium_risk_scoring.sas) in SAS OnDemand for monthly batch scoring.
3. **Tableau Dashboard Monitor**: Deploy [`tableau/build_notes.md`](file:///e:/geldium_project/tableau/build_notes.md) dashboard to provide C-suite visibility into real-time portfolio risk drift.
