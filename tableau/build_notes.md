# Tableau Dashboard Build Notes: Geldium Financial Risk Analytics

## 1. Overview & Data Source Integration
This document provides complete instructions for building the **Geldium Financial Credit Delinquency Risk Dashboard** in Tableau Desktop or Tableau Public.

- **Primary Data Feed**: [`outputs/model_predictions_for_tableau.csv`](file:///e:/geldium_project/outputs/model_predictions_for_tableau.csv) (also available at [`tableau/model_predictions_for_tableau.csv`](file:///e:/geldium_project/tableau/model_predictions_for_tableau.csv))
- **Granularity**: 1 record per customer (`Customer_ID`)
- **Key Fields in Dataset**:
  - `Customer_ID`: Unique Identifier
  - `Age`, `Income`, `Credit_Score`, `Credit_Utilization`, `Missed_Payments`, `Loan_Balance`, `Debt_to_Income_Ratio`, `Account_Tenure_Years`
  - `Employment_Status`, `Account_Type`, `Location`, `Risk_Tier`
  - `Actual_Delinquent`: Binary indicator (1 = Default/Delinquent, 0 = Current)
  - `Prob_Logistic`: Predicted probability of default from Logistic Regression model
  - `Prob_RF`: Predicted probability of default from Random Forest model
  - `Expected_Loss_Logistic`: `Loan_Balance * Prob_Logistic`
  - `Expected_Loss_RF`: `Loan_Balance * Prob_RF`

---

## 2. Calculated Fields & Parameters to Create in Tableau

### Calculated Fields
1. **Delinquency Status Label**:
   ```tableau
   IF [Actual_Delinquent] = 1 THEN "Delinquent" ELSE "Current" END
   ```
2. **Dynamic Risk Threshold Flag** (Controlled by Parameter):
   ```tableau
   IF [Prob_Logistic] >= [Probability Threshold Parameter] THEN "High Risk" ELSE "Normal Risk" END
   ```
3. **Expected Loss ($)**:
   ```tableau
   [Loan_Balance] * [Prob_Logistic]
   ```
4. **Actual Default Loss Exposure ($)**:
   ```tableau
   IF [Actual_Delinquent] = 1 THEN [Loan_Balance] ELSE 0 END
   ```

### Parameter
- **Probability Threshold Parameter**:
  - Data Type: Float
  - Range: 0.10 to 0.90, Step: 0.05, Default: 0.30

---

## 3. Dashboard Visualizations (4-Pane Executive Layout)

### Sheet 1: Risk-Tier Distribution & Credit Exposure (Bar Chart + Dual Axis)
- **Goal**: Display customer count and total loan balance across `Risk_Tier` buckets (`Low`, `Moderate`, `High`, `Very High`).
- **Columns**: `Risk_Tier` (Sorted: Low → Moderate → High → Very High)
- **Rows**: 
  - Measure 1: `CNT(Customer_ID)` (Bar Chart)
  - Measure 2: `SUM(Loan_Balance)` (Line Chart overlay or Dual Axis)
- **Color**: `Risk_Tier` using a gradient scale (Low = Teal `#2E7D32`, High/Very High = Coral/Red `#D32F2F`).
- **Tooltip**: Include % of total portfolio and average credit score per tier.

### Sheet 2: Expected Loss Heatmap Matrix (`Risk_Tier` × `Loan_Balance Bins`)
- **Goal**: Highlight financial capital at risk across loan exposure sizes.
- **Columns**: `Loan_Balance (Bin)` (Bin size: $5,000)
- **Rows**: `Risk_Tier`
- **Color Marks**: `SUM(Expected_Loss_Logistic)` (Darker red indicating concentrated dollar risk).
- **Text Marks**: `SUM(Expected_Loss_Logistic)` formatted as `$#,##0`.

### Sheet 3: Geographic & Employment Vulnerability Grid
- **Goal**: Drill into demographic pockets of delinquency risk.
- **Columns**: `Location`
- **Rows**: `Employment_Status`
- **Color Marks**: `AVG(Actual_Delinquent)` formatted as Percentage `0.0%`.
- **Annotation**: Highlight Unemployed segment in Los Angeles and Atlanta showing elevated default rates (>40%).

### Sheet 4: Model Performance & ROC Curve Overlay
- **Goal**: Executive comparison of Logistic Regression vs. Random Forest risk calibration.
- **Data Source**: Connect `outputs/roc_curve_data.json` or create plot using `Prob_Logistic` vs `Prob_RF` deciles.
- **Columns**: `False Positive Rate (FPR)`
- **Rows**: `True Positive Rate (TPR)`
- **Detail**: Dual line graph overlaying Logistic Regression (Solid Blue) and Random Forest (Dashed Orange), with a diagonal 45° reference line representing baseline random guessing.

---

## 4. Executive KPI Summary Cards (Header Banner)
Place 4 high-level KPI cards across the top of the dashboard:
1. **Total Portfolio Balance**: `SUM(Loan_Balance)` (e.g. `$20.7M`)
2. **Overall Default Rate**: `AVG(Actual_Delinquent)` (e.g. `17.3%`)
3. **Total Expected Credit Loss**: `SUM(Expected_Loss_Logistic)` (e.g. `$3.8M`)
4. **Model Recall Rate**: `67.7%` (at 0.30 decision cutoff)

---

## 5. Styling & Formatting Guidelines
- **Color Palette**: Financial Executive Theme (Dark Navy `#1A237E` headers, Slate gray background `#F5F7FA`, Accent Red `#C62828` for risk highlights).
- **Typography**: Arial / Segoe UI, clean bold headers.
- **Interactivity**: 
  - Add Global Filter on `Location` and `Account_Type`.
  - Add `Probability Threshold Parameter` slider to dynamically filter customers flagged for intervention.
