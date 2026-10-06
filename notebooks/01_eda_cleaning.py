import os
import numpy as np
import pandas as pd

def run_eda_and_cleaning(raw_csv_path="data/geldium_customers_raw.csv",
                         clean_csv_path="data/geldium_customers_clean.csv",
                         report_path="outputs/eda_summary.txt"):
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(os.path.dirname(clean_csv_path), exist_ok=True)
    
    report_lines = []
    def log(text=""):
        print(text)
        report_lines.append(text)
        
    log("================================================================================")
    log("GELDIUM FINANCIAL — CREDIT DELINQUENCY RISK ANALYTICS: EDA & CLEANING REPORT")
    log("================================================================================\n")
    
    df_raw = pd.read_csv(raw_csv_path)
    total_raw_rows = len(df_raw)
    log(f"Raw Dataset Loaded: {total_raw_rows} customer records.")
    log(f"Columns: {list(df_raw.columns)}\n")
    
    # 1. Profile Missing Values
    log("--- 1. MISSING VALUE PROFILE (RAW DATA) ---")
    missing_summary = pd.DataFrame({
        'Missing_Count': df_raw.isnull().sum(),
        'Missing_Pct': (df_raw.isnull().sum() / total_raw_rows * 100).round(2)
    })
    log(missing_summary.to_string())
    log("\n")
    
    # 2. Detect Anomalies
    log("--- 2. ANOMALY DETECTION ---")
    # Unrealistic Income: <= 1 or > 500k
    unrealistic_income_mask = (df_raw['Income'] <= 1.0) | (df_raw['Income'] > 500000.0)
    anomalous_income_count = unrealistic_income_mask.sum()
    log(f"Unrealistic Income Records (Income <= $1 or > $500,000): {anomalous_income_count}")
    if anomalous_income_count > 0:
        log("Sample anomalous income values:")
        log(str(df_raw.loc[unrealistic_income_mask, ['Customer_ID', 'Income', 'Employment_Status']].head(10)))
    
    # Low Credit Score (< 450) paired with above-median Income
    median_raw_income = df_raw.loc[~unrealistic_income_mask, 'Income'].median()
    low_cs_high_inc_mask = (df_raw['Credit_Score'] < 450) & (df_raw['Income'] > median_raw_income)
    anomalous_cs_count = low_cs_high_inc_mask.sum()
    log(f"\nLow Credit Score (< 450) paired with above-median Income (>$ {median_raw_income:,.2f}): {anomalous_cs_count}")
    if anomalous_cs_count > 0:
        log("Sample low credit score anomalies:")
        log(str(df_raw.loc[low_cs_high_inc_mask, ['Customer_ID', 'Credit_Score', 'Income']].head(10)))
    log("\n")
    
    # 3. Descriptive Statistics & Delinquency Rates
    log("--- 3. DESCRIPTIVE STATISTICS & SEGMENT DELINQUENCY RATES ---")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    log("Numeric Features Summary Statistics:")
    log(df_raw[numeric_cols].describe().round(2).to_string())
    log("\n")
    
    log("Overall Base Delinquency Rate: {:.2f}%".format(df_raw['Delinquent_Account'].mean() * 100))
    log("\nDelinquency Rate by Employment Status:")
    emp_delinq = df_raw.groupby('Employment_Status')['Delinquent_Account'].agg(['count', 'mean'])
    emp_delinq['mean'] = (emp_delinq['mean'] * 100).round(2)
    emp_delinq.columns = ['Count', 'Delinquency_Rate_%']
    log(emp_delinq.to_string())
    
    log("\nDelinquency Rate by Account Type:")
    acct_delinq = df_raw.groupby('Account_Type')['Delinquent_Account'].agg(['count', 'mean'])
    acct_delinq['mean'] = (acct_delinq['mean'] * 100).round(2)
    acct_delinq.columns = ['Count', 'Delinquency_Rate_%']
    log(acct_delinq.to_string())
    
    log("\nDelinquency Rate by Location:")
    loc_delinq = df_raw.groupby('Location')['Delinquent_Account'].agg(['count', 'mean'])
    loc_delinq['mean'] = (loc_delinq['mean'] * 100).round(2)
    loc_delinq.columns = ['Count', 'Delinquency_Rate_%']
    log(loc_delinq.to_string())
    log("\n")
    
    # 4. Target Correlations
    log("--- 4. NUMERIC FEATURE CORRELATION WITH DELINQUENT_ACCOUNT ---")
    corrs = df_raw[numeric_cols].corr()['Delinquent_Account'].sort_values(ascending=False)
    log(corrs.round(4).to_string())
    log("\n")
    
    # 5. Data Cleaning Actions
    log("--- 5. DATA CLEANING & IMPUTATION ACTIONS ---")
    df_clean = df_raw.copy()
    
    # Action 5a: Flag anomalous Income as NaN
    df_clean.loc[unrealistic_income_mask, 'Income'] = np.nan
    log(f"Action 5a: Flagged {anomalous_income_count} anomalous Income records as NaN.")
    
    # Action 5b: Median Imputation for Income and Loan_Balance
    median_income = df_clean['Income'].median()
    median_loan = df_clean['Loan_Balance'].median()
    
    income_imputed_count = df_clean['Income'].isnull().sum()
    loan_imputed_count = df_clean['Loan_Balance'].isnull().sum()
    
    df_clean['Income'] = df_clean['Income'].fillna(median_income)
    df_clean['Loan_Balance'] = df_clean['Loan_Balance'].fillna(median_loan)
    
    log(f"Action 5b: Median-imputed Income ({income_imputed_count} records imputed with ${median_income:,.2f}).")
    log(f"Action 5b: Median-imputed Loan_Balance ({loan_imputed_count} records imputed with ${median_loan:,.2f}).")
    
    # Action 5c: Drop rows with missing Credit_Score
    missing_cs_count = df_clean['Credit_Score'].isnull().sum()
    pct_dropped = (missing_cs_count / total_raw_rows * 100)
    df_clean = df_clean.dropna(subset=['Credit_Score']).copy()
    log(f"Action 5c: Dropped {missing_cs_count} rows ({pct_dropped:.2f}% of raw dataset) due to missing Credit_Score.")
    
    # Action 5d: Recompute Debt_to_Income_Ratio post-imputation
    # Re-estimating annual debt obligations based on imputed Loan_Balance and Credit_Utilization
    estimated_annual_debt = df_clean['Loan_Balance'] * 0.20 + (df_clean['Credit_Utilization'] * 90)
    df_clean['Debt_to_Income_Ratio'] = np.clip(estimated_annual_debt / df_clean['Income'], 0.05, 0.85).round(3)
    log("Action 5d: Recomputed Debt_to_Income_Ratio post-imputation based on clean Income & Loan_Balance.")
    
    # 6. Feature Engineering: Risk_Tier Buckets
    log("\n--- 6. FEATURE ENGINEERING: RISK_TIER BUCKETS ---")
    def assign_risk_tier(util):
        if util < 30.0:
            return 'Low'
        elif util < 50.0:
            return 'Moderate'
        elif util < 75.0:
            return 'High'
        else:
            return 'Very High'
            
    df_clean['Risk_Tier'] = df_clean['Credit_Utilization'].apply(assign_risk_tier)
    tier_order = ['Low', 'Moderate', 'High', 'Very High']
    df_clean['Risk_Tier'] = pd.Categorical(df_clean['Risk_Tier'], categories=tier_order, ordered=True)
    
    log("Cleaned Dataset Risk_Tier Distribution & Delinquency Rates:")
    tier_summary = df_clean.groupby('Risk_Tier', observed=False)['Delinquent_Account'].agg(['count', 'mean'])
    tier_summary['mean'] = (tier_summary['mean'] * 100).round(2)
    tier_summary.columns = ['Count', 'Delinquency_Rate_%']
    log(tier_summary.to_string())
    log("\n")
    
    log(f"Final Cleaned Dataset Size: {len(df_clean)} records ({len(df_clean)/total_raw_rows*100:.2f}% retained).")
    
    # Save Cleaned CSV
    df_clean.to_csv(clean_csv_path, index=False)
    log(f"Saved cleaned dataset to: {clean_csv_path}")
    
    # Save EDA Summary Text Report
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved text EDA summary report to: {report_path}")

if __name__ == "__main__":
    run_eda_and_cleaning()
