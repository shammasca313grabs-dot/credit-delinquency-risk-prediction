import os
import numpy as np
import pandas as pd

def generate_dataset(n_samples=1500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Customer IDs
    customer_ids = [f"GELD-{10000 + i}" for i in range(n_samples)]
    
    # 2. Categoricals
    emp_categories = ['Employed', 'Self-Employed', 'Unemployed', 'Retired']
    emp_probs = [0.62, 0.14, 0.09, 0.15]
    employment_status = np.random.choice(emp_categories, size=n_samples, p=emp_probs)
    
    acct_categories = ['Personal', 'Business']
    acct_probs = [0.78, 0.22]
    account_type = np.random.choice(acct_categories, size=n_samples, p=acct_probs)
    
    locations = ['Los Angeles', 'New York', 'Chicago', 'Houston', 'Atlanta', 'Seattle', 'Denver']
    location = np.random.choice(locations, size=n_samples)
    
    # 3. Numeric base features
    age = np.random.randint(21, 72, size=n_samples)
    account_tenure = np.round(np.random.uniform(0.5, 20.0, size=n_samples), 1)
    
    # Income: lognormal distribution (~$25k-$150k typical), reduced for Unemployed/Retired
    income = np.exp(np.random.normal(loc=11.1, scale=0.45, size=n_samples))
    unemployed_mask = (employment_status == 'Unemployed')
    retired_mask = (employment_status == 'Retired')
    income[unemployed_mask] *= np.random.uniform(0.25, 0.45, size=unemployed_mask.sum())
    income[retired_mask] *= np.random.uniform(0.50, 0.70, size=retired_mask.sum())
    income = np.round(income, 2)
    
    # Credit Score: normal mean ~675, std ~70, clipped 300-850
    credit_score = np.random.normal(loc=675, scale=70, size=n_samples)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Credit Utilization: 0-100%, right-skewed (beta distribution)
    base_util = np.random.beta(a=2.0, b=3.5, size=n_samples) * 100.0
    util_boost = np.where(unemployed_mask, np.random.uniform(10.0, 18.0, size=n_samples), 0.0)
    credit_utilization = np.clip(base_util + util_boost, 0.0, 100.0)
    credit_utilization = np.round(credit_utilization, 2)
    
    # Missed Payments: Poisson distributed, rate scales with utilization and unemployment
    lambda_param = 0.10 + (credit_utilization / 100.0) * 3.0 + unemployed_mask * 1.5
    missed_payments = np.random.poisson(lam=lambda_param)
    
    # Loan Balance
    loan_balance = np.exp(np.random.normal(loc=9.4, scale=0.6, size=n_samples))
    loan_balance = np.round(loan_balance, 2)
    
    # Debt to Income Ratio
    annual_debt = loan_balance * np.random.uniform(0.15, 0.28, size=n_samples) + (credit_utilization * 110)
    dti_ratio = np.clip(annual_debt / (income + 1e-5), 0.05, 0.85)
    dti_ratio = np.round(dti_ratio, 3)
    
    # 4. Target Delinquent_Account via standardized linear risk index (100% separable signal)
    is_unemployed = (employment_status == 'Unemployed').astype(float)
    is_business = (account_type == 'Business').astype(float)
    is_la = (location == 'Los Angeles').astype(float)
    score_dev = credit_score - 650.0
    
    raw_index = (
        0.060 * credit_utilization +
        2.10 * missed_payments -
        0.012 * score_dev +
        0.60 * is_unemployed +
        0.50 * is_business +
        0.40 * is_la +
        4.50 * dti_ratio -
        0.04 * account_tenure
    )
    
    z_index = (raw_index - np.mean(raw_index)) / np.std(raw_index)
    
    # Calibrate sharp cutoff for ~17-19% base default rate
    delinquent_account = (z_index >= 0.95).astype(int)
    
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': age,
        'Income': income,
        'Employment_Status': employment_status,
        'Credit_Score': credit_score,
        'Credit_Utilization': credit_utilization,
        'Missed_Payments': missed_payments,
        'Loan_Balance': loan_balance,
        'Debt_to_Income_Ratio': dti_ratio,
        'Account_Type': account_type,
        'Location': location,
        'Account_Tenure_Years': account_tenure,
        'Delinquent_Account': delinquent_account
    })
    
    # Inject anomalies before missingness
    # 1. Income anomalies (~8 values: 0, 1, or >$500k)
    anomaly_income_idx = np.random.choice(n_samples, size=8, replace=False)
    df.loc[anomaly_income_idx[:3], 'Income'] = 0.0
    df.loc[anomaly_income_idx[3:5], 'Income'] = 1.0
    df.loc[anomaly_income_idx[5:], 'Income'] = [550000.0, 720000.0, 950000.0]
    
    # 2. Credit_Score anomalies (~10 low scores 300-420 paired with above-median income)
    median_inc = df[df['Income'] > 1.0]['Income'].median()
    high_inc_indices = df[df['Income'] > median_inc].index.values
    anomaly_cs_idx = np.random.choice(high_inc_indices, size=10, replace=False)
    df.loc[anomaly_cs_idx, 'Credit_Score'] = np.random.uniform(300, 420, size=10)
    df['Credit_Score'] = np.round(df['Credit_Score'], 0)
    
    # Inject missingness (MCAR)
    # Income ~5.5% (~82 rows)
    inc_missing_idx = np.random.choice(df.index, size=int(0.055 * n_samples), replace=False)
    df.loc[inc_missing_idx, 'Income'] = np.nan
    
    # Loan_Balance ~5.8% (~87 rows)
    lb_missing_idx = np.random.choice(df.index, size=int(0.058 * n_samples), replace=False)
    df.loc[lb_missing_idx, 'Loan_Balance'] = np.nan
    
    # Credit_Score ~3.2% (~48 rows)
    cs_missing_idx = np.random.choice(df.index, size=int(0.032 * n_samples), replace=False)
    df.loc[cs_missing_idx, 'Credit_Score'] = np.nan
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_samples=1500, random_state=42)
    output_path = "data/geldium_customers_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records -> {output_path}")
    
    valid_df = df.dropna()
    print(f"Base delinquency rate: {df['Delinquent_Account'].mean():.2%}")
    print("Correlations with Delinquent_Account:")
    for col in ['Missed_Payments', 'Credit_Utilization', 'Debt_to_Income_Ratio', 'Credit_Score', 'Account_Tenure_Years']:
        print(f"  {col}: {valid_df[col].corr(valid_df['Delinquent_Account']):.4f}")
    
    emp_rates = df.groupby('Employment_Status')['Delinquent_Account'].mean()
    print("Delinquency rate by Employment_Status:")
    for emp, rate in emp_rates.items():
        print(f"  {emp}: {rate:.2%}")
