import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def test_data(n_samples=1500, random_state=42):
    np.random.seed(random_state)
    customer_ids = [f"GELD-{10000 + i}" for i in range(n_samples)]
    
    emp_categories = ['Employed', 'Self-Employed', 'Unemployed', 'Retired']
    employment_status = np.random.choice(emp_categories, size=n_samples, p=[0.62, 0.14, 0.09, 0.15])
    account_type = np.random.choice(['Personal', 'Business'], size=n_samples, p=[0.78, 0.22])
    location = np.random.choice(['Los Angeles', 'New York', 'Chicago', 'Houston', 'Atlanta', 'Seattle', 'Denver'], size=n_samples)
    
    age = np.random.randint(21, 72, size=n_samples)
    account_tenure = np.round(np.random.uniform(0.5, 20.0, size=n_samples), 1)
    
    income = np.exp(np.random.normal(loc=11.1, scale=0.45, size=n_samples))
    unemployed_mask = (employment_status == 'Unemployed')
    retired_mask = (employment_status == 'Retired')
    income[unemployed_mask] *= np.random.uniform(0.25, 0.45, size=unemployed_mask.sum())
    income[retired_mask] *= np.random.uniform(0.50, 0.70, size=retired_mask.sum())
    income = np.round(income, 2)
    
    credit_score = np.clip(np.random.normal(loc=675, scale=70, size=n_samples), 300, 850)
    
    base_util = np.random.beta(a=2.0, b=3.5, size=n_samples) * 100.0
    util_boost = np.where(unemployed_mask, np.random.uniform(10.0, 18.0, size=n_samples), 0.0)
    credit_utilization = np.clip(base_util + util_boost, 0.0, 100.0)
    credit_utilization = np.round(credit_utilization, 2)
    
    lambda_param = 0.10 + (credit_utilization / 100.0) * 3.0 + unemployed_mask * 1.5
    missed_payments = np.random.poisson(lam=lambda_param)
    
    loan_balance = np.exp(np.random.normal(loc=9.4, scale=0.6, size=n_samples))
    annual_debt = loan_balance * np.random.uniform(0.15, 0.28, size=n_samples) + (credit_utilization * 110)
    dti_ratio = np.clip(annual_debt / (income + 1e-5), 0.05, 0.85)
    dti_ratio = np.round(dti_ratio, 3)
    
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
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    probs = sigmoid(z_index * 15.0 - 13.5)
    delinquent_account = (probs >= 0.50).astype(int)
    
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
    
    return df

df = test_data()
print("Base rate:", round(df['Delinquent_Account'].mean(), 4))
print("Corr Missed:", round(df['Missed_Payments'].corr(df['Delinquent_Account']), 4))
print("Corr Util:", round(df['Credit_Utilization'].corr(df['Delinquent_Account']), 4))
print("Corr DTI:", round(df['Debt_to_Income_Ratio'].corr(df['Delinquent_Account']), 4))

X = df.drop(columns=['Customer_ID', 'Delinquent_Account'])
y = df['Delinquent_Account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

numeric_features = ['Age', 'Income', 'Credit_Score', 'Credit_Utilization',
                    'Missed_Payments', 'Loan_Balance', 'Debt_to_Income_Ratio',
                    'Account_Tenure_Years']
categorical_features = ['Employment_Status', 'Account_Type', 'Location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

pipe_lr = Pipeline([
    ('prep', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr', LogisticRegression(max_iter=2000, C=100.0, random_state=42))
])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]

print("\n--- LOGISTIC REGRESSION WITH POLYNOMIAL FEATURES ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

pipe_gb = Pipeline([
    ('prep', preprocessor),
    ('gb', GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42))
])
pipe_gb.fit(X_train, y_train)
y_pred_gb = pipe_gb.predict(X_test)
y_prob_gb = pipe_gb.predict_proba(X_test)[:, 1]

print("\n--- GRADIENT BOOSTING CLASSIFIER ---")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Precision:", precision_score(y_test, y_pred_gb))
print("Recall:", recall_score(y_test, y_pred_gb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_gb))
