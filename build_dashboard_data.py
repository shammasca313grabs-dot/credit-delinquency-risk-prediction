import os
import json
import pandas as pd

def build_dashboard_data():
    os.makedirs("dashboard", exist_ok=True)
    
    # 1. Load Scored Predictions
    df = pd.read_csv("outputs/model_predictions_for_tableau.csv")
    
    # 2. Load Model Results & Metrics
    with open("outputs/model_results.json", "r") as f:
        model_results = json.load(f)
        
    with open("outputs/roc_curve_data.json", "r") as f:
        roc_data = json.load(f)
        
    with open("outputs/logistic_coefficients.json", "r") as f:
        logistic_coefs = json.load(f)
        
    with open("outputs/random_forest_feature_importance.json", "r") as f:
        rf_importance = json.load(f)
        
    # 3. Aggregations for Charts
    # Chart 1: Risk Tier Breakdown
    risk_tier_agg = df.groupby('Risk_Tier', observed=False).agg(
        Count=('Customer_ID', 'count'),
        Default_Count=('Actual_Delinquent', 'sum'),
        Default_Rate=('Actual_Delinquent', lambda x: round(x.mean() * 100, 2)),
        Total_Loan=('Loan_Balance', 'sum'),
        Avg_Income=('Income', 'mean'),
        Expected_Loss=('Expected_Loss_Logistic', 'sum')
    ).reset_index()
    
    # Chart 2: Employment Status Breakdown
    emp_agg = df.groupby('Employment_Status').agg(
        Count=('Customer_ID', 'count'),
        Default_Rate=('Actual_Delinquent', lambda x: round(x.mean() * 100, 2)),
        Avg_Util=('Credit_Utilization', 'mean'),
        Total_Loan=('Loan_Balance', 'sum')
    ).reset_index()
    
    # Chart 3: Location Breakdown
    loc_agg = df.groupby('Location').agg(
        Count=('Customer_ID', 'count'),
        Default_Rate=('Actual_Delinquent', lambda x: round(x.mean() * 100, 2)),
        Total_Loan=('Loan_Balance', 'sum')
    ).reset_index()
    
    # Chart 4: Location x Employment Heatmap Grid
    grid_agg = df.groupby(['Location', 'Employment_Status'])['Actual_Delinquent'].mean().unstack().fillna(0).round(4)
    grid_data = {
        'locations': list(grid_agg.index),
        'employment_types': list(grid_agg.columns),
        'matrix': grid_agg.values.tolist()
    }
    
    # Portfolio Level Summary
    total_balance = float(df['Loan_Balance'].sum())
    total_customers = len(df)
    total_delinquents = int(df['Actual_Delinquent'].sum())
    overall_default_rate = round(total_delinquents / total_customers * 100, 2)
    total_expected_loss = float(df['Expected_Loss_Logistic'].sum())
    
    dashboard_payload = {
        "summary": {
            "total_customers": total_customers,
            "total_balance": total_balance,
            "total_delinquents": total_delinquents,
            "overall_default_rate": overall_default_rate,
            "total_expected_loss": total_expected_loss
        },
        "model_results": model_results,
        "roc_data": roc_data,
        "logistic_coefs": logistic_coefs,
        "rf_importance": rf_importance,
        "risk_tier_agg": risk_tier_agg.to_dict(orient='records'),
        "emp_agg": emp_agg.to_dict(orient='records'),
        "loc_agg": loc_agg.to_dict(orient='records'),
        "grid_data": grid_data,
        "customers": df.to_dict(orient='records')
    }
    
    output_path = "dashboard/data.json"
    with open(output_path, "w") as f:
        json.dump(dashboard_payload, f, indent=2)
        
    print(f"Dashboard data payload created successfully: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")

if __name__ == "__main__":
    build_dashboard_data()
