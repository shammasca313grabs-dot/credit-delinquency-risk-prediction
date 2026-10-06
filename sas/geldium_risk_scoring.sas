/********************************************************************************
 * PROGRAM NAME: geldium_risk_scoring.sas
 * PROJECT: Geldium Financial — Credit Delinquency Risk Analytics
 * AUTHOR: Risk Analytics Team
 * ENVIRONMENT: SAS OnDemand for Academics (ODA) / SAS Enterprise Guide
 * DESCRIPTION:
 *   1. Imports cleaned customer dataset (geldium_customers_clean.csv)
 *   2. Conducts vintage/roll-rate risk segment crosstabulations via PROC FREQ
 *   3. Replicates primary logistic regression risk scoring model via PROC LOGISTIC
 *   4. Outputs scored customer file with predicted default probabilities (P_DELINQUENT)
 ********************************************************************************/

/* 1. DEFINE FILE PATHS & IMPORT CLEANED CSV DATASET */
filename clean_csv "/home/u63000000/geldium_project/data/geldium_customers_clean.csv";

/* If running locally, adjust path accordingly */
proc import datafile=clean_csv
    out=work.geldium_clean
    dbms=csv
    replace;
    getnames=YES;
    guessingrows=2000;
run;

/* Inspect imported dataset structure */
proc contents data=work.geldium_clean;
run;

/* 2. VINTAGE & ROLL-RATE STYLE RISK SEGMENTATION ANALYSIS */
title1 "Geldium Financial — Risk Tier Distribution & Delinquency Crosstabs";

/* Crosstab 1: Delinquency Rate by Risk Tier (Credit Utilization Buckets) */
proc freq data=work.geldium_clean;
    tables Risk_Tier * Delinquent_Account / chisq nopercent norow colpcnt;
    format Delinquent_Account binary_fmt.;
run;

/* Crosstab 2: Delinquency Rate by Employment Status & Account Type */
proc freq data=work.geldium_clean;
    tables Employment_Status * Delinquent_Account 
           Account_Type * Delinquent_Account
           Employment_Status * Account_Type * Delinquent_Account / nopercent norow colpcnt;
run;

/* Crosstab 3: Geographic Risk Analysis (Location by Delinquency) */
proc freq data=work.geldium_clean;
    tables Location * Delinquent_Account / nopercent norow colpcnt;
run;
title1;

/* 3. LOGISTIC REGRESSION PREDICTIVE RISK MODELING */
title1 "Geldium Financial — PROC LOGISTIC Credit Delinquency Scorecard";

proc logistic data=work.geldium_clean plots=all;
    /* Define categorical class variables with reference levels */
    class Employment_Status (ref='Employed')
          Account_Type (ref='Personal')
          Location (ref='New York')
          Risk_Tier (ref='Low') / param=ref;

    /* Model specification: Target = Delinquent_Account (Event = '1') */
    model Delinquent_Account (event='1') = 
        Age 
        Income 
        Credit_Score 
        Credit_Utilization 
        Missed_Payments 
        Loan_Balance 
        Debt_to_Income_Ratio 
        Account_Tenure_Years
        Employment_Status 
        Account_Type 
        Location 
        Risk_Tier
        / selection=none 
          clodds=wald 
          ctable 
          outroc=work.sas_roc_data;

    /* Output scored dataset with predicted default probabilities */
    output out=work.geldium_scored 
           p=P_DELINQUENT 
           lower=P_LOWER_95 
           upper=P_UPPER_95;
run;
title1;

/* 4. POST-SCORING RISK TIER DISTRIBUTION & PROBABILITY PROFILE */
title1 "Geldium Financial — Scored Portfolio Probabilities by Risk Tier";

proc means data=work.geldium_scored n mean std min q1 median q3 max maxdec=4;
    class Risk_Tier;
    var P_DELINQUENT Loan_Balance Expected_Loss;
run;

/* Calculate Expected Financial Loss in SAS: Expected Loss = Loan_Balance * P_DELINQUENT */
data work.geldium_scored_final;
    set work.geldium_scored;
    Expected_Loss = Loan_Balance * P_DELINQUENT;
    
    /* Assign Predicted Risk Bucket based on probability cutoff (e.g. 0.30 threshold) */
    if P_DELINQUENT >= 0.30 then High_Risk_Flag = 1;
    else High_Risk_Flag = 0;
run;

/* Export Scored SAS Dataset to CSV for Enterprise Reporting */
proc export data=work.geldium_scored_final
    outfile="/home/u63000000/geldium_project/outputs/sas_scored_customers.csv"
    dbms=csv
    replace;
run;

title1 "SAS Risk Scoring Pipeline Execution Complete";
