# Loan Approval Risk Prediction: A Cost-Sensitive Classification Approach

A machine learning project for FinTech Innovations, built to support and standardize loan approval decisions using historical applicant data and business-defined cost figures.

## Overview

FinTech Innovations currently relies on manual review for loan approvals, which is slow and inconsistent across loan officers. This project builds a classification model to support that decision-making process, using 20,000 historical loan applications and the business's own cost figures:

- Missed good loan (false denial): **$8,000** lost profit
- Bad loan approved and defaulted (false approval): **$50,000** loss

Because these costs are asymmetric, the project treats this as a cost-sensitive classification problem rather than optimizing for plain accuracy.

## Project Structure

The project follows the CRISP-DM framework, broken into the following sections within the notebook:

1. Business Understanding
2. Data Understanding & Exploration (EDA)
3. Feature Understanding
4. Data Preparation / Preprocessing Pipeline
5. Modeling
6. Hyperparameter Tuning & Optimization
7. Model Evaluation & Business Impact
8. Feature Importance & Business Implications
9. Final Deliverable (Executive Summary, Recommendations, Limitations)

## Dataset

`financial_loan_data.csv` — 20,000 rows, 35 columns, including applicant demographics, financial indicators, credit history, and two possible targets: `LoanApproved` (binary) and `RiskScore` (continuous). This project uses `LoanApproved` as the target.

## Approach

- **Target:** `LoanApproved` (classification, not regression), chosen because the business cost figures map directly onto an approve/deny decision.
- **Preprocessing:** Separate pipelines for numeric, ordinal, and nominal features, combined via `ColumnTransformer`. Includes cleaning `AnnualIncome` (originally stored as text with `$` and commas), imputation, scaling, and encoding.
- **Models tested:** Logistic Regression, Decision Tree, Random Forest, XGBoost — each evaluated with 5-fold stratified cross-validation.
- **Custom evaluation metric:** A business cost scorer, `Total Cost = (False Negatives × $8,000) + (False Positives × $50,000)`, used alongside Recall and Precision to rank models by actual financial impact.
- **Tuning:** `GridSearchCV` for Logistic Regression, `RandomizedSearchCV` for XGBoost, including tuning of a preprocessing parameter (imputation strategy).

## Results

| Model | Recall | Precision | Avg. Cost (CV) |
|---|---|---|---|
| Logistic Regression (tuned) | 0.90 | 0.91 | **$3,857,600** |
| XGBoost (tuned) | 0.87 | 0.89 | $4,085,600 |
| Random Forest | 0.79 | 0.88 | $5,334,400 |
| Decision Tree | 0.77 | 0.78 | $9,870,400 |

**Final model: tuned Logistic Regression**

On the held-out test set, the final model achieved 96% accuracy and a 0.99 ROC-AUC, with a total business cost of $4,288,000 across 4,000 test applicants. Logistic Regression was selected not only for its performance, but for its interpretability — an important requirement in a regulated lending environment.

## Key Drivers of Approval Decisions

- **Risk factors:** high debt-to-income ratio, bankruptcy history, higher interest rate, unemployment
- **Positive factors:** higher monthly income, higher net worth, longer credit history, education level

One notable limitation: `CreditScore` showed a counter-intuitive negative relationship with approval, likely due to multicollinearity with `InterestRate` and `DebtToIncomeRatio`. This is flagged for further investigation rather than taken at face value.

## Recommendations

- Deploy as a decision-support tool for loan officers, not a fully automated approval system
- Route borderline high-risk approvals to manual review, given the high cost of false approvals
- Monitor approval rates across applicant segments (e.g. employment status) post-deployment for fairness
- Retrain periodically as applicant behavior and economic conditions shift

## Tools Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## Limitations & Future Work

- Multicollinearity between `CreditScore` and other risk features needs further investigation (e.g. VIF analysis)
- Class imbalance techniques such as SMOTE were not tested
- The dataset's "Good" credit score tier was underrepresented in the test set, limiting confidence in segment-level conclusions for that group
- A complementary regression model using the dataset's `RiskScore` target could provide a continuous risk estimate alongside the approve/deny decision
