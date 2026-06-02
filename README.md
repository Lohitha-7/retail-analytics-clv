# retail-analytics-clv
An end-to-end real-world retail data science project focused on RFM customer segmentation, data visualization, and predictive modeling using Random Forest Regressors to forecast Customer Lifetime Value (CLV).
🎯 Project Objective
The goal of this project is to perform an end-to-end data analysis and predictive modeling workflow on a retail transactions dataset. By analyzing historical purchase patterns, we aim to calculate Customer Lifetime Value (CLV), segment shoppers using RFM (Recency, Frequency, Monetary) analysis, and forecast future sales trends to optimize inventory management.

🛠️ The End-to-End Project Architecture
Plaintext
├── data/
│   ├── raw_transactions.csv      # Original retail sales records
│   └── processed_rfm.csv         # Engineered customer metrics
├── notebooks/
│   ├── 1_eda_and_cleaning.ipynb  # Missing value handling & trends
│   └── 2_customer_rfm_clv.ipynb  # Segmentation and modeling
├── src/
│   ├── pipeline_utils.py         # Data processing helper functions
│   └── sales_forecaster.py       # Time-series or Regression training
├── README.md                     # Project documentation & findings
└── requirements.txt              # Dependency packages
💻 Python Implementation Blueprint
This comprehensive script covers data ingestion, cleaning, feature engineering (RFM analysis), and a predictive machine learning model to estimate total spending.

Python
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set rendering aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================
# 1. SYNTHETIC DATA GENERATION (Retail Core)
# ==========================================
np.random.seed(42)
n_records = 2000

start_date = dt.datetime(2025, 1, 1)
date_list = [start_date + dt.timedelta(days=int(np.random.randint(0, 500))) for _ in range(n_records)]

data = {
    'InvoiceNo': np.random.randint(500000, 580000, n_records),
    'CustomerID': np.random.randint(12000, 18000, n_records),
    'InvoiceDate': date_list,
    'Quantity': np.random.choice([1, 2, 3, 5, 10, 20], n_records, p=[0.4, 0.3, 0.15, 0.1, 0.03, 0.02]),
    'UnitPrice': np.random.exponential(15, n_records) + 0.5,
    'Country': np.random.choice(['United Kingdom', 'Germany', 'France', 'India'], n_records, p=[0.8, 0.1, 0.05, 0.05])
}

df = pd.DataFrame(data)
# Calculate Total Purchase Value
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

print("--- Initial Dataset Profile ---")
print(df.head())

# ==========================================
# 2. FEATURE ENGINEERING (RFM Segmentation)
# ==========================================
# Set snapshot date as the day after the last recorded transaction
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'count',                                   # Frequency
    'TotalAmount': 'sum'                                    # Monetary Value
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalAmount': 'Monetary'
}, inplace=True)

print("\n--- Engineered RFM Customer Profiles ---")
print(rfm.head())

# ==========================================
# 3. PREDICTIVE MODELING (Forecasting Value)
# ==========================================
# Objective: Predict a customer's Monetary value based on Recency and Frequency features
X = rfm[['Recency', 'Frequency']]
y = rfm['Monetary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.2, random_state=42)

# Train a robust Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate predictions
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- Model Metrics ---")
print(f"R² Score (Variance Explained): {r2:.4f}")
print(f"RMSE (Error Margin): ${rmse:.2f}")

# ==========================================
# 4. RESULTS VISUALIZATION
# ==========================================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='dodgerblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Customer Value')
plt.xlabel('Actual Monetary Value ($)')
plt.ylabel('Predicted Monetary Value ($)')
plt.tight_layout()
plt.show()
📈 Key Visualizations & Portraying Insights
To deliver a high-impact presentation or dashboard summary, your final project report should highlight these three crucial elements:

1. RFM Distribution Patterns
Using 3D scatter plots or side-by-side box plots, show how your customer segments behave.

High-Value Champions: Low Recency (shopped very recently), High Frequency, High Monetary scores.

At-Risk Customers: High Recency (haven't returned in a long time), but historically high monetary values. This signals a need for targeted re-engagement campaigns.

2. Actual vs. Predicted Diagnostics
The scatter plot comparison included in the script maps model predictions directly against actual purchase realities. The red dashed line represents a perfect forecasting model. Points clustering tightly around this line indicate a highly accurate model.

3. Seasonality and Sales Trends
Resampling the transaction data into daily or monthly buckets helps reveal critical timeline behaviors, such as holiday purchase spikes or regular seasonal dips.
