"""
================================================================================
PROJECT TITLE  : End-to-End Real-World Retail Analytics & CLV Prediction Pipeline
DOMAIN         : E-Commerce / Retail Operations
AUTHOR         : Applied Learning Case Study
DATE           : June 2026
================================================================================
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def initialize_workspace():
    """Establishes an enterprise-grade project workspace structure."""
    target_folders = ['data', 'output']
    print("=" * 80)
    print("[SYSTEM] INITIALIZING DATA SCIENCE WORKSPACE ENVIRONMENT")
    print("=" * 80)
    for folder in target_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[INIT] Successfully generated directory path: './{folder}/'")
        else:
            print(f"[INIT] Existing folder path detected: './{folder}/'")

def simulate_enterprise_transactions(num_records=3000):
    """
    Simulates a highly realistic retail database ledger.
    Models exponential pricing decay, quantity variances, regional tags,
    and a 12-month transaction log window.
    """
    print("\n" + "-" * 50)
    print(f"[STAGE 1] INGESTING & SIMULATING RAW TRANSACTION RECORDS")
    print("-" * 50)
    np.random.seed(42)
    
    # Establish a clean 365-day historical timeline window ending in June 2026
    anchor_date = dt.datetime(2026, 6, 2)
    simulated_dates = [anchor_date - dt.timedelta(days=int(np.random.randint(0, 365))) for _ in range(num_records)]
    
    raw_payload = {
        'InvoiceNo': np.random.randint(710000, 790000, num_records),
        'CustomerID': np.random.randint(15000, 19500, num_records),
        'InvoiceDate': simulated_dates,
        'Quantity': np.random.choice([1, 2, 3, 5, 10, 15], num_records, p=[0.50, 0.25, 0.12, 0.08, 0.03, 0.02]),
        'UnitPrice': np.random.exponential(14.0, num_records) + 0.99,
        'Region': np.random.choice(['North America', 'EMEA', 'APAC', 'LATAM'], num_records, p=[0.45, 0.30, 0.15, 0.10])
    }
    
    df_transactions = pd.DataFrame(raw_payload)
    # Calculate row-level linear line totals
    df_transactions['TotalSpent'] = df_transactions['Quantity'] * df_transactions['UnitPrice']
    
    # Export raw backup file
    raw_path = 'data/raw_transactions.csv'
    df_transactions.to_csv(raw_path, index=False)
    print(f"[SUCCESS] Exported {num_records} transaction entries to: '{raw_path}'")
    print(df_transactions.head(3))
    return df_transactions

def process_rfm_features(df_raw):
    """
    Executes advanced feature engineering on customer transaction logs.
    Aggregates metrics to produce structural Recency, Frequency, and Monetary vectors.
    """
    print("\n" + "-" * 50)
    print("[STAGE 2] EXECUTING STRUCTURAL RFM FEATURE ENGINEERING")
    print("-" * 50)
    
    # Set evaluation reference milestone to 1 day past maximum recorded date
    processing_snapshot = df_raw['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # Group rows by distinct customer ID mappings
    rfm_features = df_raw.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (processing_snapshot - x.max()).days, # Recency (Days)
        'InvoiceNo': 'count',                                         # Frequency (Count)
        'TotalSpent': 'sum'                                           # Monetary (Value)
    })
    
    # Format structural descriptors cleanly
    rfm_features.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSpent': 'Monetary'
    }, inplace=True)
    
    # Export engineered features for audit traceability
    features_path = 'data/processed_rfm.csv'
    rfm_features.to_csv(features_path)
    print(f"[SUCCESS] Feature conversion finalized. Compiled {rfm_features.shape[0]} unique profiles.")
    print(rfm_features.head(3))
    return rfm_features

def execute_model_training(df_rfm):
    """
    Splits data partitions, scales feature insights, trains a 
    Random Forest Regressor, and prints evaluation metrics.
    """
    print("\n" + "-" * 50)
    print("[STAGE 3] SUPERVISED ML PIPELINE: TRAIN & EVALUATE")
    print("-" * 50)
    
    # Separate predictors from target values
    X = df_rfm[['Recency', 'Frequency']]
    y = df_rfm['Monetary']
    
    # Establish a reliable 80/20 data partition split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.2, random_state=42)
    print(f"[INFO] Training Split Matrix Size : {X_train.shape}")
    print(f"[INFO] Testing Split Matrix Size  : {X_test.shape}")
    
    # Initialize and train the regressor model
    clv_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    print("[INFO] Tuning and fitting Random Forest ensemble structural paths...")
    clv_model.fit(X_train, y_train)
    
    # Generate predictive evaluations on the unseen test partition
    predictions = clv_model.predict(X_test)
    
    # Compute mathematical evaluation criteria
    r2_metric = r2_score(y_test, predictions)
    rmse_metric = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n" + "=" * 45)
    print("          PIPELINE MODEL PERFORMANCE         ")
    print("=" * 45)
    print(f" Variance Explained (R² Score)  : {r2_metric:.4f}")
    print(f" Model Error Margin (RMSE)       : ${rmse_metric:.2f}")
    print(f" Mean Customer Baseline Value    : ${y_test.mean():.2f}")
    print("=" * 45 + "\n")
    
    return y_test, predictions

def generate_performance_plots(y_true, y_pred):
    """Generates high-resolution diagnostic charts to evaluate performance curves."""
    print("-" * 50)
    print("[STAGE 4] GENERATING ENTERPRISE PERFORMANCE VISUALIZATIONS")
    print("-" * 50)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Plot true values against the model's predictions
    sns.scatterplot(
        x=y_true, 
        y=y_pred, 
        alpha=0.65, 
        color='#4f46e5', 
        edgecolors='none', 
        s=55, 
        ax=ax,
        label='Customer Profile Records'
    )
    
    # Draw a perfect prediction reference line
    min_bound = min(y_true.min(), y_pred.min())
    max_bound = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_bound, max_bound], 
        [min_bound, max_bound], 
        color='#ef4444', 
        linestyle='--', 
        lw=2.5, 
        label='Perfect Forecasting Baseline (y = x)'
    )
    
    # Format labels and titles cleanly
    ax.set_title('Applied Model Diagnostics: Actual vs. Predicted Monetary Value (CLV)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('True Historical Customer Value ($)', fontsize=11, fontweight='medium')
    ax.set_ylabel('Model Predicted Customer Value ($)', fontsize=11, fontweight='medium')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    chart_destination = 'output/actual_vs_predicted_clv.png'
    plt.savefig(chart_destination, dpi=300)
    print(f"[SUCCESS] High-resolution visualization successfully exported to: '{chart_destination}'")
    plt.close()

if __name__ == "__main__":
    # Execute the structured, end-to-end data science pipeline sequentially
    initialize_workspace()
    raw_df = simulate_enterprise_transactions(num_records=3500)
    rfm_matrix = process_rfm_features(raw_df)
    test_actuals, model_preds = execute_model_training(rfm_matrix)
    generate_performance_plots(test_actuals, model_preds)
    print("\n" + "=" * 80)
    print("[FINISHED] END-TO-END DATA PIPELINE CYCLE COMPLETED SUCCESSFULLY")
    print("=" * 80)