import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

import mlflow

mlflow.set_experiment("MLflow Quickstart10")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

df = pd.read_csv('bank_transactions_data_2.csv')

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])

df['Day_Of_Week'] = df['TransactionDate'].dt.day_name()

df['TransactionHours'] = df['TransactionDate'].dt.hour

mean = df.groupby(['AccountID'])['TransactionAmount'].transform('mean')
std = df.groupby('AccountID')['TransactionAmount'].transform('std')
df['Z-Score'] = (df['TransactionAmount'] - mean ) / (std + 1e-6)

df['AmountBalanceRatio'] = df['TransactionAmount'] / (df['AccountBalance'] + 1e-6)

# High Login Attempts
df['HighLoginAttempts'] = (df['LoginAttempts'] > 3).astype(int)

# Risk Score
Risk_Score = {
    'Student' : 0,
    'Engineer' : 1,
    'Doctor' : 2,
    'Retired' : 3
}

df['RiskScore'] = df['CustomerOccupation'].map(Risk_Score)
df['TransactionType'] = df['TransactionType'].map({'Credit':0,'Debit':1})

df = df.drop(columns = ['TransactionID','AccountID','Location','DeviceID','IP Address','CustomerOccupation','MerchantID','TransactionDate','PreviousTransactionDate','Day_Of_Week'])

print(df.isnull().mean()*100)

df['Z-Score'] = df['Z-Score'].fillna(0)

print(df.shape)

ohe = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
arr = ohe.fit_transform(df[['Channel']])
new_df = pd.DataFrame(arr, columns = ohe.get_feature_names_out())
bank = pd.concat((df, new_df), axis=1)

bank.drop('Channel',axis=1,inplace=True)


pipe = Pipeline([
    ('scaler', StandardScaler())
])

scaled = pipe.fit_transform(bank)

from sklearn.ensemble import IsolationForest

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
arr = pca.fit_transform(scaled)
pca_df = pd.DataFrame(arr, columns = ['PCA1','PCA2'])


model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', IsolationForest(n_estimators=100,max_samples='auto',contamination=0.03))
])

mlflow.log_artifact(__file__)
    # Log the model
model_info = mlflow.sklearn.log_model(sk_model=pipe)

forest_pred = model.fit_predict(bank)
params = model.get_params()

mlflow.log_params(params)

iso_pca = PCA(n_components=2)
arr2 = iso_pca.fit_transform(scaled)
forest_pcadf = pd.DataFrame(arr2, columns=['PCA1','PCA2'])
forest_pcadf['forest_preds'] = forest_pred
forest_pcadf['forest_preds'] = forest_pcadf['forest_preds'].map({1:'Normal',-1:'Fraud'})

forest_pcadf[forest_pcadf['forest_preds'] == -1].head()
anomaly_count = 2512
accuracy = 100*list(forest_pcadf['forest_preds']).count(-1)/(anomaly_count)
mlflow.log_metric("accuracy", accuracy)

# ======================================
# Data Monitoring
def monitor_dataset_quality(dataset, reference_dataset=None):
    """Monitor dataset quality and compare against reference if provided."""

    data = dataset.df if hasattr(dataset, "df") else dataset

    quality_metrics = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "missing_values_total": data.isnull().sum().sum(),
        "missing_values_pct": (data.isnull().sum().sum() / data.size) * 100,
        "duplicate_rows": data.duplicated().sum(),
        "duplicate_rows_pct": (data.duplicated().sum() / len(data)) * 100,
    }

    # Numeric column statistics
    numeric_cols = data.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        quality_metrics.update(
            {
                f"{col}_mean": data[col].mean(),
                f"{col}_std": data[col].std(),
                f"{col}_missing_pct": (data[col].isnull().sum() / len(data)) * 100,
            }
        )

    with mlflow.start_run(run_name="Data_Quality_Check"):
        mlflow.log_input(dataset, context="quality_monitoring")
        mlflow.log_metrics(quality_metrics)

        # Compare with reference dataset if provided
        if reference_dataset is not None:
            ref_data = (
                reference_dataset.df
                if hasattr(reference_dataset, "df")
                else reference_dataset
            )

            # Basic drift detection
            drift_metrics = {}
            for col in numeric_cols:
                if col in ref_data.columns:
                    mean_diff = abs(data[col].mean() - ref_data[col].mean())
                    std_diff = abs(data[col].std() - ref_data[col].std())
                    drift_metrics.update(
                        {f"{col}_mean_drift": mean_diff, f"{col}_std_drift": std_diff}
                    )

            mlflow.log_metrics(drift_metrics)

    return quality_metrics




