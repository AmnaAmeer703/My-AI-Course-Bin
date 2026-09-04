import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

import mlflow

mlflow.set_experiment("MLflow Tracking-Monitoring")

# Enable autologging
mlflow.tensorflow.autolog()


df = pd.read_csv('south-korean-pollution-data.csv')
print(df.head())

df = df.drop(columns=['Unnamed: 0'])

df['date'] = pd.to_datetime(df['date'])
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month

data = df[['date','pm25']]
data.set_index('date',inplace=True)

pm25 = df['pm25'].astype(int).values.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(pm25)

window_size = 12
X = []
y = []
target_values = df.index[window_size:]

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, train_values, test_values = train_test_split(
    X, y, target_values, test_size=0.1, shuffle=False
)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


model = Sequential()
model.add(LSTM(units=128, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

with mlflow.start_run():
    print("model.fit(x_train,y_train , epochs=5) :         \n" , model.fit(X_train,y_train , epochs=10 , batch_size=20 ))


    mlflow.log_artifact(__file__)

    model_info = mlflow.sklearn.log_model(model, "LSTM")
    print("[INFO] Predict via network...")
    predictions = model.predict(X_test)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_predictions = scaler.inverse_transform(train_predictions).flatten()
    test_predictions = scaler.inverse_transform(test_predictions).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()


    print("model.evaluate(x_test,y_test) :    \n" , model.evaluate(X_test,y_test) )

    rmse = np.sqrt(np.mean((y_test - test_predictions)**2))
    print(f'RMSE: {rmse:.2f}')

   
    r2_score = r2_score(y_test,test_predictions)

    mlflow.log_metric("accuracy", r2_score)

    print("Model summary:    \n " , model.summary() )

    print()

# =================================================
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
# ============================================