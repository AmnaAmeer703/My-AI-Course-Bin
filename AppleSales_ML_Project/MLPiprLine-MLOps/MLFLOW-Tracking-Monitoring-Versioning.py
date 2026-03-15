import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.metrics as r2_score
import warnings
warnings.filterwarnings('ignore')
import mlflow

mlflow.set_experiment("MLflow Quickstart5")

product = pd.read_csv("Apple_Retail_Dataset/products.csv")
sales = pd.read_csv("Apple_Retail_Dataset/sales.csv")
warranty = pd.read_csv("Apple_Retail_Dataset/warranty.csv")
stores = pd.read_csv("Apple_Retail_Dataset/stores.csv")
category = pd.read_csv("Apple_Retail_Dataset/category.csv")

product = product.rename(columns={'Product_ID':'product_id'})

stores = stores.rename(columns={'Store_ID':'store_id'})

category = category.rename(columns={'category_id':'Category_ID'})

# Merge product and sales csv file
product_sales = pd.merge(product,sales, on = 'product_id')

# Merge Product_sales and stores CSV file
product_sales_stores = pd.merge(product_sales,stores, on = 'store_id')

# Merge Product_sales_stores and category Csv File
product_sales_stores_category = pd.merge(product_sales_stores,category, on = 'Category_ID')

df = product_sales_stores_category

# We can create a new column Total_Sales by multiply 2 columns 'quantity' and 'prices'
df['Total_Sales'] = df['Price'] * df['quantity']

df['Launch_Date'] = pd.to_datetime(df['Launch_Date'])
df['Launch_Year'] = pd.to_datetime(df['Launch_Date']).dt.month
df['Launch_Month'] = pd.to_datetime(df['Launch_Date']).dt.year


# MACHINE LEARNING

df = df.drop(['Launch_Date','product_id','Category_ID','sale_date','store_id','sale_id'],axis=1)
X = df.drop('Total_Sales',axis=1)
y = df['Total_Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# MACHINE LEARNING

X = df.drop('Total_Sales',axis=1)
Y = df['Total_Sales']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

params = {
    "n_estimators":30,
    "max_depth": 10,
    "criterion": "squared_error",
}

Random_Forest = RandomForestRegressor(**params)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



preprocessor = ColumnTransformer([
    ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['sex','smoker','region'])
],remainder='passthrough')


# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    pipe3 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('rfr',Random_Forest)
])

    pipe3.fit(X_train,y_train)
    mlflow.log_artifact(__file__)
    # Log the model
    model_info = mlflow.sklearn.log_model(sk_model=pipe)

    # Predict on the test set, compute and log the loss metric
    y_pred = pipe.predict(X_test)

    accuracy = sklearn.metrics.r2_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.set_tag("Training Info -", "Basic Random Forest model for data")

    """
    Load the model back for inference.
    After logging the model, we can perform inference by:

    Loading the model using MLflow's pyfunc flavor.
    Running Predict on new data using the loaded model.
    """
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    df_feature_names = df.columns

    result = pd.DataFrame(X_test, columns=df.columns)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result[:4])

# ================================
# MLFLOW HyperPerameter Tuning
import optuna
import sklearn

mlflow.set_experiment("Hyperparameter Tuning Experiment")
def objective(trial):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        rf_max_depth = trial.suggest_int("rf_max_depth",10,50)
        rf_n_estimators = trial.suggest_int("rf_n_estimators",10,30, step=10)
        rf_max_features = trial.suggest_float("rf_max_features", 0.2, 1.0)
        params = {
            "max_depth": rf_max_depth,
            "n_estimators": rf_n_estimators,
            "max_features": rf_max_features,
        }
        
        # Log current trial's parameters
        mlflow.log_params(params)

        Random_Forest = RandomForestRegressor(**params)
        preprocessor = ColumnTransformer([
            ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['sex','smoker','region'])
            ],remainder='passthrough')

        pipe3.fit(X_train, y_train)

        y_pred = pipe3.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        # Log current trial's error metric
        mlflow.log_metrics({"error": error})

        # Log the model file
        mlflow.sklearn.log_model(pipe3,'Random-Forest-Model')
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)
        return error
    
    # Create a parent run that contains all child runs for different trials
with mlflow.start_run(run_name="study") as run:
    # Log the experiment settings
    n_trials = 30
    mlflow.log_param("n_trials", n_trials)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Log the best trial and its run ID
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metrics({"best_error": study.best_value})
    if best_run_id := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id", best_run_id)

# ============================================
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
# Model Registry

import mlflow
from mlflow import MlflowClient

# Register a model from a logged run

# The Model Registry tracks models by versions and organizes them into predefined stages:
client = MlflowClient()
model_uri = "runs:/<run_id>/model"
registered_model_name = "Apple Retail Sales Predictor"
client.create_registered_model(registered_model_name)
client.create_model_version(registered_model_name, model_uri, "Source: Experiment 1")
print(f"Model '{registered_model_name}' registered successfully!")

# Transition a model version to Staging
client.transition_model_version_stage(
    name="Apple Retail Sales Predictor",
    version=1,
    stage="Staging"
)
print("Model transitioned to Staging!")

# Add a description to the model
client.update_registered_model(
    name="Apple Retail Sales Predictor",
    description="Apple Retail Sales."
)

# Add tags to a specific model version
client.set_model_version_tag(
    name="Apple Retail Sales Predictor",
    version=1,
    key="Framework",
    value="scikit-learn"
)

# Querying Registered Models

# Search for models
models = client.search_registered_models(filter_string="tags.Framework = 'scikit-learn'")
for model in models:
    print(f"Model Name: {pipe3.name}, Latest Version: {pipe3.latest_versions[0].version}")

# ========================================
# UI Integration
# mlflow ui
# Train and Log Model:
# with mlflow.start_run():      Train and log model     
# mlflow.sklearn.log_model(pipe, "pipe")
# Register Model:
# client.create_model_version("LoanDefaultPredictor", "runs:/<run_id>/model", "Source: Experiment 1")
# Validate Model in Staging:
# client.transition_model_version_stage("LoanDefaultPredictor", 1, "Staging")
# Deploy Model to Production:
# client.transition_model_version_stage("LoanDefaultPredictor", 1, "Production")
# Archive Old Models:
# client.transition_model_version_stage("LoanDefaultPredictor", 1, "Archived")

# pip install mlflow
# Run the Server:
# mlflow server --port 5000
