import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import mlflow

mlflow.set_experiment("MLflow Quickstart")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

product = pd.read_csv("Apple_Retail_Dataset/products.csv")
sales = pd.read_csv("Apple_Retail_Dataset/sales.csv")
warranty = pd.read_csv("Apple_Retail_Dataset/warranty.csv")
stores = pd.read_csv("Apple_Retail_Dataset/stores.csv")
category = pd.read_csv("Apple_Retail_Dataset/category.csv")


# ===========================================
# Merge All the CSV files to create a New Datasets

product = product.rename(columns={'Product_ID':'product_id'})
print(product)

stores = stores.rename(columns={'Store_ID':'store_id'})
print(stores)

category = category.rename(columns={'category_id':'Category_ID'})
print(category)

# Merge product and sales csv file
product_sales = pd.merge(product,sales, on = 'product_id')

# Merge Product_sales and stores CSV file
product_sales_stores = pd.merge(product_sales,stores, on = 'store_id')

# Merge Product_sales_stores and category Csv File
product_sales_stores_category = pd.merge(product_sales_stores,category, on = 'Category_ID')

df = product_sales_stores_category
print('df head:',df.head())

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
    ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['Product_Name','Store_Name','City','Country','category_name'])
],remainder='passthrough')


# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('rfr',Random_Forest)
])

    pipe.fit(X_train,y_train)
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
        rf_n_estimators = trial.suggest_int("rf_n_estimators",1,20, step=10)
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
            ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['Product_Name','Store_Name','City','Country','category_name'])
            ],remainder='passthrough')

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        # Log current trial's error metric
        mlflow.log_metrics({"error": error})

        # Log the model file
        mlflow.sklearn.log_model(pipe,'Random-Forest-Model')
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

# pip install mlflow
# Run the Server:
# mlflow server --port 5000
