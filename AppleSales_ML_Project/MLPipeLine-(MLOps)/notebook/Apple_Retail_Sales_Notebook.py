import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

product = pd.read_csv("Apple_Retail_Dataset/products.csv")
sales = pd.read_csv("Apple_Retail_Dataset/sales.csv")
warranty = pd.read_csv("Apple_Retail_Dataset/warranty.csv")
stores = pd.read_csv("Apple_Retail_Dataset/stores.csv")
category = pd.read_csv("Apple_Retail_Dataset/category.csv")

# Overview of warranty file
print('warranty head:',warranty.head())

print('warranty shape:',warranty.shape)
print('warranty dtypes:',warranty.dtypes)
print('warranty info:', warranty.info())
print('warranty describe:', warranty.describe())
print('warranty null values:', warranty.isnull().sum()*100)

# ===========================================
# Overview of Product file

print('product head:',product.head())

print('product shape:',product.shape)
print('product dtypes:',product.dtypes)
print('product info:', product.info())
print('product describe:', product.describe())
print('product null values:', product.isnull().sum()*100)

# ===========================================
# Overview of sales file
print('sales head:',sales.head())

print('sales shape:',sales.shape)
print('sales dtypes:',sales.dtypes)
print('sales info:', sales.info())
print('sales describe:', sales.describe())
print('sales null values:', sales.isnull().sum()*100)

# ===========================================
# Overview of stores file
print('stores head:',stores.head())

print('stores shape:',stores.shape)
print('stores dtypes:',stores.dtypes)
print('stores info:', stores.info())
print('stores describe:', stores.describe())
print('stores null values:', stores.isnull().sum()*100)

#============================================
# Overview of category file
print('category head:',category.head())

print('category shape:',category.shape)
print('category dtypes:',category.dtypes)
print('category info:', category.info())
print('category describe:', category.describe())
print('category null values:', category.isnull().sum()*100)

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

print('information of data:',df.info())
print('Types of Data:',df.dtypes)
print('Shape of Data:',df.shape)
print('Statistical Analysis of Data:',df.describe())

# Find The Missing Valus Of Data

print('Missing Values:',df.isnull().sum()*100)

# Correlation Between Numerical Values Of Data
print('Correlation Between Numerical Values')
print(df.corr(numeric_only=True))

print('HeatMap To see The Graphical Represention of Correlation Between Numerical Values')

sns.heatmap(df.corr(numeric_only=True),annot=True,linewidth=2,square=2,fmt='.1f')
plt.show()

from scipy.stats import chi2_contingency
score = []
for feature in df.columns[:-1]:
    ct = pd.crosstab(df['Total_Sales'],df[feature])
    pvalue = chi2_contingency(ct)[1]
    score.append(pvalue)

print(pvalue)

pd.Series(score,index=df.columns[:-1]).sort_values(ascending=True).plot(kind='bar')


X = df[['Price','quantity','Total_Sales']]

categorical_columns = [var for var in df.columns if df[var].dtypes == 'object']
numerical_columns = [var for var in df.columns if df[var].dtypes != 'object']

# To Find The Distribution Of Numerical Columns

df[numerical_columns].hist(bins=30, figsize=(8,3))

# KDE Plot for visualizing the estimated probalilty density function of a continous variable

cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(14,5))
    sns.kdeplot(X[columns])

# Box Plot To check the outliers in data

cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(14,5))
    sns.boxplot(X[columns])

# Lets see the Graphical Representation of Dataset Using Seaborn and Advance Graphic Library 'Plotly'
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode, iplot,plot
import plotly.graph_objects as go
from plotly.graph_objects import scatter
pyo.init_notebook_mode()
import plotly.io as pio
pio.renderers.default = 'iframe'

# UNIVARIATE ANALYSIS
print('unique values of columns City:',df['City'].nunique())
print('Value Vounts of City:',df['City'].value_counts())

sns.countplot(df,x='City')
plt.xticks(rotation=75,fontsize=7)
plt.show()

Cities = df['City'].value_counts()
sns.countplot(df,x='Country')
plt.xticks(rotation=75,fontsize=7)
plt.show()

plt.pie(df['category_name'].value_counts(),labels=['Accessories','Smartphone','Audio','Tablet','Desktop','Laptop','Wearable','Subscription Service','Streaming Device','Smart Speaker'],autopct='%1.f%%',pctdistance=0.85,explode=(0.2,0,0,0,0,0,0,0,0,0))
centre_circle = plt.Circle((0,0),0.75,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

print('Unique Values of column Store_Name:',df['Store_Name'].nunique())
print('Value Counts of Store Name:',df['Store_Name'].value_counts())

# Top 30 Store Name
store_name = df['Store_Name'].value_counts(30)
sns.barplot(x=store_name.index,y=store_name.values)
plt.xticks(rotation=90,fontsize=7)
plt.show()

print('Unique Values of column Product Name:',df['Product_Name'].value_counts())
print('Value Counts of Product Name:',df['Product_Name'].value_counts())

# Top 20 Product Name
product_name = df['Product_Name'].value_counts(20)
sns.barplot(x=product_name.index,y=product_name.values)
plt.xticks(rotation=90,fontsize=5)
plt.show()

# MULTIVARIATE ANALYSIS

# Sort data on the basis of 'Price' and 'Product Name'
sorted_df = df.sort_values(by=['Price','Product_Name'],ascending=False)
print(sorted_df)

# Sort data on the basis of 'Total_Sales' and 'Product_Name'
sorted_df1 = df.sort_values(by=['Total_Sales','Product_Name'],ascending=False)
print(sorted_df1)

sale_date = pd.to_datetime(df['sale_date'])
df['sale_month'] = pd.to_datetime(df['sale_date']).dt.month
df['sale_year'] = pd.to_datetime(df['sale_date']).dt.year
print(df.dtypes)
# Highest Apple Product Price
higest_product_price = df.groupby('Product_Name')['Price'].sum().nlargest(10)
print(higest_product_price)

# To Find Which Apple Product have Highest Sale
higest_product_sales = df.groupby('Product_Name')['Total_Sales'].sum().nlargest(10)
print(higest_product_sales)

# To find Which Product have Highest Price
higest_product_price = df.groupby('Product_Name').sum()['Price'].sort_values(ascending=False)
print(higest_product_price)

# To find Which Product have Lowest Price
lowest_product_price = df.groupby('Product_Name').sum()['Price'].sort_values(ascending=True)
print(lowest_product_price)

# To find Which Product have lowest Sale
lowest_product_sale = df.groupby('Product_Name').sum()['Total_Sales'].sort_values(ascending=True)
print(lowest_product_sale)

# Let's find which Country have more Sales of Apple products
which_country_have_more_sales_apple_products = df.groupby(['City','Product_Name'])['Total_Sales'].sum().nlargest(50).reset_index()
print(which_country_have_more_sales_apple_products)

# Let's Overview The London Data about Apple Products
# LONDON DATA
Group = df.groupby('City')

london = Group.get_group('London')[['category_name','Product_Name','quantity','Total_Sales','Store_Name']]
print(london)

# Which dubai store has higest Apple product Sales
group1 = london.groupby('Store_Name')['Total_Sales'].sum().sort_values(ascending=False)
print(group1)

# Let's Discover The Data About Apple Product in London's Store 'Apple Covent Garden'
london_Apple_covent_garden_data = df.loc[df['Store_Name'] == 'Apple Covent Garden']
print(london_Apple_covent_garden_data)

group = london_Apple_covent_garden_data.groupby('category_name')['Total_Sales'].sum().sort_values(ascending=False)
print(group)

# Which Apple Product have higest Price in London Apple Covent Garden Data
fig = px.bar(london_Apple_covent_garden_data, x='category_name',y='Price',hover_data=['quantity'],color='sale_year',text_auto='0.2s')
fig.show()

# Which Apple Product have heighest Sales in London Covent Garden Mall
fig = px.bar(london_Apple_covent_garden_data,x='category_name',y='Total_Sales',hover_data='quantity',color='sale_year',text_auto='0.2s')
fig.show()

# Which Apple Product have more demand to sell in high quantity
fig = px.bar(london_Apple_covent_garden_data,x='category_name',y='quantity',hover_data='sale_month',color='sale_year',text_auto='0.2s')
fig.show()

fig = px.scatter(london_Apple_covent_garden_data,x='category_name',y='Total_Sales',color='quantity')
fig.show()

category_name_total_sales = london_Apple_covent_garden_data.groupby('category_name')['Total_Sales'].sum().sort_values(ascending=False)
print(category_name_total_sales)

fig = px.sunburst(london_Apple_covent_garden_data,path=['sale_month','sale_year','category_name'],values='Total_Sales')
fig.update_traces(textinfo = 'label + percent parent')
fig.show()

# Let's Discover The USA Data
Group1 = df.groupby('Country')

USA = Group1.get_group('United States')[['category_name','Product_Name','quantity','Total_Sales','Store_Name']]
print(USA)

# Which USA store has higest Apple product Sales
USA_Store_data = USA.groupby('Store_Name')['Total_Sales'].sum().sort_values(ascending=False)
print(USA_Store_data)

# Lets Dicover the data about USA store 'Apple Fifth Avenue'
USA_Apple_5th_Avenue = df.loc[df['Store_Name'] == 'Apple Fifth Avenue']
print(USA_Apple_5th_Avenue)

# Category With Total_Sales
usa_group = USA_Apple_5th_Avenue.groupby('category_name')['Total_Sales'].sum().sort_values(ascending=False)
print(usa_group)

# Which Apple Product have heighest Sales in USA Apple Fifth Avenue
fig = px.bar(USA_Apple_5th_Avenue,x='category_name',y='Total_Sales',hover_data='quantity',color='sale_year',text_auto='0.2s')
fig.show()

usa = USA.groupby(['category_name','Store_Name']).size().unstack().fillna(0)
print(usa)

usa.plot(kind='bar',stacked=True,figsize=(12,12),cmap='Set1')


df['Launch_Date'] = pd.to_datetime(df['Launch_Date'])
df['Launch_Year'] = pd.to_datetime(df['Launch_Date']).dt.month
df['Launch_Month'] = pd.to_datetime(df['Launch_Date']).dt.year


# MACHINE LEARNING

df = df.drop(['Launch_Date','product_id','Category_ID','sale_date','store_id','sale_id'],axis=1)
X = df.drop('Total_Sales',axis=1)
y = df['Total_Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

Lr = LinearRegression()
Decision_Tree = DecisionTreeRegressor(criterion='squared_error',splitter='best',max_depth=5,)
Random_Forest = RandomForestRegressor(n_estimators=30,criterion='squared_error',max_depth=10)
GradientBoosting_Regressor = GradientBoostingRegressor(n_estimators=30,learning_rate=0.1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


preprocessor = ColumnTransformer([
    ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['Product_Name','Store_Name','City','Country','category_name'])
],remainder='passthrough')

pipe1 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('linearRegression',Lr)
])
pipe2 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('DecisionTreeRegressor',Decision_Tree)
])
pipe3 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('RandomForest',Random_Forest)
])
pipe4 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('GradientBoosting',GradientBoosting_Regressor)
])

pipe1.fit(X_train,y_train)
pipe2.fit(X_train,y_train)
pipe3.fit(X_train,y_train)
pipe4.fit(X_train,y_train)

y_pred = pipe1.predict(X_test)
y_pred1 = pipe2.predict(X_test)
y_pred2 = pipe3.predict(X_test)
y_pred3 = pipe4.predict(X_test)

# Accuracy for linear Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print('Accuracy For Linear Regression')

print(f'mean absolute error: {mae:.2f}')
print(f'mean squared error: {mse:.2f}')
print(f'root mean squared error: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')

# Accuracy for Decision Tree

print('Accuracy For Decision tree')

mae = mean_absolute_error(y_test,y_pred1)
mse = mean_squared_error(y_test,y_pred1)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred1)

print(f'mean absolute error: {mae:.2f}')
print(f'mean squared error: {mse:.2f}')
print(f'root mean squared error: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')

print('Accuracy For Random forest Regressor')

mae = mean_absolute_error(y_test,y_pred2)
mse = mean_squared_error(y_test,y_pred2)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred2)

print(f'mean absolute error: {mae:.2f}')
print(f'mean squared error: {mse:.2f}')
print(f'root mean squared error: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')

print('Accuracy for Gradient Boosting Regressor')

mae = mean_absolute_error(y_test,y_pred3)
mse = mean_squared_error(y_test,y_pred3)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred3)

print(f'mean absolute error: {mae:.2f}')
print(f'mean squared error: {mse:.2f}')
print(f'root mean squared error: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')

import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe3,open('pipe3.pkl','wb'))

import IPython.display
from IPython.display import FileLink
FileLink('pipe3.pkl')
FileLink('df.pkl')

# ====================================
# ML FLOW

import optuna

def objective(trial):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=10)
        rf_max_features = trial.suggest_float("rf_max_features", 0.2, 1.0)
        params = {
            "max_depth": rf_max_depth,
            "n_estimators": rf_n_estimators,
            "max_features": rf_max_features,
        }
        # Log current trial's parameters
        mlflow.log_params(params)

        regressor_obj = sklearn.ensemble.RandomForestRegressor(**params)
        regressor_obj.fit(X_train, y_train)

        y_pred = regressor_obj.predict(X_val)
        error = sklearn.metrics.mean_squared_error(y_val, y_pred)
        # Log current trial's error metric
        mlflow.log_metrics({"error": error})

        # Log the model file
        mlflow.sklearn.log_model(regressor_obj, name="model")
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