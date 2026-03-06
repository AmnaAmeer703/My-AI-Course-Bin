import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Section1-Question1-InsurranceData.csv')

print(df.head())

print('info:', df.info())

print('dtypes:', df.dtypes)

print('describe:', df.describe())

print('Columns Names:', df.columns)

print('Duplicated Values:', df.duplicated().sum())

print('Shape Of The Data:', df.shape)

print(df['age'].max())
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
    ct = pd.crosstab(df['charges'],df[feature])
    pvalue = chi2_contingency(ct)[1]
    score.append(pvalue)

print(pvalue)

pd.Series(score,index=df.columns[:-1]).sort_values(ascending=True).plot(kind='bar')



categorical_columns = [var for var in df.columns if df[var].dtypes == 'object']
numerical_columns = [var for var in df.columns if df[var].dtypes != 'object']

# To Find The Distribution Of Numerical Columns

df[numerical_columns].hist(bins=30, figsize=(8,3))

# KDE Plot for visualizing the estimated probalilty density function of a continous variable

X = df[['age','bmi','children','charges']]

cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(14,5))
    sns.kdeplot(X[columns])

# Box Plot To check the outliers in data

cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(14,5))
    sns.boxplot(X[columns])


# EXPLORATORY DATA ANALYSIS


variables = ['age','bmi','children']
for var in variables:
    plt.figure()
    sns.regplot(x=var, y='charges', data=df).set(title=f'Regression plot of {var} and Charges');
    plt.show()


# UNIVARIATE ANALYSIS

# graphical represntation of column 'region'
print('unique values of columns region:',df['region'].nunique())
print('Value Counts of region:',df['region'].value_counts())

print(df['region'].value_counts())
sns.countplot(df,x='region')
plt.xticks(rotation=75,fontsize=7)
plt.show()

plt.pie(df['region'].value_counts(),labels=['southwest','northeast','southeast','northwest'],autopct='%1.f%%',pctdistance=0.85,explode=(0.2,0,0,0))
centre_circle = plt.Circle((0,0),0.75,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# graphical represntation of column 'sex'
print(df['sex'].value_counts())
sns.countplot(df,x='sex')
plt.xticks(rotation=75,fontsize=7)
plt.show()

# graphical represntation of column 'smoker'
print(df['smoker'].value_counts())
sns.countplot(df,x='smoker')
plt.xticks(rotation=75,fontsize=7)
plt.show()

# MULTIVARIATE ANALYSIS

# Lets see the Graphical Representation of Dataset Using Seaborn and Advance Graphic Library 'Plotly'
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode, iplot,plot
import plotly.graph_objects as go
from plotly.graph_objects import scatter
pyo.init_notebook_mode()
import plotly.io as pio
pio.renderers.default = 'iframe'

# Let's find the reltionship between dependent and independent variables

g = sns.scatterplot(data=df,x='bmi',y='charges', hue='sex')
g.figure.suptitle("scatterplot")
g.figure.show()

# Sort data on the basis of 'sex' and 'smoker' which smoke more 'female' or 'male'
sorted_df = df.sort_values(by=['sex','smoker'],ascending=False)
print(sorted_df)

fig = px.bar(df, x='sex',y='smoker')
fig.show()
# Let's see which 'sex' 'male', or 'female' have more 'charges'

fig = px.scatter(df,x='sex',y='charges')
fig.show()

# Which sex have more charges who smoke or not according to region
fig = px.bar(df, x='smoker',y='charges',hover_data=['region'],color='sex',text_auto='0.2s')
fig.show()

# Which age group smoke more

fig = px.bar(df, x='smoker',y='age')
fig.show()


# Lets use Groupby method to see which region has more male or female smoke
Group = df.groupby('region')

region = Group.get_group('southeast')[['sex','age','bmi','smoker','charges']]
print(region)


# A Beautiful Sunburst graph to represent the whole data
fig = px.sunburst(df,path=['sex','age','children','smoker','region'],values='charges')
fig.update_traces(textinfo = 'label + percent parent')
fig.show()


fig = px.bar(df,x='region',y='charges',hover_data='sex',color='smoker',text_auto='0.2s')
fig.show()

AgeGroup_SmokeMore = df.groupby(['sex','age']).size().unstack().fillna(0)
print(AgeGroup_SmokeMore)

AgeGroup_SmokeMore.plot(kind='bar',stacked=True,figsize=(12,12),cmap='Set1')


# MACHINE LEARNING

X = df.drop('charges',axis=1)
Y = df['charges']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

Lr = LinearRegression()
Decision_Tree = DecisionTreeRegressor(criterion='squared_error',splitter='best',max_depth=2,)
Random_Forest = RandomForestRegressor()
GradientBoosting_Regressor = GradientBoostingRegressor(loss='squared_error',max_depth=2,learning_rate=0.1)

preprocessor = ColumnTransformer([
    ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),['sex','smoker','region'])
],remainder='passthrough')

# ==============================================
# Let's use the Randomized Search CV to get the best Parameters to acieve higher accuracy
# use Randomized SearchSV for Random Forest Regressor Regressor Model

from sklearn.model_selection import RandomizedSearchCV
max_depth = [1,2,4,5,7,10,20,30,40,50,100,150,200,250,300]
n_estimators = [1,2,4,5,6,7,10,20,30,40,50,100,150,200]
criterion = ['squared_error']

param_grid = {'max_depth':max_depth,
              'n_estimators':n_estimators,
              'criterion':criterion}
rfr_grid = RandomizedSearchCV(Random_Forest,param_grid,cv=5,verbose=2)

RandomForest_pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('rfr_grid',rfr_grid)
])

RandomForest_pipe.fit(X_train,y_train)
predictions = RandomForest_pipe.predict(X_test)

# Accuracy of Random Forest Regressor by using RandomizedSearchCV

print('Accuracy for Random Forest Regressor')

mae = mean_absolute_error(y_test,predictions)
mse = mean_squared_error(y_test,predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,predictions)

print(f'mean absolute error: {mae:.2f}')
print(f'mean squared error: {mse:.2f}')
print(f'root mean squared error: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')

# R2 Score is 90
# ==============================================
# Linear Regression
pipe1 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('linearRegression',Lr)
])
# Decision Tree
pipe2 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('DecisionTreeRegressor',Decision_Tree)
])
# Gradient Boosting Regressor
pipe4 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('Gradient Boosting Regressor',GradientBoosting_Regressor)
])


pipe1.fit(X_train,y_train)
pipe2.fit(X_train,y_train)
pipe4.fit(X_train,y_train)


y_pred = pipe1.predict(X_test)
y_pred1 = pipe2.predict(X_test)
y_pred4 = pipe4.predict(X_test)

# Accuracy for linear Regression

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print('Accuracy For Linear Regression')

print(f'mean absolute error: {mae:.2f}')
print(f'mean squared error: {mse:.2f}')
print(f'root mean squared error: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')
# R2 Score is 80

# =============================
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
# R2 Score is 83

# =====================================
print('Accuracy For Gradient Boosting Regressor')

mae = mean_absolute_error(y_test,y_pred4)
mse = mean_squared_error(y_test,y_pred4)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred4)

print(f'mean absolute error: {mae:.2f}')
print(f'mean squared error: {mse:.2f}')
print(f'root mean squared error: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')
# R2 Score is 90
# ==============================================
import matplotlib.pyplot as plt
fig = plt.figure()
sns.regplot(x=y_test,y=predictions,ci=68,fit_reg=True,scatter_kws={'color':'red'},line_kws={'color':'blue'})
plt.title('y_test vs predictions')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

# ===============================================
# As Gradient Boosting Regressor got higher Accuracy so we dump pipe4 = Gradient Boosting Model
import pickle
pickle.dump(pipe4,open('pipe4.pkl','wb'))
pickle.dump(df,open('df.pkl','wb'))








