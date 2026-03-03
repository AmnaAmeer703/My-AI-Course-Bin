import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df_cab = pd.read_csv('archive/cab_rides.csv')
df_weather = pd.read_csv('archive/weather.csv')

print(df_cab)

print(df_cab.info())
print(df_cab.dtypes)
print(df_cab.shape)
print(df_cab.describe())
print(df_cab.isnull().mean()*100)
print(df_cab.duplicated().sum())

df_cab = df_cab.dropna()

df_cab['time_stamp'] = pd.to_datetime(df_cab['time_stamp'],unit='ms')

df_cab['date'] = pd.to_datetime(df_cab['time_stamp'].dt.date)

from datetime import datetime
df_cab['time'] = pd.to_datetime(df_cab['time_stamp']).dt.time
df_cab['hour'] = pd.to_datetime(df_cab['time_stamp']).dt.hour
df_cab['weekday'] = pd.to_datetime(df_cab['time_stamp']).dt.weekday

df_cab.head()

def f(x):
    if (x>4) and (x<=8):
        return 'Early Morning'
    elif (x>8) and (x<=12):
        return 'Morning'
    elif (x>12) and (x<=16):
        return 'Noon'
    elif (x>16) and (x<=20):
        return 'Evening'
    elif (x>20) and (x<=24):
        return 'Night'
    elif (x<=4):
        return 'Late Night'
    
df_cab['Period_Of_Time'] = df_cab['hour'].apply(f)

df_cab.drop(columns=['time_stamp','id','time','product_id'],inplace=True)

# Find The Correlation Between Numeric Columns

df_cab.corr(numeric_only=True)
sns.heatmap(df_cab.corr(numeric_only=True),annot=True,fmt='.1f',linewidth=2,square=True)

from scipy.stats import chi2_contingency
score = []
for feature in df_cab.columns[:-1]:
    ct = pd.crosstab(df_cab['price'],df_cab[feature])
    pvalue = chi2_contingency(ct)[1]
    score.append(pvalue)

pd.Series(score,index=df_cab.columns[:-1]).sort_values(ascending=True)

pd.Series(score,index=df_cab.columns[:-1]).sort_values(ascending=True).plot(kind='bar')

# Let's Find the Distribution of Data Columns
X = df_cab[['distance','price','surge_multiplier']]
cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(1, figsize=(2,2))
    sns.histplot(X[columns])

X = df_cab[['distance','price','surge_multiplier']]
cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(1, figsize=(2,2))
    sns.kdeplot(X[columns])

# Let's Check The Outliers Of Numerical Columns Through BoxPlot Graph

X = df_cab[['distance','price','surge_multiplier']]
cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(1, figsize=(2,2))
    sns.boxplot(X[columns])


# EXPLORTARY DATA ANALYSIS

sns.scatterplot(data=df_cab,x='distance',y='price',hue='cab_type')
plt.show()

sns.scatterplot(data=df_cab,x='Period_Of_Time',y='price',hue='cab_type')
plt.show()

import plotly.offline as pyo
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objects as go
from plotly.graph_objects import scatter
pyo.init_notebook_mode()
import plotly.io as pio
pio.renderers.default = 'iframe'

figure = px.scatter(data_frame=df_cab,x='distance',y='price',size='hour')
figure.show()



# UNIVARIATE ANALYSIS
labels = df_cab['cab_type'].value_counts()
Labels = ['Uber','Lyft']
plt.pie(df_cab['cab_type'].value_counts(),labels=['Uber','Lyft'],autopct='%1.f%%',explode=(0,0.2))
plt.show()

Labels = df_cab['cab_type'].value_counts()
labels = ['Uber','Lyft']
colors= ['#FF0000','#0000FF']
plt.pie(Labels,labels=labels,colors=colors,autopct='%1.f%%',pctdistance=0.85,explode=(0,0))
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

Labels = df_cab['name'].value_counts()
labels = ['UberXL','WAV','Black SUV','Black','UberX','UberPool','Lux','Lyft','Lux Black XL','Lyft XL','Lux Blaxk','Shared']
plt.pie(Labels,labels=labels,autopct='%1.f%%',pctdistance=0.85,explode=(0,0.2,0,0,0,0,0,0,0,0,0,0))
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

Labels = df_cab['destination'].value_counts()
labels = ['Financial District','Theatre District','Back Bay','Haymarket Square','Boston University','Fenway','North End','Northeastern University','South Station','West End','Beacon Hill','North Station']
plt.pie(Labels,labels=labels,autopct='%1.f%%',pctdistance=0.85,explode=(0,0,0.2,0,0,0,0,0,0,0,0,0))
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

sns.barplot(data=df_cab,x='name',y='price')
plt.xticks(rotation=90)
plt.show()

sns.barplot(data=df_cab,y='cab_type',x='price')
plt.xticks(rotation=90)
plt.show()

# MULTIVARIATE ANALYSIS

sns.barplot(data=df_cab,x='hour',y='price',hue='cab_type')
plt.xticks(rotation=90)
plt.show()

sns.barplot(data=df_cab,x='destination',y='price',hue='Period_Of_Time')
plt.xticks(rotation=90)
plt.show()

fig = px.bar(df_cab,x='destination',y='price',color='cab_type',text_auto='0.2s',title='Period_Of_Time')
fig.show()

fig = px.bar(df_cab,x='destination',y='price',hover_data=['cab_type','hour'],color='cab_type',labels={'price':'prices of cab'})
fig.show()



# lET'S CHECK WHICH CAB HAVE AVERAGE FARE USING GROUPBY METHOD

Average_Fare = df_cab.groupby(['cab_type','source','destination','name'])['price'].mean().reset_index()
print(Average_Fare)

fig = px.sunburst(Average_Fare,path=['cab_type','destination','source','name'],values='price')
fig.update_traces(textinfo='label+percent parent')
fig.update_layout()
fig.show()

df_cab.groupby('cab_type')['price'].mean()

# let's Check Which Cab Have Maximum Fares

Maximum_Fare = df_cab.groupby(['cab_type','name'])['price'].max().reset_index()
print(Maximum_Fare)

Group = df_cab.groupby('name')

df_cab.groupby('name')['price'].sum().sort_values(ascending=False).plot(kind='bar')


df_cab.groupby('name')['price'].max().sort_values(ascending=False)

# A BEAUTIFUL ANIMATED SCATTERPLOT USING PLOTLY
fig = px.scatter(df_cab,x='distance',y='price',animation_frame='name',animation_group='surge_multiplier',size='hour',color='cab_type',hover_name='cab_type',log_x=True,size_max=100,range_x=[1,50],range_y=[10,100])
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration']=700
fig.show()

# LET'S APPLY MACHINE LEARNING MODELS TO PREDICT THE FARES

df_cab = df_cab.drop(columns=['date','hour','weekday'])

X = df_cab[['distance','destination','cab_type','source','surge_multiplier','name','Period_Of_Time']]
y = df_cab[['price']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_cab.drop('price',axis=1),df_cab['price'],test_size=0.2,random_state=42)

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=50,criterion='squared_error',max_depth=20)
rfr.fit(X_train,y_train)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
trf1 = ColumnTransformer([
    ('ohe',OneHotEncoder(sparse_output=False,drop='first'),['destination','cab_type','source','name','Period_Of_Time'])
],remainder='passthrough')

UberLyftPipeline = Pipeline(steps=[
    ('trf1',trf1),
    ('rfr',rfr)
])
UberLyftPipeline.fit(X_train,y_train)

y_pred = UberLyftPipeline.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import math
from math import sqrt

print('MSE',mean_squared_error(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
print('MASE',sqrt(mean_absolute_error(y_test,y_pred)))
print('r2_score',r2_score(y_test,y_pred))

r2 = r2_score(y_test,y_pred)
n = X_test.shape[0]
p = X_test.shape[1]
R2 = 1-((1-r2)*(n-1)/(n-1-p))
R2

m = UberLyftPipeline.coef_
print(m)

b = UberLyftPipeline.intercept_
print(b)

# Cross Valaidation
from sklearn.model_selection import cross_val_score

cross_val_score = cross_val_score(UberLyftPipeline, X, y, score=['accuracy']).mean()
print(cross_val_score)

import matplotlib.pyplot as plt
fig = plt.figure()
sns.regplot(x=y_test,y=y_pred,ci=68,fit_reg=True,scatter_kws={'color':'red'},line_kws={'color':'blue'})
plt.title('y_test vs y_pred')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

import pickle
pickle.dump(df_cab,open('df_cab.pkl','wb'))
pickle.dump(UberLyftPipeline,open('UberLyftPipeline.pkl','wb'))

