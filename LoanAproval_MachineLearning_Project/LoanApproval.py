import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/loan-approval-classification-data/loan_data.csv')
print(df)

print(df.info())
print(df.shape())
print(df.describe)
print(df.isnull().mean()*100)
print(df.dtypes())
print(df.duplicated().sum())

print(df.columns)


print(df['previous_loan_defaults_on_file'].value_counts())

categorical_columns = [var for var in df.columns if df[var].dtypes == 'object']
numerical_columns = [var for var in df.columns if df[var].dtypes != 'object']

# Let''s Check the Data Distribution using Histogram Chart

df[numerical_columns].hist(bins=30, figsize=(12,10))
plt.show()

print(df['person_age'].max())
# An Outlier found in column 'person_age'
# Using 'iqr' method to remove "Outliers" in 'person_age' column

percentile75 = df['person_age'].quantile(0.75)
percentile25 = df['person_age'].quantile(0.25)

upper_limit = percentile75 + 1.5*iqr
lower_limit = percentile25 - 1.5*iqr

df1 = df.copy()
df1['person_age'] = np.where(
    df1['person_age']>upper_limit,
    upper_limit,
    np.where(
        df1['person_age']<lower_limit,
        lower_limit,
        df1['person_age']
    )
)

print(df1['person_age'].max())

# Let's Check The Probality Density Of Continous data 
X = df[numerical_columns]
for columns in X:
    fig, axs = plt.subplots(figsize=(2,2))
    sns.kdeplot(X[columns])

# Checking The Outliers Using Box Plot

X = df[numerical_columns]
for columns in X:
    fig, axs = plt.subplots(figsize=(2,2))
    sns.boxplot(X[columns])

#Correlation between Loan Amount and Loan Interest Rate
correlation = df['loan_status'].corr(df['person_income'])
print(f'correlation between loan status and person income: {correlation:.2f}')

#Correlation between Loan Amount and Interest Rate
Correlation = df['loan_amnt'].corr(df['loan_int_rate'])
print(f'correlation between loan amount and loan interest rate: {correlation:.2f}')

df.corr(numeric_only=True)

sns.heatmap(df.corr(numeric_only=True),annot=True,fmt='.1f',linewidth=2,square=False)

# EXPLORATORY DATA ANALYSIS

Most_loan_people_according_to_creditability = df['person_home_ownership'].value_counts()
print(Most_loan_people_according_to_creditability)

sns.barplot(x=Most_loan_people_according_to_creditability.values,y=Most_loan_people_according_to_creditability.index)
plt.show()

sns.scatterplot(data=df,y='loan_int_rate',x='loan_amnt')
plt.show()

sns.countplot(data=df1,x='person_gender')

sns.countplot(data=df1,x='loan_status')
plt.show()

reason_of_loan = df['loan_intent'].value_counts()
sns.countplot(df,x='loan_intent')
plt.xticks(rotation=90)
plt.show()

# column person_education
Labels = df['person_education'].value_counts()
labels = ['Bachelor','Associate','High School','Master','Doctorate']
plt.pie(Labels,labels=labels,autopct='%1.f%%',pctdistance=0.85,explode=(0,0.1,0,0,0))
centre_circle=plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Column Person_home_ownership
Labels = df['person_home_ownership'].value_counts()
labels = ['RENT','MORTGAGE','OWN','OTHER']
plt.pie(Labels,labels=labels,autopct='%1.f%%',pctdistance=0.85,explode=(0,0.1,0,0))
centre_circle=plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# MULTIVRIATE ANALYSIS
sns.barplot(data=df,y='person_income',x='person_education',hue='person_gender')
plt.show()

import plotly.offline as pyo
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objects as go
from plotly.graph_objects import scatter
pyo.init_notebook_mode()
import plotly.io as pio
pio.renderers.default = 'iframe'

fig = px.histogram(data_frame=df,y='person_income',x='person_education',color='person_gender',text_auto='0.2s')
fig.show()

fig = px.histogram(data_frame=df,y='person_income',x='loan_intent',color='person_home_ownership',text_auto='0.2s')
fig.show()

fig = px.bar(data_frame=df,x='person_home_ownership',y='person_income',hover_data=['person_education'],color='person_gender')
fig.show()

fig = px.sunburst(df1,path=['person_home_ownership','loan_status'],values='person_income')
fig.update_traces(textinfo='label+percent parent')
fig.show()

fig = px.sunburst(df1,path=['person_home_ownership','person_education','loan_status'],values='person_income')
fig.update_traces(textinfo='label+percent parent')
fig.show()

# Analysis Using GroupBy Method

Loan_status = df.groupby(['loan_status','person_home_ownership','person_education'])['person_income'].mean().reset_index()
print(Loan_status)

fig = px.sunburst(Loan_status,path=['loan_status','person_home_ownership','person_education'],values='person_income')
fig.update_traces(textinfo='label+percent parent')
fig.update_layout()
fig.show()

loan_status = df.groupby(['loan_status','person_education','person_home_ownership','loan_intent'])['person_income'].mean().reset_index()
loan_status

fig = px.sunburst(loan_status,path=['loan_status','loan_intent','person_income'],values='person_income')
fig.update_traces(textinfo='label+percent parent')
fig.update_layout()
fig.show()

fig = px.histogram(data_frame=df,y='person_income',x='loan_intent',color='person_home_ownership',text_auto='0.2s')
fig.show()

fig = px.sunburst(loan_status,path=['loan_status','loan_intent','person_home_ownership'],values='person_income')
fig.update_traces(textinfo='label+percent parent')
fig.update_layout()
fig.show()

Group = df.groupby('loan_status')

Loan_Rejected_ratio_loan_intent = Group.get_group(0)[['loan_intent','person_home_ownership','person_income']].reset_index()
print(Loan_Rejected_ratio_loan_intent)

fig = px.sunburst(Loan_Rejected_ratio_loan_intent,path=['loan_intent','person_home_ownership'],values='person_income')
fig.update_traces(textinfo='label+percent parent')
fig.show()

Group1 = df.groupby('loan_status')

Loan_Accepted_ratio_loan_intent = Group1.get_group(1)[['loan_intent','person_home_ownership','person_income']].reset_index()
print(Loan_Accepted_ratio_loan_intent)

fig = px.sunburst(Loan_Accepted_ratio_loan_intent,path=['loan_intent','person_home_ownership'],values='person_income')
fig.update_traces(textinfo='label+percent parent')
fig.show()

# MACHINE LEARNING

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,recall_score

scaler = StandardScaler()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df1.drop('loan_status',axis=1),df1['loan_status'],test_size=0.2,random_state=42)


from sklearn.ensemble import RandomForestClassifier
Random = RandomForestClassifier(criterion='entropy',n_estimators=20,n_jobs=15,max_depth=10,max_features='sqrt')

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
n_estimators = [10,20,30,40,50]
n_jobs = [5,10,15,20]
max_depth = [5,10,15,20]
max_features =['sqrt']
criterion= ['gini','entropy']

param_grid = {'n_estimators':n_estimators,
             'n_jobs':n_jobs,
             'max_depth':max_depth,
             'max_features':max_features,
             'criterion':criterion
             }
print(param_grid)
RFC = RandomForestClassifier()

RFC_grid = RandomizedSearchCV(RFC,param_grid,cv=5,refit=True,verbose=3)

RFC_grid.fit(X_train,y_train)

pred1 = RFC_grid.predict(X_test)

print(accuracy_score(y_test,pred1))



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
trf1 = ColumnTransformer([
    ('ohe',OneHotEncoder(drop='first',sparse_output=False),['person_gender','person_home_ownership','loan_intent','previous_loan_defaults_on_file']),
    ('oe',OrdinalEncoder(categories=[['High School','Bachelor','Associate','Master','Doctorate']]),['person_education'])
],remainder='passthrough')
trf2 = ColumnTransformer([
    ('scaler',StandardScaler(),slice(18))
])
Random = RandomForestClassifier(n_estimators=20,n_jobs=15,max_depth=10,criterion='gini',max_features='sqrt')

loan_pipe = Pipeline(steps=[
    ('trf1',trf1),
    ('trf2',trf2),
    ('Random',Random)
])

loan_pipe.fit(X_train,y_train)

pred = loan_pipe.predict(X_test)

print(accuracy_score(y_test,pred))

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))

sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt='d',cmap='Blues')


from sklearn.model_selection import cross_val_score
cross_val_score = cross_val_score(loan_pipe,X_train,y_train,cv=5,scoring='accuracy')
print(cross_val_score)

import pickle
pickle.dump(df1,open('df1.pkl','wb'))
pickle.dump(loan_pipe,open('loan_pipe.pkl','wb'))

from IPython.display import FileLink
FileLink('loan_pipe.pkl')
FileLink('df1.pkl')