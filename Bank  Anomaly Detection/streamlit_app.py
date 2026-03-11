import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import numpy as np
import datetime
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
import pickle

st.set_page_config(page_title="Bank Transaction Anomaly Detection", layout="wide")  # Responsive layout

st.title("📈 Bank Transaction Anomaly Detection App")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
else:
    st.info("Upload a CSV file to begin!")
df = pickle.load(open('df (2).pkl','rb'))

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])

daily_transactions = df.groupby(df['TransactionDate'].dt.date)['TransactionID'].count()
print(daily_transactions)

location = df['Location'].value_counts(5)

st.header('Analysis Of Data')
col1, col2 = st.columns(2)
with col1:
    st.header('Bar Chart Between Channel, TranasactionAmount and Login Attempts')
    st.bar_chart(df,x='Channel',y='TransactionAmount',color='LoginAttempts',x_label='Channel',y_label='TransactionAmount')

with col2:
     fig=sns.relplot(kind='scatter',x=df['AccountBalance'],y=df['TransactionAmount'],palette='summer',hue=df['LoginAttempts'])
     plt.title('TransactionAmount vs AccountBalance')
     st.pyplot(fig)
st.header('Daily Transactions')
st.line_chart(daily_transactions)


df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
daily_transactions = df.groupby(df['TransactionDate'].dt.date)['TransactionID'].count()
monthly_transactions = df.groupby(df['TransactionDate'].dt.to_period('M'))['TransactionID'].count()
Transactions_per_hour = df.groupby(df['TransactionDate'].dt.hour)['TransactionID'].count()



mean = df.groupby(['AccountID'])['TransactionAmount'].transform('mean')
std = df.groupby('AccountID')['TransactionAmount'].transform('std')
df['Z-Score'] = (df['TransactionAmount']-mean)/(std+1e-6)

df['IsNewLocation'] = (df['Location'] != df.groupby(['AccountID'])['Location'].transform('first')).astype(int)
df['IsNewDevice'] = (df['Location'] != df.groupby(['AccountID'])['DeviceID'].transform('first')).astype(int)
df['IsNewMerchant'] = (df['Location'] != df.groupby(['AccountID'])['MerchantID'].transform('first')).astype(int)

df['AmountBalanceRation'] = df['TransactionAmount'] / (df['AccountBalance'] + 1e-6)

df['HighLoginAttempts'] = (df['LoginAttempts'] > 3).astype(int)

Risk_Score = {
    'Student' : 0,
    'Engineer': 1,
    'Doctor': 2,
    'Retired':3
}
df['Risk_Score'] = df['CustomerOccupation'].map(Risk_Score)
df['Transaction_Type'] = df['TransactionType'].map({'Credit':0,'Debit':1})

df = df.drop(columns=['TransactionID','AccountID','Location','DeviceID','TransactionType','IP Address','CustomerOccupation','MerchantID','TransactionDate','PreviousTransactionDate'])

df['Z-Score'] = df['Z-Score'].fillna(0)

#Model For Anomaly Detection
ohe = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
arr = ohe.fit_transform(df[['Channel']])
new_df = pd.DataFrame(arr,columns=ohe.get_feature_names_out())
bank = pd.concat((df,new_df),axis=1)
bank.drop('Channel',axis=1,inplace=True)
pipe = Pipeline([
    ('scaler',StandardScaler())
])

scaled = pipe.fit_transform(bank)

forest_pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('model',IsolationForest(contamination=0.03))
])
forest_pred = forest_pipe.fit_predict(bank)
iso_pca = PCA(n_components=2)
arr2 = iso_pca.fit_transform(scaled)
forest_pcadf = pd.DataFrame(arr2, columns=['PCA1','PCA2'])
forest_pcadf['forest_preds'] = forest_pred
forest_pcadf['forest_preds'] = forest_pcadf['forest_preds'].map({1:'Normal',-1:'Fraud'})
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,6))
sns.scatterplot(data=forest_pcadf,x='PCA1',y='PCA2',hue='forest_preds',legend=False,palette='viridis',label='Normal')
anomilies=forest_pcadf[forest_pcadf['forest_preds'] == 'Fraud']
plt.scatter(
    anomilies['PCA1'],anomilies['PCA2'],
    marker='X',s=100,c='red',edgecolor='black',label='Anomaly'
)
plt.legend()
plt.title('Anomaly Detection Using Isolation Forest')
plt.show()
import plotly.express as px
if not st.checkbox('Hide Graph',False,key=10):

     st.scatter_chart(data = forest_pcadf,x='PCA1',y='PCA2',color='forest_preds',x_label='PCA1',y_label='PCA2')
     anomalies=forest_pcadf[forest_pcadf['forest_preds'] == 'Fraud']


st.write('Total Fraud Detection:', len(forest_pcadf[forest_pcadf['forest_preds'] == 'Fraud']))