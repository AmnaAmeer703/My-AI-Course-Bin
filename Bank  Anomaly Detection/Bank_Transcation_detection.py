import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bank_transactions_data_2.csv')
print(df)

print(df.info())

print(df.describe())

print(df.isnull().mean()*100)

# Correlation
print(df.corr(numerical_only=True))
sns.heatmap(df.corr(numeric_only=True),annot=True,fmt='.1f',linewidth=2,square=False)

categorical_columns = [var for var in df.columns if df[var].dtypes == 'object']
numerical_columns = [var for var in df.columns if df[var].dtypes != 'object']

# Distribution of Data
df[numerical_columns].hist(bins=30, figsize=(12,10))

# Checking Outliers
X = df[numerical_columns]
for columns in X:
    fig, axs = plt.subplots(figsize=(2,2))
    sns.boxplot(X[columns])

# Density
X = df[numerical_columns]
for columns in X:
    fig, axs = plt.subplots(figsize=(2,2))
    sns.kdeplot(X[columns])

# Exploratory Data Analysis

df['CustomerOccupation'].value_counts()

plt.pie(df['TransactionType'].value_counts(),labels=['Debit','Credit'],autopct='%1.f%%')
plt.show()

# Countplot of Column 'Location'

sns.countplot(data=df, x='Location')
plt.xticks(rotation=90)
plt.show()

# Countplot of Column 'Channel'
sns.countplot(data=df, x='Channel')
plt.show()

# pie chart
plt.pie(df['CustomerOccupation'].value_counts(),labels=['Student','Doctor','Enginner','Retired'],autopct='%1.f%%',pctdistance=0.85,explode=(0,0,0.2,0))
centre_circle = plt.Circle((0,0),0.75,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# line plot between column 'TranslationDate' and TranslationAmount'
sns.lineplot(x=df['TransactionDate'],y=df['TransactionAmount'])

# regplot
sns.relplot(kind='scatter',x=df['AccountBalance'],y=df['TransactionAmount'],palette='summer',hue=df['LoginAttempts'])
plt.title('TransactionAmount vs AccountBalance')

# barplot between column 'Channel' and 'TranslationAmount', hue='LoginAttempts'

sns.barplot(x=df['Channel'],y=df['TransactionAmount'],hue=df['LoginAttempts'])
plt.show()

# Histogram plot by using groupby method between column 'DeviceID' and 'IP Address'
sns.histplot((df.groupby(['DeviceID'])['IP Address'].nunique().values))

# Histogram plot between 'AccountID' and 'Location' using groupby method
sns.histplot(df.groupby(['AccountID'])['Location'].nunique().values)

# Daily, Monthly Transaction, Hourly Transaction
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])

daily_transactions = df.groupby(df['TransactionDate'].dt.date)['TransactionID'].count()
print(daily_transactions)

monthly_transactions = df.groupby(df['TransactionDate'].dt.to_period('M'))['TransactionID'].count()
print(monthly_transactions)

transaction_per_hour = df.groupby(df['TransactionDate'].dt.hour)['TransactionID'].count()
print(transaction_per_hour)

df['Day_Of_Week'] = df['TransactionDate'].dt.day_name()
transaction_per_day = df['Day_Of_Week'].value_counts()
transaction_per_day

df['TransactionHours'] = df['TransactionDate'].dt.hour

df['TimeElapsed'] = df.groupby(['AccountID'])['TransactionDate'].diff().dt.total_seconds().fillna(0) / 60

# lineplot to represent Daily Transaction
daily_transactions.plot(kind='line')
plt.xlabel('Date')
plt.ylabel('Number of transactions')
plt.show()

# Detect Fraud
# Z-Score w.r.t each AccountID

mean = df.groupby(['AccountID'])['TransactionAmount'].transform('mean')
std = df.groupby('AccountID')['TransactionAmount'].transform('std')
df['Z-Score'] = (df['TransactionAmount'] - mean ) / (std + 1e-6)

# Does a User use a new Location, Device, IP Adress and Merchant in Transaction
df['IsNewLocation'] = (df['Location'] != df.groupby(['AccountID'])['Location'].transform('first')).astype(int)
df['IsNewDevice'] = (df['DeviceID'] != df.groupby(['AccountID'])['DeviceID'].transform('first')).astype(int)
df['IsNewMerchant'] = (df['MerchantID'] != df.groupby(['AccountID'])['MerchantID'].transform('first')).astype(int)

# Transaction Amount and Balance Ration
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

# Drop Irrelevent Columns
df = df.drop(columns = ['TransactionID','AccountID','Location','DeviceID','IP Address','CustomerOccupation','MerchantID','TransactionDate','PreviousTransactionDate','Day_Of_Week'])

print(df.isnull().mean()*100)

df['Z-Score'] = df['Z-Score'].fillna(0)

# Anomaly Detection Usin K-Means 

import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
arr = ohe.fit_transform(df[['Channel']])
new_df = pd.DataFrame(arr, columns = ohe.get_feature_names_out())
bank = pd.concat((df, new_df), axis=1)

bank.drop('Channel',axis=1,inplace=True)

pipe = Pipeline([
    ('scaler', StandardScaler())
])

scaled = pipe.fit_transform(bank)

from sklearn.cluster import KMeans
l = []
for i in range(1,15):
    km = KMeans(n_clusters = i)
    km.fit_predict(scaled)
    l.append(km.inertia_)
sns.lineplot(l)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
arr = pca.fit_transform(scaled)
pca_df = pd.DataFrame(arr, columns = ['PCA1','PCA2'])

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KMeans(n_clusters=6))
])

labels = pipe2.fit_predict(scaled)
pca_df['kmean_labels'] = labels

model = pipe2.named_steps['model']

centers = model.fit_transform(scaled)
min_dict = np.min(centers, axis=1)
threshold = np.percentile(min_dict, 97)
pca_df['DistanceFromClosetCluster'] = min_dict
pca_df['IsAnomaly'] = (pca_df['DistanceFromClosetCluster'] > threshold).astype(int)

plt.figure(figsize=(10,6))
sns.scatterplot(
    x='PCA1',y='PCA2',
    hue='kmean_labels',
    data = pca_df,
    palette='viridis',
    alpha=0.6
)

#Overlay Anomalies
anomilies = pca_df[pca_df['IsAnomaly'] == 1]
plt.scatter(
    anomilies['PCA1'],anomilies['PCA2'],
    marker='X', s=100, c='red', edgecolor = 'black',
    label = 'Amonaly'
    
)

plt.title('K-Means Cluster With Anomilies (PCA)')
plt.legend()
plt.show()
print('total anomilies detected:', len(anomilies))

# Result: Total Anomaly Detection are 76

# Anomaly Detection using Isolation Forest

from sklearn.ensemble import IsolationForest

forest_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', IsolationForest(contamination=0.03))
])

forest_pred = forest_pipe.fit_predict(bank)

iso_pca = PCA(n_components=2)
arr2 = iso_pca.fit_transform(scaled)
forest_pcadf = pd.DataFrame(arr2, columns=['PCA1','PCA2'])
forest_pcadf['forest_preds'] = forest_pred
forest_pcadf['forest_preds'] = forest_pcadf['forest_preds'].map({1:'Normal',-1:'Fraud'})

plt.figure(figsize=(12,6))
sns.scatterplot(data = forest_pcadf, x='PCA1',y='PCA2', hue='forest_preds',legend=False,palette='viridis',label='Normal')
anomilies = forest_pcadf[forest_pcadf['forest_preds'] == 'Fraud']
plt.scatter(
    anomilies['PCA1'], anomilies['PCA2'],
    marker = 'X', s=100, c='red', edgecolor = 'black', label = 'Anomaly'
)
plt.legend()
plt.title('Anomaly Detection Using Isolation Forest')
plt.show()
print('Total fraud Detection: ', len(forest_pcadf[forest_pcadf['forest_preds'] == 'Fraud']))

# Result: Total Fraud Detection : 76

# Anomaly Detection Using DBSCAN

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(scaled)
distances, indices = nbrs.kneighbors(scaled)

# Sort the 5th Nearest Neighbors distances
sorted_distances = np.sort(distances[:,4], axis=0)

plt.figure(figsize=(8,5))
plt.plot(sorted_distances
        , marker = 'o', linestyle='-')
plt.title('k-distance plot', fontsize=14)
plt.xlabel('Points',fontsize=12)
plt.ylabel('5th Nearest Neighbors Distance', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

db = DBSCAN(eps =3)
dbscan_preds = db.fit_predict(scaled)

dbpca = PCA(n_components=2)
arr3 = dbpca.fit_transform(scaled)
dbscan_pcadf = pd.DataFrame(arr3, columns=['PCA1','PCA2'])
dbscan_pcadf['db_preds'] = dbscan_preds
dbscan_pcadf['db_preds'] = np.where(dbscan_pcadf['db_preds'] == -1, 'Fraud','Normal')

plt.figure(figsize=(12,6))
sns.scatterplot(data = dbscan_pcadf, x='PCA1',y='PCA2',hue='db_preds',legend=False, label='Normal')
anomalies = dbscan_pcadf[dbscan_pcadf['db_preds'] == 'Fraud']

plt.scatter(
    anomalies['PCA1'],anomalies['PCA2'],
    marker = 'X', s=100, c='red', edgecolor='black',label='Anomaly'
)
plt.legend()
plt.title('Anomaly Detection Using DBSCAN')
plt.show()
print('Total fraud Detection:',len(dbscan_pcadf[dbscan_pcadf['db_preds']=='Fruad']))

# Anomaly Detection Using Local Outlier Factor (LOF)

from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=6)
lof_labels = lof.fit_predict(scaled)

lof_pca = PCA(n_components=2)
arr4 = lof_pca.fit_transform(scaled)
lof_pcadf = pd.DataFrame(arr4, columns=['PC1','PC2'])
lof_pcadf['lof_labels'] = lof_labels
lof_pcadf['lof_labels'] = np.where(lof_pcadf['lof_labels'] == -1, 'Fraud','Normal')

plt.figure(figsize=(12,6))
sns.scatterplot(data = lof_pcadf, x='PC1',y='PC2',hue='lof_labels',palette='viridis',legend=False,label='Normal')
anomalies = lof_pcadf[lof_pcadf['lof_labels'] == 'Fraud']

plt.scatter(
    anomalies['PC1'],anomalies['PC2'],
    marker = 'X',s=100, c='red', edgecolor='black',label='Anomaly'
)
plt.legend()
plt.title('Anomaly Detection Using Local Outlier Factor')
plt.show()
print('Total Anomalies Detected:',len(lof_pcadf[lof_pcadf['lof_labels']=='Fraud']))

# Result: Total Anomaly Detection: 61




