import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("Acoustic_Extinguisher_Fire_Dataset\Acoustic_Extinguisher_Fire_Dataset.xlsx")
print('df:',df.head())

print('information of data:', df.info())

print('null values in data:', df.isnull().mean()*100)

print('describe the data:', df.describe().round(2).T)

print('shape of data:', df.shape)

print('duplicated values:', df.duplicated().sum())

print('type of data:', df.dtypes)

print(df['FUEL'].value_counts())

g = sns.countplot(df['FUEL'].value_counts())
g.figure.suptitle('Barplot')
g.figure.show()

plt.pie(df['FUEL'].value_counts(),labels=['gasoline','thinner','kerosene','lpg'],autopct='%1.0f%%',shadow=True)
plt.show()

print(df['STATUS'].value_counts())

g = sns.countplot(df['STATUS'].value_counts())
g.figure.suptitle('Barplot')
g.figure.show()


# TO CHECK THE CORRELATION BETWEEN NUMERICAL COLUMNS
correlations = df.corr(numeric_only=True)
g = sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues')
g.figure.show()


X = df[['DISTANCE','DESIBEL','AIRFLOW','FREQUENCY']]

# BOX PLOT FOR NUMERICAL DATA TO CHECK THE OUTLIERS
cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(2,2))
    sns.boxplot(X[columns])

# HISTOGRAM CHART TO CHECK THE DISTRIBUTION IF NUMERICAL DATA

cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(2,2))
    sns.histplot(X[columns])

Fuel = df.groupby('FUEL')
FUEL_STATUS = df.groupby('FUEL')['STATUS']
print(FUEL_STATUS)

X = df.drop('STATUS',axis=1)
y = df['STATUS']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first',sparse_output=False), ['FUEL'])
    ],remainder='passthrough')

log_regressor = LogisticRegression()
pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('LogisticRegressor',log_regressor)
])

pipe.fit(X_train,y_train)
predictions = pipe.predict(X_test)

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(" Model Evaluation using Confusion Matrix : " , confusion_matrix)

class_names=[0,1]
fig, ax = plt.subplots()

# create heatmap
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
plt.Text(0.5,257.44,'Predicted label');

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

