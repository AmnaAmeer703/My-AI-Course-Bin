import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('Date_Fruit_Datasets.xlsx')
print(df.head())

print('information of data:', df.info())

print('null values in data:', df.isnull().mean()*100)

print('describe the data:', df.describe().round(2).T)

print('shape of data:', df.shape)

print('duplicated values:', df.duplicated().sum())

print('type of data:', df.dtypes)

print(df['Class'].value_counts())

g = sns.countplot(df['Class'].value_counts())
g.figure.suptitle('Barplot')
g.figure.show()


plt.pie(df['Class'].value_counts(),labels=['DOKOL','SAFAVI','ROTANA','DEGLET','SOGAY','IRAQI','BERHI'])
centre_circle = plt.Circle((0,0),0.75,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# TO CHECK THE CORRELATION BETWEEN NUMERICAL COLUMNS
correlations = df.corr(numeric_only=True)
g = sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues')
g.figure.show()

X = df.drop('Class',axis=1)

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

X = df.drop('Class',axis=1)
y = df['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

logistic_regression = LogisticRegression(random_state=20)
SVC_Classifier = SVC()
Decision_Tree_Classifier = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=30)
Random_Forest_Classifier = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=30)
Gradient_Boosting_Classifier = GradientBoostingClassifier(n_estimators=30,learning_rate=0.1)



logistic_regression.fit(X_train, y_train)
SVC_Classifier.fit(X_train,y_train)
Decision_Tree_Classifier.fit(X_train,y_train)
Random_Forest_Classifier.fit(X_train,y_train)
Gradient_Boosting_Classifier.fit(X_train,y_train)

y_pred1 = logistic_regression.predict(X_test)
y_pred2 = SVC_Classifier.predict(X_test)
y_pred3 = Decision_Tree_Classifier.predict(X_test)
y_pred4 = Random_Forest_Classifier.predict(X_test)
y_pred5 = Gradient_Boosting_Classifier.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report

model_preds = {
    "Logistic Regression": y_pred1,
    "Support Vector Machine": y_pred2,
    "Decision Tree": y_pred3,
    "Random Forest": y_pred4
}

for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")

