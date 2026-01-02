import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv')
print(df.head())

print('information of data:', df.info())

print('null values in data:', df.isnull().mean()*100)

print('describe the data:', df.describe().round(2).T)

print('shape of data:', df.shape)

print('duplicated values:', df.duplicated)

print('type of data:', df.dtypes)

g = sns.countplot(df['Target'].value_counts())
g.figure.suptitle('Barplot')
g.figure.show()

g = sns.countplot(df['Gender'].value_counts())
g.figure.suptitle("Barplot of Gender")
g.figure.show()
# Let's find the reltionship between dependent and independent variables

g = sns.scatterplot(data=df,x='Previous qualification',y='Course', hue='Gender')
g.figure.suptitle("scatterplot")
g.figure.show()

g = sns.barplot(data=df,x='Gender',y='Target')
g.figure.suptitle("barplot")
g.figure.show()

correlations = df.corr(numeric_only=True)
g = sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues')
g.figure.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])


X = df.drop('Target',axis=1) # Features
y = df['Target'] # Target variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=16)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(" Model Evaluation using Confusion Matrix : " , cnf_matrix)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1,2] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
plt.Text(0.5,257.44,'Predicted label');

from sklearn.metrics import classification_report
target_names = ['Dropout', 'Graduate','Enrolled']
print(classification_report(y_test, y_pred, target_names=target_names))

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()