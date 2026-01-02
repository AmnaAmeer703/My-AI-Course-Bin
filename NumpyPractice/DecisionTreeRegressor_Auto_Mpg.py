import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('NumpyPractice\Auto_mpg.csv')
print(df)

print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df.dtypes)

print("df.describe().round(2).T:    \n",df.describe().round(2).T)

print(df.isnull().sum()*100)


print(df.duplicated().sum())

print(df['car name'].value_counts())
print(df['origin'].value_counts())
print(df['model year'].value_counts())
print(df['horsepower'].value_counts())

df['horsepower'] = pd.to_numeric(df['horsepower'].replace('?',np.nan),errors = 'coerce')

print(df.dtypes)
print(df['horsepower'].median())
df['horsepower'] = df['horsepower'].replace(np.nan, 93.5) # as the calcultaed median of column horsepoer is '93.5' and we fill the null values of horsepower column with median value of horsepower
print(df.isnull().sum()*100)


g = sns.countplot(df['origin'].value_counts())
g.figure.suptitle("Barplot of car name")
g.figure.show()
# Let's find the reltionship between dependent and independent variables

g = sns.scatterplot(data=df,x='acceleration',y='mpg')
g.figure.suptitle("scatterplot acceleration and mpg")
g.figure.show()


g = sns.scatterplot(data=df,x='horsepower',y='mpg')
g.figure.suptitle("scatterplot horsepower and mpg")
g.figure.show()

correlations = df.corr(numeric_only=True)
print("correlations...\n" , correlations)


g = sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues').set(title='Heat map of Consumption Data - Pearson Correlations')
g.figure.suptitle("Correlation Matrix")
g.figure.show()


categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df,one_hot_df],axis=1)
df_encoded = df_encoded.drop(categorical_columns,axis=1)
print(df_encoded)

print(df_encoded.dtypes)
print(df_encoded.isnull().sum()*100)



X = df_encoded.drop('mpg',axis=1)
Y = df_encoded['mpg']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

X_scaled = scaler.transform(X.values)

print(X_scaled[0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=30,random_state=42)

print(X_train)
print(y_train)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(criterion='squared_error',max_depth=10)
dt.fit(X_train,y_train)
from sklearn.metrics import r2_score
pred1 = dt.predict(X_test)
print(r2_score(y_test,pred1))

