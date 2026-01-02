import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('insurance.csv')
print(df)

print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df.dtypes)

print("df.describe().round(2).T:    \n",df.describe().round(2).T)

print(df.isnull().sum()*100)

print(df.duplicated().sum())

print(df['region'].value_counts())

variables = ['age','bmi','children']
for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='charges', data=df).set(title=f'Regression plot of {var} and Charges');
    plt.show()


read = input("Wait here: \n")

g = sns.countplot(df['sex'].value_counts())
g.figure.suptitle("Barplot of Ocean Sex")
g.figure.show()
# Let's find the reltionship between dependent and independent variables

g = sns.scatterplot(data=df,x='bmi',y='charges', hue='sex')
g.figure.suptitle("scatterplot")
g.figure.show()

g = sns.barplot(data=df,x='sex',y='charges')
g.figure.suptitle("barplot")
g.figure.show()

correlations = df.corr(numeric_only=True)
g = sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues')
g.figure.show()

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df,one_hot_df],axis=1)
df_encoded = df_encoded.drop(categorical_columns,axis=1)
print(df_encoded)


X = df_encoded.drop('charges',axis=1)
Y = df_encoded['charges']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

print(X_train)
print(y_train)


#Training a Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

print(regressor.intercept_)


print(regressor.coef_)


feature_names = X.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients, 
                              index = feature_names, 
                              columns = ['Coefficient value'])
print(coefficients_df)


y_pred = regressor.predict(X_test)


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted.....\n" , results)


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)

print(" R2 also comes implemented by default into the score method of Scikit-Learn's linear regressor class...\n", regressor.score(X_test, y_test))

# Construct a pipeline for making Frontend Website
X = df.drop('charges',axis=1)
Y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

print(X_train)
print(y_train)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first',sparse_output=False), ['sex','smoker','region'])
    ],remainder='passthrough')

Regressor = LinearRegression()
pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('Regressor',Regressor)
])

pipe.fit(X_train,y_train)
predictions = pipe.predict(X_test)
r2_score = r2_score(y_test,predictions)
print(r2_score)
