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

variables = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin']

for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='mpg', data=df).set(title=f'Regression plot of {var} and Petrol Consumption');
    plt.show()


read = input("Wait here: \n")


plt.figure()

correlations = df.corr(numeric_only=True)
print("correlations...\n" , correlations)

g = sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues').set(title='Heat map of Consumption Data - Pearson Correlations')
plt.show()

g = sns.countplot(df['origin'].value_counts()).set(title='Barplot of car name')
plt.show()
# Let's find the reltionship between dependent and independent variables

g = sns.scatterplot(data=df,x='acceleration',y='mpg').set(title='Scatterplot acceleration and mpg')
plt.show()


g = sns.scatterplot(data=df,x='horsepower',y='mpg').set(title='Scatterplot horsepower and mpg')
plt.show()



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


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

print(X_train)
print(y_train)


#Training a Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=True,fit_intercept=False)

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




