#https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

""""Multiple Linear Regression
Until this point, we have predicted a value with linear regression using only one variable. There is a different scenario that we can consider, where we can predict 
using many variables instead of one, and this is also a much more common scenario in real life, where many things can affect some result.

For instance, if we want to predict the gas consumption in US states, it would be limiting to use only one variable, for instance, gas taxes to do it, since more than 
just gas taxes affects consumption. There are more things involved in the gas consumption than only gas taxes, such as the per capita income of the people in a certain
 area, the extension of paved highways, the proportion of the population that has a driver's license, and many other factors. Some factors affect the consumption more 
 than others - and here's where correlation coefficients really help!"""

# https://www.kaggle.com/datasets/harinir/petrol-consumption


#Following what we did with the linear regression, we will also want to know our data before applying multiple linear regression. First, we can import the data with pandas read_csv() method:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_to_file = 'week8/petrol_consumption.csv'
df = pd.read_csv(path_to_file)

print("df.head():  \n",df.head())

print("df.shape: \n" , df.shape)

print("df.describe().round(2).T:    \n",df.describe().round(2).T)


import seaborn as sns # Convention alias for Seaborn

variables = ['Petrol_tax', 'Average_income', 'Paved_Highways','Population_Driver_licence(%)']

for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='Petrol_Consumption', data=df).set(title=f'Regression plot of {var} and Petrol Consumption');
    plt.show()


read = input("Wait here: \n")


plt.figure()
"""We can also calculate the correlation of the new variables, this time using Seaborn's heatmap() to help us spot the strongest and weaker correlations based on warmer (reds) and cooler (blues) tones:"""
correlations = df.corr()
print("correlations...\n" , correlations)
# annot=True displays the correlation values
g = sns.heatmap(correlations, annot=True).set(title='Heat map of Consumption Data - Pearson Correlations')
# Display the plot
plt.show()
read = input("Wait for me....")

"""Preparing the Data
Following what has been done with the simple linear regression, after loading and exploring the data, we can divide it into features and targets. The main difference is that now our features have 4 columns instead of one.

We can use double brackets [[ ]] to select them from the dataframe:"""

y = df['Petrol_Consumption']
X = df[['Average_income', 'Paved_Highways',
       'Population_Driver_licence(%)', 'Petrol_tax']]

SEED = 200
#After setting our X and y sets, we can divide our data into train and test sets. We will be using the same seed and 20% of our data for training:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=SEED)

#After splitting the data, we can train our multiple regression model. Notice that now there is no need to reshape our X data, once it already has more than one dimension:
print("X.shape # (48, 4):     \n", X.shape )   


#To train our model we can execute the same code as before, and use the fit() method of the LinearRegression class:

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

#After fitting the model and finding our optimal solution, we can also look at the intercept:
print("regressor.intercept_......\n", regressor.intercept_)

#And at the coefficients of the features
print("regressor.coef_ " , regressor.coef_)


"""feature_names = X.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients, 
                              index = feature_names, 
                              columns = ['Coefficient value'])
print(coefficients_df)"""

feature_names = X.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients, 
                              index = feature_names, 
                              columns = ['Coefficient value'])
print(coefficients_df)


#In the same way we had done for the simple regression model, let's predict with the test data:
y_pred = regressor.predict(X_test)


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted.....\n" , results)



"""
https://scikit-learn.org/stable/api/sklearn.metrics.html

sklearn.metrics
Score functions, performance metrics, pairwise metrics and distance computations.

Luckily, we don't have to do any of the metrics calculations manually. The Scikit-Learn package already comes with functions that can be used to find out the values 
of these metrics for us. Let's find the values for these metrics using our test data. First, we will import the necessary modules for calculating the MAE and MSE errors. Respectively, the mean_absolute_error and mean_squared_error:
"""

"""After exploring, training and looking at our model predictions - our final step is to evaluate the performance of our multiple linear regression. We want to understand if our predicted values are too far from our actual values. We'll do this in the same way we had previously done, by calculating the MAE, MSE and RMSE metrics.
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')



""""To dig further into what is happening to our model, we can look at a metric that measures the model in a different way, it doesn't consider our individual data values such as MSE, RMSE and MAE, but takes a more general approach to the error, the R2:"""
actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)

"""The R2 doesn't tell us about how far or close each predicted value is from the real data - it tells us how much of our target is being captured by our model."""

"""R2 also comes implemented by default into the score method of Scikit-Learn's linear regressor class. We can calculate it like this:
"""
print(" R2 also comes implemented by default into the score method of Scikit-Learn's linear regressor class...\n", regressor.score(X_test, y_test))



