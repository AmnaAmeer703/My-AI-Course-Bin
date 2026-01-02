#https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
""""

If you had studied longer, would your overall scores get any better?

One way of answering this question is by having data on how long you studied for and what scores you got. We can then try to see if there is a pattern in that data, and if in that pattern, when you add to the hours, it also ends up adding to the scores percentage.

For instance, say you have an hour-score dataset, which contains entries such as 1.5h and 87.5% score. It could also contain 1.61h, 2.32h and 78%, 97% scores. The kind of data type that can have any intermediate value (or any level of 'granularity') is known as continuous data.

Another scenario is that you have an hour-score dataset which contains letter-based grades instead of number-based grades, such as A, B or C. Grades are clear values that can be isolated, since you can't have an A.23, A+++++++++++ (and to infinity) or A * e^12. The kind of data type that cannot be partitioned or defined more granularly is known as discrete data.

Based on the modality (form) of your data - to figure out what score you'd get based on your study time - you'll perform regression or classification.

Regression is performed on continuous data, while classification is performed on discrete data. Regression can be anything from predicting someone's age, the house of a price, or value of any variable. Classification includes predicting what class something belongs to (such as whether a tumor is benign or malignant)."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Let's read the CSV file and package it into a DataFrame:
df = pd.read_csv('week8/student_scores.csv')

#Once the data is loaded in, let's take a quick peek at the first 5 values using the head() method:
print(df.head())

#We can also check the shape of our dataset via the shape property:
print("df.shape:         " , df.shape)

#So, what's the relationship between these variables? A great way to explore relationships between variables is through Scatter plots. We'll plot the hours on the X-axis and scores on the Y-axis, and for each pair, a marker will be positioned based on their values:
df.plot.scatter(x='Hours', y='Scores', title='Scatter Plot of hours and scores percentages');
plt.show()

""""The equation that describes any straight line is:
$$
y = a*x+b
$$
In this equation, y represents the score percentage, x represents the hours studied. b is where the line starts at the Y-axis, also called the Y-axis intercept and a defines if the line is going to be more towards the upper or lower part of the graph (the angle of the line), so it is called the slope of the line.

By adjusting the slope and intercept of the line, we can move it in any direction. Thus - by figuring out the slope and intercept values, we can adjust a line to fit our data!
"""

""""
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html

As the hours increase, so do the scores. There's a fairly high positive correlation here! Since the shape of the line the points are making appears to be straight 
- we say that there's a positive linear correlation between the Hours and Scores variables. How correlated are they? The corr() method calculates and displays the 
 correlations between numerical variables in a DataFrame:"""

print("df.corr():        " , df.corr())

"""Pandas also ships with a great helper method for statistical summaries, and we can describe() the dataset to get an idea of the mean, maximum, minimum, etc. 
values of our columns:

"""

print("df.describe():                    " , df.describe())


print(" df['Scores'] :     " , df['Scores'])
print("  df['Hours']   :    ", df['Hours']   )

"""To separate the target and features, we can attribute the dataframe column values to our y and X variables:"""
#The .reshape() method takes in two arguments: the first is the number of columns you want the dataframe to have, and the second is the number of rows you want the dataframe to have.
y = df['Scores'].values.reshape(-1, 1)
X = df['Hours'].values.reshape(-1, 1)
  

"""Note: df['Column_Name'] returns a pandas Series. Some libraries can work on a Series just as they would on a NumPy array, but not all libraries have this awareness. In some cases, you'll want to extract the underlying NumPy array that describes your data. This is easily done via the values field of the Series."""

print("y :  " , y)
print("X :   " , X)

#Scikit-Learn's linear regression model expects a 2D input, and we're really offering a 1D array if we just extract the values:

print(df['Hours'].values) # [2.5 5.1 3.2 8.5 3.5 1.5 9.2 ... ]
print(df['Hours'].values.shape) # (25,)
#It expects a 2D input because the LinearRegression() class (more on it later) expects entries that may contain more than a single value (but can also be a single value). In either case - it has to be a 2D array, where each element (hour) is actually a 1-element array:

print(X.shape) # (25, 1)
print(X)      # [[2.5] [5.1]  [3.2] ... ]

"""
The method randomly takes samples respecting the percentage we've defined, but respects the X-y pairs, lest the sampling would totally mix up the relationship. Some common train-test splits are 80/20 and 70/30.

Since the sampling process is inherently random, we will always have different results when running the method. To be able to have the same results, or reproducible results, we can define a constant called SEED that has the value of the meaning of life (42):

"""
SEED = 42

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

#Now, if you print your X_train array - you'll find the study hours, and y_train contains the score percentages:

print(X_train) # [[2.7] [3.3] [5.1] [3.8] ... ]
print(y_train) # [[25] [42] [47] [35] ... ]


#Training a Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


#Now, we need to fit the line to our data, we will do that by using the .fit() method along with our X_train and y_train data:

regressor.fit(X_train, y_train)
#If no errors are thrown - the regressor found the best fitting line! The line is defined by our features and the intercept/slope. In fact, we can inspect the intercept and slope by printing the regressor.intecept_ and regressor.coef_ attributes, respectively:

print(regressor.intercept_)

#For retrieving the slope (which is also the coefficient of x):

print(regressor.coef_)

"""
Making Predictions
To avoid running calculations ourselves, we could write our own formula that calculates the value:
"""
def calc(slope, intercept, hours):
    return slope*hours+intercept

score = calc(regressor.coef_, regressor.intercept_, 9.5)
print(score) # [[94.80663482]]

#However - a much handier way to predict new values using our model is to call on the predict() function:

# Passing 9.5 in double brackets to have a 2 dimensional array
score = regressor.predict([[9.5]])
print(score) # 94.80663482

""""Our result is 94.80663482, or approximately 95%. Now we have a score percentage estimate for each and every hour we can think of. But can we trust those estimates? In the answer to that question is the reason why we split the data into train and test in the first place. Now we can predict using our test data and compare the predicted with our actual results - the ground truth results."""

#To make predictions on the test data, we pass the X_test values to the predict() method. We can assign the results to the variable y_pred:

y_pred = regressor.predict(X_test)
#The y_pred variable now contains all the predicted values for the input values in the X_test. We can now compare the actual output values for X_test with the predicted values, by arranging them side by side in a dataframe structure:

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

#Though our model seems not to be very precise, the predicted percentages are close to the actual ones. Let's quantify the difference between the actual and predicted values to gain an objective view of how it's actually performing.


"""
https://scikit-learn.org/stable/api/sklearn.metrics.html

sklearn.metrics
Score functions, performance metrics, pairwise metrics and distance computations.

Luckily, we don't have to do any of the metrics calculations manually. The Scikit-Learn package already comes with functions that can be used to find out the values 
of these metrics for us. Let's find the values for these metrics using our test data. First, we will import the necessary modules for calculating the MAE and MSE errors. Respectively, the mean_absolute_error and mean_squared_error:
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
"""Now, we can calculate the MAE and MSE by passing the y_test (actual) and y_pred (predicted) to the methods. The RMSE can be calculated by taking the square root of 
the MSE, to to that, we will use NumPy's sqrt() method:
"""
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
#We will also print the metrics results using the f string and the 2 digit precision after the comma with :.2f:

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')