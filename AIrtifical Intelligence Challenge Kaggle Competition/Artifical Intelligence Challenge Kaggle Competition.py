import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

submission = pd.read_csv('AI Challenge Kaggle Competition/sample_submission (1).csv')
df = pd.read_csv('AI Challenge Kaggle Competition/AI_Thought_Chain_Dataset_1000.csv')

print(df)

print('Data dtypes:', df.dtypes)

print('Information of data:', df.info())

print('Null Values:', df.isnull().mean()*100)

# To Check The Value Counts of Column 'question'
print(df['question'].value_counts())

# To Check The Value Counts of Column 'question'
print(df['reasoning_trace'].value_counts())

# Feature Engineering

# Extract the number form 'questions' = 'What is N + N ?'

import re
def extract_numbers(n):
    numbers = re.findall(r'\d+',n)
    return int(numbers[0]) if numbers else 0
df['extracted_numbers_from_question'] = df['question'].apply(extract_numbers)

# Let's find the length of reansoning trace

df['reasoning_trace_length'] = df['reasoning_trace'].apply(len)
print(df['reasoning_trace_length'])

# Let's check if reasoning trace get longer if number increase

plt.plot(df['extracted_numbers_from_question'],df['reasoning_trace_length'],color='blue')
plt.title('bar plot')
plt.xlabel('Numbers')
plt.ylabel('reasoning trace length')
plt.show()

# check the Distribution of Correct Answers

plt.hist(df['correct_answer'])
plt.title('Distibution of Correct Answer')
plt.xlabel('Answer value')
plt.show()

X = df[['extracted_numbers_from_question']]
y = df['correct_answer']

# Let's apply Machine Learning Model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)

pred = LR.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test,pred)
mse = mean_squared_error(y_test,pred)
rmse = np.sqrt(y_test,pred)

print(mae)
print(mse)
print(rmse)

r2 = r2_score(y_test,pred)
print(r2)

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
RFR.fit(X_train,y_train)
y_pred = RFR.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(y_test,pred)

print(mae)
print(mse)
print(rmse)

r2 = r2_score(y_test,y_pred)
print(r2)

# To Submit the Projuct in Competition we have to make a Submission File:
# This Code will not run on your vs code beacuse it is use to make a submission file to submit in copetition


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

if not submission.empty:
    submission['extracted_numbers_from_question'] = submission['id']
    y_pred = RFR.predict(submission[['extracted_numbers_from_question']])
    submission['correct_answer'] = y_pred
    final_submission = submission[['id','correct_answer']]
    final_submission.to_csv('submission.csv',index=False)
    print("\nSubmission file 'submission.csv' generated sucessfully.")
    print(final_submission.head())