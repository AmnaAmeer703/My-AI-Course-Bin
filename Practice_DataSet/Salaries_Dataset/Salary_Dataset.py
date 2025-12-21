import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ds_salaries.csv")
print('df:',df.head())

print('information of data:', df.info())

print('null values in data:', df.isnull().mean()*100)

print('describe the data:', df.describe().round(2).T)

print('shape of data:', df.shape)

print('duplicated values:', df.duplicated().sum())

print('type of data:', df.dtypes)

print(df['experience_level'].value_counts())
sns.countplot(data=df,x='experience_level')
plt.show()

df['experience_level'].value_counts().plot.pie(autopct="%1.1f%%",figsize=(6,6))
plt.show()

print(df['employment_type'].value_counts())
sns.countplot(data=df,x='employment_type')
plt.show()

Top_20_Jobs = df['job_title'].value_counts().head(30)
print(Top_20_Jobs)

sns.barplot(x=Top_20_Jobs.index,y=Top_20_Jobs.values)
plt.xticks(rotation=90,fontsize=7)
plt.show()

print(df['salary_currency'].value_counts())

Top_Company_Residence = df['employee_residence'].value_counts().head(30)
print(Top_Company_Residence)
sns.barplot(x=Top_Company_Residence.index,y=Top_Company_Residence.values)
plt.xticks(rotation=90,fontsize=7)
plt.show()


print(df['company_location'].value_counts())

print(df['company_size'].value_counts())
sns.countplot(data=df,x='company_size')
plt.show()

# ===============================================
# Let's See in which Job Domain Salaries are High Using Groupby Method

Job_domain_higher_salaries = df.groupby('job_title')['salary'].sum().nlargest(10)
print(Job_domain_higher_salaries)

Job_domain_higher_salaries.plot(kind='bar')

Job_domain_higher_salaries_with_experience = df.groupby(['job_title','experience_level'])['salary'].sum().nlargest(10).reset_index()
Job_domain_higher_salaries_with_experience

g = sns.barplot(data=Job_domain_higher_salaries_with_experience,x='job_title',y='salary',hue='experience_level')
g.figure.suptitle('Barplot')
g.figure.show()



# ===============================================
# TO CHECK THE CORRELATION BETWEEN NUMERICAL COLUMNS
correlations = df.corr(numeric_only=True)
g = sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues')
g.figure.show()

sns.boxplot(data=df,x='salary')
plt.show()

X = df.drop('salary',axis=1)
Y = df['salary']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


preprocessor = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'), ['experience_level','employment_type','job_title','salary_currency','employee_residence','company_location','company_size'])
    ],remainder='passthrough')

LinearRegression = LinearRegression()
pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('LinearRegressor',LinearRegression)
])

pipe.fit(X_train,y_train)
predictions = pipe.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
#We will also print the metrics results using the f string and the 2 digit precision after the comma with :.2f:

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

SVC_Regressor = SVC()
Decision_Tree_Regressor = DecisionTreeRegressor()
Random_Forest_Regressor = RandomForestRegressor(n_estimators=300,random_state=42)
Gradient_Boosting_Regressor = GradientBoostingRegressor(n_estimators=300,learning_rate=0.1)

#==============================================
print("Accuracy For Support Vector Machine")

pipe1 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('SVC',SVC_Regressor)
])

pipe1.fit(X_train,y_train)
pred1 = pipe1.predict(X_test)


mae = mean_absolute_error(y_test, pred1)
mse = mean_squared_error(y_test, pred1)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred1)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

# =====================================================

print("Accuracy For Random Forest Regressor")

pipe2 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('RandomForestRegressor',Random_Forest_Regressor)
])

pipe2.fit(X_train,y_train)
pred2 = pipe2.predict(X_test)

print('Acuuracy For Random Forest Regressor')
mae = mean_absolute_error(y_test, pred2)
mse = mean_squared_error(y_test, pred2)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred2)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')
# =====================================================
print('Acuuracy For Decesion Tree Regressor')

pipe3 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('Decesion_Tree_Regressor',Decision_Tree_Regressor)
])

pipe3.fit(X_train,y_train)
pred3 = pipe3.predict(X_test)

mae = mean_absolute_error(y_test, pred3)
mse = mean_squared_error(y_test, pred3)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred3)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

# =====================================================
print('Acuuracy For Gradient Boosting Regressor')

pipe4 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('Gradiant_Boosting_Regressor',Gradient_Boosting_Regressor)
])

pipe4.fit(X_train,y_train)
pred4 = pipe4.predict(X_test)
mae = mean_absolute_error(y_test, pred4)
mse = mean_squared_error(y_test, pred4)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred4)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')