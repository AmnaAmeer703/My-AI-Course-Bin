import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("NumpyPractice\RealEstateNumpyPractice\FastFoodRestaurants.csv",delimiter=",")
print(df)

print("df - data types" , df.dtypes)

print("df.info():   " , df.info() )

print("Summary of Statistics of DataFrame using describe() method", df.describe())

print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()

city = df['city']
print("access the Name column: df : ")
print(city)
print()

city_name = df[['city','name']]
print("access multiple columns: df : ")
print(city_name)
print()

second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

second_row8 = df.loc[df['city'] == 'Athens','name':'websites']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

row = df.loc[6231]
print("#Selecting a single row using .loc")
print(row)
print()

three_rows = df.loc[[1637, 5379]]
print("#Selecting multiple rows using .loc")
print(three_rows)
print()

row6 = df.loc[:5379,['city','websites']]
print("#Selecting multiple columns using .loc")
print(row6)
print()

second_row2 = df.iloc[[1, 3,5]]
print("#Selecting multiple rows using .iloc")
print(second_row2)
print()

second_row3 = df.iloc[2:5]
print("#Selecting a slice of rows using .iloc")
print(second_row3)
print()

second_row5 = df.iloc[:,2]
print("#Selecting a single column using .iloc")
print(second_row5)
print()

#Selecting multiple columns using .iloc
second_row6 = df.iloc[:,[2,4]]
print("#Selecting multiple columns using .iloc")
print(second_row6)
print()

#Selecting a slice of columns using .iloc
second_row7 = df.iloc[:,2:4]
print("#Selecting a slice of columns using .iloc")
print(second_row7)
print()

#Combined row and column selection using .iloc
second_row8 = df.iloc[[1, 3,5],2:4]
print("#Combined row and column selection using .iloc")
print(second_row8)
print()

# Data Visualization Seaborn and Matplotlib

sns.set_theme(style='darkgrid')

# Create a plot
sns.lineplot(x='latitude', y='longitude', data=df)
plt.show()


sns.set_theme(style='whitegrid')
sns.lineplot(x='latitude', y='longitude', data=df)
plt.show()

sns.set_theme(style='dark')
sns.lineplot(x='latitude', y='longitude', data=df)
plt.show()

sns.set_theme(style='white')
sns.lineplot(x='latitude', y='longitude', data=df)
plt.show()

sns.set_theme(style='ticks')
sns.lineplot(x='latitude', y='longitude', data=df)
plt.show()

sns.set_theme(style='darkgrid', rc={'axes.facecolor': 'grey', 'grid.color': 'white'})

sns.lineplot(x='latitude', y='longitude', data=df)
plt.show()

