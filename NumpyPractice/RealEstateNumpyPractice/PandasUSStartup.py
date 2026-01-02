import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Lenovo/Documents/GitHub/My-AI-Course-Bin/NumpyPractice/Startups in 2021 end.csv",delimiter=",")

print(df)

print("df - data types" , df.dtypes)

print("df.info():   " , df.info() )

# display the last three rows
print('Last three Rows:')
print(df.tail(3))

# display the first three rows
print('First Three Rows:')
print(df.head(3))
print()

#Summary of Statistics of DataFrame using describe() method.
print("Summary of Statistics of DataFrame using describe() method", df.describe())

#Counting the rows and columns in DataFrame using shape(). It returns the no. of rows and columns enclosed in a tuple.
print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()


df['Valuation ($B)'] = df['Valuation ($B)'].str.replace('$','')
print(df['Valuation ($B)'])
print(df)
df['Valuation ($B)'] = df['Valuation ($B)'].astype(float)
print(df.dtypes)
df['Date Joined'] = pd.to_datetime(df['Date Joined'])

DateJoined = df['Date Joined']
print('access the Column: df: ')
print(DateJoined)

data = df.head(50)

Industry = df['Industry']
print("access the Name column: df : ")
print(Industry)
print()

Company = df['Company']
print('access the column: df: ')
print(Company)
print()

Company_Valuation = df[['Company','Valuation ($B)']]
print("access multiple columns: df : ")
print(Company_Valuation)
print()

second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

second_row8 = df.loc[df['Company'] == 'SpaceX','Valuation ($B)':'Select Investors']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

row = df.loc[932]
print("#Selecting a single row using .loc")
print(row)
print()

three_rows = df.loc[[500, 600]]
print("#Selecting multiple rows using .loc")
print(three_rows)
print()

row6 = df.loc[:931,['Company','Select Investors']]
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

# Data Visualization Using Seaborn
sns.set_theme(style='darkgrid')


# Create a plot
sns.lineplot(x='City', y='Valuation ($B)', data=data)
plt.show()

# Other themes can be set similarly
sns.set_theme(style='whitegrid')
sns.lineplot(x='City', y='Valuation ($B)', data=data)
plt.show()

sns.set_theme(style='dark')
sns.lineplot(x='City', y='Valuation ($B)', data=data)
plt.show()

sns.set_theme(style='white')
sns.lineplot(x='City', y='Valuation ($B)', data=data)
plt.show()

sns.set_theme(style='ticks')
sns.lineplot(x='City', y='Valuation ($B)', data=data)
plt.show()

# Customize the theme
sns.set_theme(style='darkgrid', rc={'axes.facecolor': 'grey', 'grid.color': 'white'})

# Create a plot
sns.lineplot(x='City', y='Valuation ($B)', data=data)
plt.show()

g=sns.displot(data=data, x="City" , y="Valuation ($B)" , hue="City",  kind='hist'  )
g.figure.suptitle("sns.displot(data=df, x=City , y=Valuation ($B) , hue=City,  kind='hist'  )"  )

# Display the plot
g.figure.show()
#g.figure.clear()

#kind='kde'
g=sns.displot(data=data, x="Date Joined" , y="Valuation ($B)" , kind='kde'  )
g.figure.suptitle("sns.displot(data=df, x=City , y=Date Joined, kind='kde'  )"  )

# Display the plot
g.figure.show()

#kind='kde'
g=sns.kdeplot(data=data, x="Valuation ($B)")
g.figure.suptitle("sns.kdeplot(data=df,x=Valuation ($B))"  )

# Display the plot
g.figure.show()
#g.figure.clear()
g = sns.histplot(data=data, x='City', y='Valuation ($B)', hue='Country', multiple="stack")
g.figure.suptitle("sns.histplot(data=df, x='City', y='Valuation ($B)', hue='Country', multiple=stack)"  )
# Display the plot
g.figure.show()
#g.figure.clear()

g = sns.scatterplot(x='City', y='Valuation ($B)', data=data)
g.figure.suptitle("sns.scatterplot(x='City', y='Valuation (4B)', data=df)"  )
g.figure.show()
#g.figure.clear()

g=sns.lineplot(data=data, x="City" , y="Valuation ($B)"  )
g.figure.suptitle("sns.lineplot(data=df, x=City , y=Valuation ($B) )"  )
# Display the plot
g.figure.show()
#g.figure.clear()

g=sns.barplot(data=data, x="City", y="Valuation ($B)", legend=False)
g.figure.suptitle("sns.barplot(data=df, x=City, y=Valuation ($B), legend=False)"  )
# Display the plot
g.figure.show()
#g.figure.clear()


g=sns.catplot(data=data, x="City", y="Valuation ($B)")
g.figure.suptitle("sns.catplot(data=df, x=state, y=price)"  )
# Display the plot
g.figure.show() 
#g.figure.clear()

#.pivot(index="Model", columns="agency", values="price")
glue = data.pivot(columns="City", values="Valuation ($B)")

g=sns.heatmap(glue)
g.figure.suptitle("sns.heatmap(glue)  - glue = df.pivot(columns=City, values=Valuation ($B))"  )
# Display the plot
g.figure.show()
