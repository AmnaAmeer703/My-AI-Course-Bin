import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("NumpyPractice\RealEstateNumpyPractice\RealEstate-USA.csv",delimiter=",")
print(df)

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

city = df['city']
print("access the Name column: df : ")
print(city)
print()

city_housesize_price = df[['city','house_size','price']]
print("access multiple columns: df : ")
print(city_housesize_price)
print()

second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df.loc[[1, 3]]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

#Selecting a slice of rows using .loc
second_row3 = df.loc[1:5]
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()


#Conditional selection of rows using .loc
second_row4 = df.loc[df['city'] == 'Washington Court House']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df.loc[:8000,'brokered_by']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:7000,['house_size','price']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df.loc[:1500,'status':'price']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df.loc[df['city'] == 'Washington Court House','status':'house_size']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

df_index_col = pd.read_csv('NumpyPractice\RealEstateNumpyPractice\RealEstate-USA.csv',delimiter=",",index_col='zip_code')

print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())
# Second cycle - with index_col as property_id

#Selecting a single row using .loc
second_row = df_index_col.loc[601]
print("#Selecting a single row using .loc")
print(second_row)
print()


#Conditional selection of rows using .loc
second_row4 = df_index_col.loc[df_index_col['city'] == 'Washington Court House']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Combined row and column selection using .loc
second_row8 = df_index_col.loc[df_index_col['city'] == 'Washington Court House','brokered_by':'house_size']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

second_row = df_index_col.iloc[0]
print("#Selecting a single row using .iloc")
print(second_row)
print()

#Selecting multiple rows using .iloc
second_row2 = df_index_col.iloc[[1, 3,5]]
print("#Selecting multiple rows using .iloc")
print(second_row2)
print()

#Selecting a slice of rows using .iloc
second_row3 = df_index_col.iloc[2:5]
print("#Selecting a slice of rows using .iloc")
print(second_row3)
print()

#Selecting a single column using .iloc
second_row5 = df_index_col.iloc[:,2]
print("#Selecting a single column using .iloc")
print(second_row5)
print()

#Selecting multiple columns using .iloc
second_row6 = df_index_col.iloc[:,[2,4]]
print("#Selecting multiple columns using .iloc")
print(second_row6)
print()

#Selecting a slice of columns using .iloc
second_row7 = df_index_col.iloc[:,2:4]
print("#Selecting a slice of columns using .iloc")
print(second_row7)
print()

#Combined row and column selection using .iloc
second_row8 = df_index_col.iloc[[1, 3,5],2:4]
print("#Combined row and column selection using .iloc")
print(second_row8)
print()

# Data Visualization Using Seaborn
sns.set_theme(style='darkgrid')


# Create a plot
sns.lineplot(x='house_size', y='price', data=df)
plt.show()

# Other themes can be set similarly
sns.set_theme(style='whitegrid')
sns.lineplot(x='house_size', y='price', data=df)
plt.show()

sns.set_theme(style='dark')
sns.lineplot(x='house_size', y='price', data=df)
plt.show()

sns.set_theme(style='white')
sns.lineplot(x='house_size', y='price', data=df)
plt.show()

sns.set_theme(style='ticks')
sns.lineplot(x='house_size', y='price', data=df)
plt.show()

# Customize the theme
sns.set_theme(style='darkgrid', rc={'axes.facecolor': 'grey', 'grid.color': 'white'})

# Create a plot
sns.lineplot(x='house_size', y='price', data=df)
plt.show()

g=sns.displot(data=df, x="state" , y="price" , hue="status",  kind='hist'  )
g.figure.suptitle("sns.displot(data=df, x=state , y=price , hue=status,  kind='hist'  )"  )

# Display the plot
g.figure.show()
#g.figure.clear()

#kind='kde'
g=sns.displot(data=df, x="price" , y="prev_sold_date" , kind='kde'  )
g.figure.suptitle("sns.displot(data=df, x=price , y=prev_sold_date , kind='kde'  )"  )

# Display the plot
g.figure.show()

#kind='kde'
g=sns.kdeplot(data=df, x="price")
g.figure.suptitle("sns.kdeplot(data=df, x=price)"  )

# Display the plot
g.figure.show()
#g.figure.clear()
g = sns.histplot(data=df, x='state', y='price', hue='state', multiple="stack")
g.figure.suptitle("sns.histplot(data=df, x='state', y='price', hue='state', multiple=stack)"  )
# Display the plot
g.figure.show()
#g.figure.clear()

g = sns.scatterplot(x='state', y='price', data=df)
g.figure.suptitle("sns.scatterplot(x='state', y='price', data=df)"  )
g.figure.show()
#g.figure.clear()

g=sns.lineplot(data=df, x="state" , y="price"  )
g.figure.suptitle("sns.lineplot(data=df, x=state , y=price  )"  )
# Display the plot
g.figure.show()
#g.figure.clear()

g=sns.barplot(data=df, x="state", y="price", legend=False)
g.figure.suptitle("sns.barplot(data=df, x=state, y=price, legend=False)"  )
# Display the plot
g.figure.show()
#g.figure.clear()


g=sns.catplot(data=df, x="state", y="price")
g.figure.suptitle("sns.catplot(data=df, x=state, y=price)"  )
# Display the plot
g.figure.show() 
#g.figure.clear()

#.pivot(index="Model", columns="agency", values="price")
glue = df.pivot(columns="state", values="price")

g=sns.heatmap(glue)
g.figure.suptitle("sns.heatmap(glue)  - glue = df.pivot(columns=state, values=price)"  )
# Display the plot
g.figure.show()