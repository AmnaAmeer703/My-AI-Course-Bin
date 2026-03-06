import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('south-korean-pollution-data.csv')
print(df.head())

# 'pm25' - fine particulate matter (PM2.5) (µg/m3)
# 'pm10' as 'fine particulate matter (PM10) (µg/m3)'
# 'o3' as 'Ozone (O3) (µg/m3)'
# 'no2' as 'Nitrogen Dioxide (NO2) (ppm)'
# 'so2' as 'Sulfur Dioxide (SO2) (ppm)'
# 'co' as 'Carbon Monoxide (CO) (ppm)'

# drop unwanted column 'Unnamed: 0'

df = df.drop(columns=['Unnamed: 0'])

print('column of the data:', df.columns)

print('Column dtype:', df.dtypes)

print('Statistical Summary of Data:', df.describe().T)

print('Shape of Data:', df.shape)

print('Missing Values:', df.isnull().mean()*100)

print('Duplicated Values:', df.duplicated().sum())

df['date'] = pd.to_datetime(df['date'])
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month

# Let's Explore The Data

data = df[['date','pm25']]
data.set_index('date',inplace=True)

plt.figure(figsize=(20,8))
sns.lineplot(data=data,color='blue')
plt.grid()
plt.show()

_, ax = plt.subplots(figsize=(20,8))
sns.boxplot(x=data.index.year,y=data.values[:,0],ax=ax)
plt.grid()
plt.show()

_, ax = plt.subplots(figsize=(20,8))
sns.boxplot(x=data.index.month,y=data.values[:,0],ax=ax)
plt.grid()
plt.show()


# Let's Check Wether the data is stationary or not, Let's find pvalue. let's do Hypothesis Testing for this
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['pm25'])
print(result)

print('ADF Statistics: %f' % result[0])
print('p-value: %f' % result[1])

# After execution ADF Statistics is -9.173149
# p-value id 0.000000


def adfuller_test(pm25):
    result = adfuller(pm25)
    label = ['ADF Test Statistics','p-value','$LAGS USED','Number Of Observation']
    for value,label in zip(result,label):
        print(label+':'+str(value))
    if result[1]<=0.05:
        print('Strong Evedience Against Null Hypothesis (Ho), Reject Null Hypothesis,Data is Stationary')
    else:
        print('Week Evedience Against Null Hypothesis, Data is not Stationary')

adfuller_test(df['pm25'])

# The result after execution is:
# ADF Test Statistics:-9.173148896853732
# p-value:2.3585547320143835e-15
# $LAGS USED:51
# Number of Observation:34478
# Strong Evedience agaisnt Null Hypothesis(H0),reject null hypothesis,Data is stationary

from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df['pm25'],model='add',period=30)

print(decompose)

decompose.plot()

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['pm25'])


# Let's Explore More Data and Perform EDA on both Numerical and Categorical Columns

# Let's Check the Distribution and Denstiy of columns 'pm25','pm10','o3','no2','so2','co'

X = df[['pm25','pm10','o3','no2','so2','co']]

cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(14,5))
    sns.histplot(X[columns])

cols = X.columns
for columns in cols:
    fig, axs = plt.subplots(figsize=(14,5))
    sns.kdeplot(X[columns])

# Correlation using 'Heatmap'

print(df.corr(numeric_only=True))
sns.heatmap(df.corr(numeric_only=True),annot=True,linewidth=2,square=2,fmt='.1f')
plt.show()

# Pair Plot

sns.pairplot(df);
plt.show()


# Let's view the ghraphical representation of categorical columns of data 'City' and 'District'
print(df['City'].value_counts())

plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
ax = sns.countplot(x='City',data=df)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() *1.01, p.get_height() *1.01 ))
plt.grid()

# pie chart to see the ghrapical representation of 'District' Column
print(df['District'].value_counts())

plt.pie(df['District'].value_counts(),labels=['Seoul','Gyeonggi','Chungnam','Gangwon','Paju-Si','Seo-Gu','Nam-Gu ','Chuncheon-Si','Gyeongbuk','Jeonnam','Jeonbuk','Chungbuk'],autopct='%1.f%%',pctdistance=0.85,explode=(0.2,0,0,0,0,0,0,0,0,0,0,0))
centre_circle = plt.Circle((0,0),0.75,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


# Let's Check In Which Year South Korea have most Pollution in the air

year = df.groupby('year').max()['pm25'].sort_values(ascending=False)
print(year)

# our data shows that 2019 was the most polluted year in south korea

# Let's Check Year 2019 in which city of south korea have most pollution

year_most_pollution = df[df['year'] == 2019]

most_polluted_city = year_most_pollution.groupby('City').max()['pm25'].sort_values(ascending=False)
print(most_polluted_city)

# So the most polluted city in 2019 is "Jeongnim-Dong"

# Let's explore more on most polluted city of south korea in 2019 and see through ghraphical represention

data = year_most_pollution[year_most_pollution['City'] == 'Jeongnim-Dong']

data = data[['date','pm25']]

data.set_index('date',inplace=True)

fig = plt.figure(figsize=(15,5))
plt.plot(data,color='blue')
plt.xlabel('date')
plt.ylabel('pm25')
plt.title('Jeongnim-Dong pm25 count in year 2019')
plt.show()


#==========================================
# Let's Apply LSTM Model for prediction
pm25 = df['pm25'].astype(int).values.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(pm25)

window_size = 12
X = []
y = []
target_values = df.index[window_size:]

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, train_values, test_values = train_test_split(
    X, y, target_values, test_size=0.1, shuffle=False
)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


model = Sequential()
model.add(LSTM(units=128, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = scaler.inverse_transform(train_predictions).flatten()
test_predictions = scaler.inverse_transform(test_predictions).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

from sklearn.metrics import r2_score
rmse = np.sqrt(np.mean((y_test - test_predictions)**2))
print(f'RMSE: {rmse:.2f}')

from sklearn.metrics import r2_score
r2_score = r2_score(y_test,test_predictions)
print(r2_score)

plt.figure(figsize=(12, 6))
plt.plot(test_values, y_test, label='Actual Values')
plt.plot(test_values, test_predictions, label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('date')
plt.ylabel('pm25')
plt.legend()
plt.show()

# Plot predictions
plt.figure(figsize=(10, 6))

# Plot actual data
plt.plot(df.index[window_size:], df['pm25'][window_size:], label='Actual', color='blue')

# Plot training predictions
plt.plot(df.index[window_size:window_size+len(train_predictions)], train_predictions, label='Train Predictions',color='green')

# Plot testing predictions
test_pred_index = range(window_size+len(train_predictions), window_size+len(train_predictions)+len(test_predictions))
plt.plot(df.index[test_pred_index], test_predictions, label='Test Predictions',color='orange')

plt.title('South Korea Pollution Time Series Forecasting')
plt.xlabel('date')
plt.ylabel('pm25')
plt.legend()
plt.show()

forecast_period = 30
forecast = []

# Use the last sequence from the test data to make predictions
last_sequence = X_test[-1]

for _ in range(forecast_period):
    # Reshape the sequence to match the input shape of the model
    current_sequence = last_sequence.reshape(1, window_size, 1)
    # Predict the next value
    next_prediction = model.predict(current_sequence)[0][0]
    # Append the prediction to the forecast list
    forecast.append(next_prediction)
    # Update the last sequence by removing the first element and appending the predicted value
    last_sequence = np.append(last_sequence[1:], next_prediction)

# Inverse transform the forecasted values
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
print(forecast)
# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(pm25):], scaler.inverse_transform(pm25), label='Actual')
plt.plot(pd.date_range(start=df.index[-1], periods=forecast_period, freq='M'), forecast, label='Forecast')
plt.title('South Korea Pollution (30-day Forecast)')
plt.xlabel('date')
plt.ylabel('pm25')
plt.legend()
plt.show()

# ============================================
#           Gated Recurrent Unit (GRU)

data = pd.read_csv('south-korean-pollution-data.csv')
print(data.head())

data['date'] = pd.to_datetime(data['date'])

data = data[['date','pm25']]
data.set_index('date',inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0]) 
        y.append(data[i + time_step, 0]) 
    return np.array(X), np.array(y)

time_step = 100 
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

model1 = Sequential()
model1.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model1.add(GRU(units=50))
model1.add(Dense(units=1)) 

METRICS = metrics=['accuracy', 
                   Precision(name='precision'),
                   Recall(name='recall')]

model1.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics = METRICS)


model1.fit(X, y, epochs=10, batch_size=32)


input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
predicted_values = model1.predict(input_sequence)


predicted_values = scaler.inverse_transform(predicted_values)
print(f"The predicted pm25 for the next day is: {predicted_values[0][0]:.2f}")

# We Dump the LSTM model
import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(model,open('model.pkl','wb'))















