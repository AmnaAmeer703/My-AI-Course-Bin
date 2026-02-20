import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('BrentOilPrices.csv')
print(df)

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df.set_index('Date',inplace=True)

plt.figure(figsize=(20,8))
sns.lineplot(data=df,x=df.index,y='Price')
plt.grid()
plt.show()

_, ax = plt.subplots(figsize=(20,8))
sns.boxplot(x=df.index.year,y=df.values[:,0],ax=ax)
plt.grid()
plt.show()

_, ax = plt.subplots(figsize=(20,8))
sns.boxplot(x=df.index.month,y=df.values[:,0],ax=ax)
plt.grid()
plt.show()


from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Price'])
print(result)

print('ADF Statistics: %f' % result[0])
print('p-value: %f' % result[1])

def adfuller_test(Price):
    result = adfuller(Price)
    label = ['ADF Test Statistics','p-value','$LAGS USED','Number Of Observation']
    for value,label in zip(result,label):
        print(label+':'+str(value))
    if result[1]<=0.05:
        print('Strong Evedience Against Null Hypothesis (Ho), Reject Null Hypothesis,Data is Stationary')
    else:
        print('Week Evedience Against Null Hypothesis, Data is not Stationary')

adfuller_test(df['Price'])

from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df['Price'],model='add',period=30)
print(decompose)

decompose.plot()

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Price'])

#AVERAGE YEARLY PRICE
df_yearly_sum = df.resample('A').mean()
sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
sns.lineplot(x=df_yearly_sum.index,y='Price',data=df_yearly_sum,marker='o',color='green')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.show()

#AVERAGE PRICE QUATERLY
df_quaterly_sum = df.resample('Q').mean()
sns.set(style='whitegrid')
plt.figure(figsize=(12,6))
sns.lineplot(x=df_quaterly_sum.index,y='Price',data=df_quaterly_sum,marker='o',color='green')
plt.xlabel('Quater')
plt.ylabel('Price')
plt.legend()
plt.show()


Price = df['Price'].astype(float).values.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Price)

window_size = 12
X = []
y = []
target_dates = df.index[window_size:]

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, target_dates, test_size=0.2, shuffle=False
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

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

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
plt.plot(dates_test, y_test, label='Actual Production')
plt.plot(dates_test, test_predictions, label='Predicted Production')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()

# Plot predictions
plt.figure(figsize=(10, 6))

# Plot actual data
plt.plot(df.index[window_size:], df['Price'][window_size:], label='Actual', color='blue')

# Plot training predictions
plt.plot(df.index[window_size:window_size+len(train_predictions)], train_predictions, label='Train Predictions',color='green')

# Plot testing predictions
test_pred_index = range(window_size+len(train_predictions), window_size+len(train_predictions)+len(test_predictions))
plt.plot(df.index[test_pred_index], test_predictions, label='Test Predictions',color='orange')

plt.title('Brent Oil Prices Time Series Forecasting')
plt.xlabel('Year')
plt.ylabel('Prices')
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
plt.plot(df.index[-len(Price):], scaler.inverse_transform(Price), label='Actual')
plt.plot(pd.date_range(start=df.index[-1], periods=forecast_period, freq='M'), forecast, label='Forecast')
plt.title('Brent Oil Time Series Forecasting (30-day Forecast)')
plt.xlabel('Year')
plt.ylabel('Prices')
plt.legend()
plt.show()