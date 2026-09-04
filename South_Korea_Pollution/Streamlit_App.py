import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='South Korea pollution TimeSeries Forecasting App',layout='wide')
st.title('South Korea Pollution Time Series Forecasting App')

df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


df['date'] = pd.to_datetime(df['date'])
df.set_index(['date'],inplace=True)

 # Data Visualization
st.subheader("Data Exploration and Visualization")

result = adfuller(df['pm25'])

# To Preview The Statistical Analysis ('p-value','Data Stationary or not','LAGS') on Streamlit App
adf_results = {
    'ADF Statistic': result[0],
    'p-value': result[1],
    'Num Lags': result[2],
    'Num Observations': result[3],
    'Critical Values (1%)': result[4]['1%'],
    'Critical Values (5%)': result[4]['5%'],
    'Critical Values (10%)': result[4]['10%']
}
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ADF Statistic", round(adf_results['ADF Statistic'], 4))
with col2:
    st.metric("p-value", round(adf_results['p-value'], 4))
with col3:
    st.metric("Stationary?", "Data is Sataionary" if adf_results['p-value'] < 0.05 else "No")

col4, col5 = st.columns(2)
with col4:
    st.header('Statistical Analysis')
    st.write('statistical result:',adf_results)
with col5:
    st.header('Auto Correlation Plot')
    fig, ax = plt.subplots()
    autocorrelation_plot(df['pm25'], ax=ax)
    st.pyplot(fig)

st.title("Decomposition Preview")
st.subheader("Time Series Components")

from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df['pm25'],model='add',period=30)
# Using tabs to show components
tab1, tab2, tab3, tab4 = st.tabs(["Original", "Trend", "Seasonal", "Residual"])

with tab1:
    st.line_chart(df['pm25'])
with tab2:
    st.line_chart(decompose.trend)
with tab3:
    st.line_chart(decompose.seasonal)
with tab4:
    st.line_chart(decompose.resid)


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


date = st.slider("Select number of Days", 1, 30, step=1)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = scaler.inverse_transform(train_predictions).flatten()
test_predictions = scaler.inverse_transform(test_predictions).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

forecast_period = 30
forecast = []

last_sequence = X_test[-1]

for _ in range(forecast_period):
    current_sequence = last_sequence.reshape(1, window_size, 1)
    next_prediction = model.predict(current_sequence)[0][0]
    forecast.append(next_prediction)
    last_sequence = np.append(last_sequence[1:], next_prediction)

forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
print(forecast)

st.header('Forecast Pollution in South Korea for Next 30 Days')
if st.button('forecast'):
    forecast = st.dataframe(forecast)