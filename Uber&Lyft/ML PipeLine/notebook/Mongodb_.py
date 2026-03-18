import pandas as pd
import numpy as np
import pymongo
import warnings
warnings.filterwarnings('ignore')

df_cab = pd.read_csv('archive/cab_rides.csv')
df_weather = pd.read_csv('archive/weather.csv')

print(df_cab)

df_cab = df_cab.dropna()

df_cab['time_stamp'] = pd.to_datetime(df_cab['time_stamp'],unit='ms')

df_cab['date'] = pd.to_datetime(df_cab['time_stamp'].dt.date)

from datetime import datetime
df_cab['time'] = pd.to_datetime(df_cab['time_stamp']).dt.time
df_cab['hour'] = pd.to_datetime(df_cab['time_stamp']).dt.hour
df_cab['weekday'] = pd.to_datetime(df_cab['time_stamp']).dt.weekday

def f(x):
    if (x>4) and (x<=8):
        return 'Early Morning'
    elif (x>8) and (x<=12):
        return 'Morning'
    elif (x>12) and (x<=16):
        return 'Noon'
    elif (x>16) and (x<=20):
        return 'Evening'
    elif (x>20) and (x<=24):
        return 'Night'
    elif (x<=4):
        return 'Late Night'
    
df_cab['Period_Of_Time'] = df_cab['hour'].apply(f)

df_cab.drop(columns=['time_stamp','id','time','product_id'],inplace=True)

df = df_cab.drop(columns=['date','hour','weekday'])

data = df.to_dict(orient='records')

DB_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
CONNECTION_URL = "mongodb+srv://vikashdas770:WtwsW3eh6T3J0h6z@cluster0.0aygk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(CONNECTION_URL)
data_base = client[DB_NAME]
collection = data_base[COLLECTION_NAME]

rec = collection.insert_many(data)

df = pd.DataFrame(list(collection.find()))
df.head(2)

