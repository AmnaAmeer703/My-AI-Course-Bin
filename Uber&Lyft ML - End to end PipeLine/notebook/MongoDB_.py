import pandas as pd
import pymongo

df = pd.read_csv('archive/cab_rides.csv')
df.head()

df = df.dropna()

df['time_stamp'] = pd.to_datetime(df['time_stamp'],unit='ms')

df['date'] = pd.to_datetime(df['time_stamp'].dt.date)

from datetime import datetime
df['time'] = pd.to_datetime(df['time_stamp']).dt.time
df['hour'] = pd.to_datetime(df['time_stamp']).dt.hour
df['weekday'] = pd.to_datetime(df['time_stamp']).dt.weekday

df.head()

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
    
df['Period_Of_Time'] = df['hour'].apply(f)

df.drop(columns=['time_stamp','id','time','product_id'],inplace=True)

df = df.drop(columns=['date','hour','weekday'])

data = df.to_dict(orient='records')

DB_NAME = "Proj3"
COLLECTION_NAME = "Proj3-Data"
CONNECTION_URL = "mongodb+srv://xxxxxxxxxxx:xxxxxxxxxxxxx@cluster0.0aygk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(CONNECTION_URL)
data_base = client[DB_NAME]
collection = data_base[COLLECTION_NAME]

# Uploading data to MongoDB
rec = collection.insert_many(data)

# Load back data from mongodb

df = pd.DataFrame(list(collection.find()))
df.head(2)