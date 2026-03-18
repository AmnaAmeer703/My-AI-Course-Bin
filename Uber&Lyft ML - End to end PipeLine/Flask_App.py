from flask import Flask, render_template, request
from jinja2 import Template
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('UberLyftPipeline.pkl','rb'))
df = pickle.load(open('df_cab.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    cab_type= sorted(df['cab_type'].unique())
    destination = sorted(df['destination'].unique(),reverse=True)
    source = sorted(df['source'].unique())
    name = sorted(df['name'].unique())
    Period_Of_Time = sorted(df['Period_Of_Time'].unique())

    return render_template('index1.html',cab_type=cab_type,destination=destination,source=source,name=name,Period_Of_Time=Period_Of_Time)

@app.route('/predict', methods=['GET','POST'])
def predict():

    distance = int(request.form.get['distance'])
    cab_type = request.form.get['cab_type']
    destination = request.form.get['destination']
    source = request.form.get['source']
    serge_multiplier = int(request.form.get['surge_multiplier'])
    name = request.form.get['name']
    Period_Of_Time = request.form.get['Period_Of_Time']

    print(distance,cab_type,destination,source,serge_multiplier,name,Period_Of_Time)

    input = pd.DataFrame([[distance,cab_type,destination,source,serge_multiplier,name,Period_Of_Time]])
    prediction = pipe.predict(input)
    print(prediction)
    predict = round(prediction[0],2)
    return render_template('index.html',prediction_text='your pridicted value is {}'.format(predict))
if __name__=="__main__":
    app.run(debug=True,port=5500/predict)