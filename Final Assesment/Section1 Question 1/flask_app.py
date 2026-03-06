from flask import Flask, render_template, request
from jinja2 import Template
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))
df = pd.read_csv('insurance.csv')

app = Flask(__name__)

@app.route('/')
def index1():
    sex= sorted(df['sex'].unique())
    smoker = sorted(df['smoker'].unique(),reverse=True)
    region= sorted(df['region'].unique())

    return render_template('index1.html',sex=sex,smoker=smoker,region=region)

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form.get['age'])
    sex = request.form.get['sex']
    bmi = int(request.form.get['bmi'])
    children = int(request.form.get['children'])
    smoker = request.form.get['smoker']
    region = request.form.get['region']

    print(age,sex,bmi,children,smoker,region)

    input = pd.DataFrame([[age,sex,bmi,children,smoker,region]])
    prediction = pipe.predict(input)
    print(prediction)
    output = round(prediction[0],2)
    return render_template('index1.html',prediction_text='your pridicted value is {}'.format(output))
if __name__=="__main__":
    app.run(debug=False,host=5500)
