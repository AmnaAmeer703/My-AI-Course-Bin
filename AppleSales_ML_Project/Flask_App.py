from flask import Flask, render_template, request
from jinja2 import Template
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('pipe3.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    Product_Name= sorted(df['Product_Name'].unique())
    Store_Name = sorted(df['Store_Name'].unique(),reverse=True)
    City = sorted(df['City'].unique())
    Country = sorted(df['Country'].unique())
    category_name = sorted(df['category_name'].unique())

    return render_template('index1.html',Product_Name=Product_Name,Store_Name=Store_Name,City=City,Country=Country,category_name=category_name)

@app.route('/predict', methods=['POST'])
def predict():

    Product_Name = request.form.get['Product_Name']
    Price = int(request.form.get['Price'])
    quantity = int(request.form.get['quantity'])
    Store_Name = request.form.get['Store_Name']
    City = request.form.get['City']
    Country = request.form.get['Country']
    category_name = request.form.get['category_name']
    sale_year = int(request.form.get['sale_year'])
    sale_month = int(request.form.get['sale_month'])
    Launch_Year = int(request.form.get['Launch_Year'])
    Launch_Month = int(request.form.get['Launch_Month'])

    print(Product_Name,Price,quantity,Store_Name,City,Country,category_name,sale_year,sale_month,Launch_Year,Launch_Month)

    input = pd.DataFrame([[Product_Name,Price,quantity,Store_Name,City,Country,category_name,sale_year,sale_month,Launch_Year,Launch_Month]])
    prediction = pipe.predict(input)
    print(prediction)
    predict = round(prediction[0],2)
    return render_template('index.html',prediction_text='your pridicted value is {}'.format(predict))
if __name__=="__main__":
    app.run(debug=True,port=5500/predict)