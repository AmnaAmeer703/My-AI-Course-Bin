from flask import Flask, render_template, request
from jinja2 import Template
import numpy as np
import pandas as pd
import pickle
model= pickle.load(open('df1.pkl','rb'))
pipe = pickle.load(open('loan_pipe.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    person_gender= sorted(model['person_gender'].unique())
    person_education = sorted(model['person_education'].unique())
    person_home_ownership= sorted(model['person_home_ownership'].unique())
    loan_intent = sorted(model['loan_intent'].unique())
    previous_loan_defaults_on_file = sorted(model['previous_loan_defaults_on_file'].unique())


    return render_template('index.html',person_gender=person_gender,person_education=person_education,person_home_ownership=person_home_ownership,loan_intent=loan_intent,previous_loan_defaults_on_file=previous_loan_defaults_on_file)

@app.route("/index", methods=['GET','POST'])
def predict():

    person_age = int(request.form.get['person_age'])
    person_gender = request.form.get['person_gender']
    person_education = request.form.get['person_education']
    person_income = int(request.form.get['person_income'])
    person_emp_exp = int(request.form.get['person_emp_exp'])
    person_home_ownership = request.form.get['Origin_Airport_ID']
    loan_amnt = int(request.form.get['loan_amnt'])
    loan_intent = request.form.get['loan_intent']
    loan_int_rate = int(request.form.get['loan_int_rate'])
    loan_percent_income = int(request.form.get['loan_percent_income'])
    cb_person_cred_hist_length = int(request.form.get['cb_person_cred_hist_length'])
    credit_score = int(request.form.get['credit_score'])
    previous_loan_defaults_on_file = request.form.get['previous_loan_defaults_on_file']

    print(person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file)
    input = pd.DataFrame([[person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file]],columns=["person_age","person_gender","person_education","person_income","person_emp_exp","person_home_ownership","loan_amnt","loan_intent","loan_int_rate","loan_percent_income","cb_person_cred_hist_length","credit_score","previous_loan_defaults_on_file"])

    prediction = pipe.predict(input)
    print(prediction)
    output = 'LOAN APPROVED' if prediction[0] == 0 else "LOAN REJECTED"

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))


if __name__=="__main__":
    app.run(debug=True)