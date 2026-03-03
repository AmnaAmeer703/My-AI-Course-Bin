import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

import pickle
score = int

person_gender = ['male','female']
person_educaton = ['Bachelor','Associate','High School','Master','Doctorate']
person_home_ownership = ['RENT','MORTGAGE','OWN','OTHER']
loan_intent = ['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMRIMPROVEMENT']
previous_loan_defaults_on_file = ['Yes','No']

st.title('LOAN APPLICATION PREDICTOR')
pipe = pickle.load(open('loan_pipe.pkl','rb'))
model = pickle.load(open('df1.pkl','rb'))


col1,col2 = st.columns(2)

with col1:
    person_age = st.number_input('person_age',step=1,format='%i')
with col2:
    person_gender = st.selectbox('select the person_gender',sorted(person_gender))

col3,col4 = st.columns(2)

with col3:
    person_education = st.selectbox('select the person_education',sorted(person_educaton))
with col4:
    person_income = st.number_input('person_income',step=1,format='%i')

col5,col6 = st.columns(2)

with col5:
    person_emp_exp = st.number_input('person_emp_exp',step=1,format="%i")
with col6:
    person_home_ownership = st.selectbox('person_home_ownership',sorted(person_home_ownership))

col7,col8 = st.columns(2)

with col7:
    loan_amnt = st.number_input('loan_amnt')
with col8:
    loan_intent = st.selectbox('loan_intent',sorted(loan_intent))

col9,col10 = st.columns(2)

with col9:
    loan_int_rate = st.number_input('loan_int_rate')
with col10:
    loan_percent_income = st.number_input('loan_percent_income')

col11,col12,col13 = st.columns(3)

with col11:
    cb_person_cred_hist_length = st.number_input('cb_person_cred_hist_length')
with col12:
    credit_score = st.number_input('credit_score')
with col13:
    previous_loan_defaults_on_file = st.selectbox('previous_loan_defaults_on_file',sorted(previous_loan_defaults_on_file))

st.markdown("If the output is 0 its means the Loan is Approved, and if the output is 1 its means Loan is Rejected")

if st.button('Predict Loan Approval Prediction'):
    input_fare = pd.DataFrame({'person_age':[person_age],'person_gender':[person_gender],'person_education':[person_education],'person_income':[person_income],'person_emp_exp':[person_emp_exp],'person_home_ownership':[person_home_ownership],'loan_amnt':[loan_amnt],'loan_intent':[loan_intent],'loan_int_rate':[loan_int_rate],'loan_percent_income':[loan_percent_income],'cb_person_cred_hist_length':[cb_person_cred_hist_length],'credit_score':[credit_score],'previous_loan_defaults_on_file':[previous_loan_defaults_on_file]})

    result = pipe.predict(input_fare)
    st.text(result)