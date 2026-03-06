import pickle

import streamlit as st
import pandas as pd
score = int

data = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('pipe4.pkl','rb'))


sex = ['female','maale']
smoker = ['yes','no']
region = ['southeast','southwest','northwest','northeast']
st.title('INSURANCE PREDICTION')


index = st.number_input('index')
age = st.number_input('age')

sex = st.selectbox('select the sex',sorted(sex))

col4, col5 = st.columns(2)
with col4:
    bmi = st.number_input('bmi')
with col5:
    children = st.number_input('children')

smoker = st.selectbox('select the smoker',sorted(smoker))

region = st.selectbox('select the region',sorted(region))

if st.button('predict Insurance Predictibility'):
    input_insurance = pd.DataFrame({'index':[index],'age':[age],'sex':[sex],'bmi':[bmi],'children':[children],'smoker':[smoker],'region':[region]})

    result = model.predict(input_insurance)
    st.text(result)
