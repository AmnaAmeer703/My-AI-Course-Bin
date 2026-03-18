import pickle

import streamlit as st
import pandas as pd
score = int

data = pickle.load(open('df_cab.pkl','rb'))
model = pickle.load(open('UberLyftPipeline.pkl','rb'))


cab_type = ['Uber','Lyft']
destination = ['Financial District','Black Bay','Theatre District','Haymarket Square','Boston University','Fenway','Northeastern University','North End','South Station','North Station','West End','Beacon Hill']
source = ['Financial District','Black Bay','Theatre District','Haymarket Square','Boston University','Fenway','Northeastern University','North End','South Station','North Station','West End','Beacon Hill']
name = ['Shared','Lux','Lyft','Lux Black XL','Lyft XL','Lux Black','Uber XL','Black','Uber X','WAV','Black SUV','UberPool']
Period_Of_Time = ['Early Morning','Morning','Noon','Evening','Night','Late Night']


st.title('Uber & Lyft Fare PREDICTIONS')


distance= st.number_input('distance')

col2, col3 = st.columns(2)
with col2:
    cab_type = st.selectbox('select the Cab Type',sorted(cab_type))
with col3:
    destination = st.selectbox('select the Destination',sorted(destination))

source = st.selectbox('select the source',sorted(source))

surge_multiplier = st.number_input('surge_multiplier')

col6, col7 = st.columns(2)
with col6:
    name = st.selectbox('select the cab name',sorted(name))
with col7:
    Period_Of_Time = st.selectbox('select the Period Of Time',sorted(Period_Of_Time))


if st.button('Uber & Lyft Cab Fares Predictibility'):
    Cab_Fares = pd.DataFrame({'distance':[distance],'cab_type':[cab_type],'destination':[destination],'source':[source],'surge_multiplier':[surge_multiplier],'name':[name],'Period_Of_Time':[Period_Of_Time]})

    result = model.predict(Cab_Fares)
    st.text(result)
