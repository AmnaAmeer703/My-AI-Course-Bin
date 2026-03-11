from dotenv import load_dotenv
import streamlit as st
import os
from google import genai
import google.generativeai as genai
from google.generativeai import types

load_dotenv()
GEN_API_KEY = os.getenv("GEN_API_key")
genai.configure(api_key=GEN_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")



st.title("Chatbot with Output Insights")

st.caption("To end the conversation, write 'exit' and press the button")

st.divider()

st.header("Start Conversation")

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(
        history=[
            {"role": "model", "parts": "You are a friendly chatbot"},
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )


if 'satisfaction_rating' not in  st.session_state:
    st.session_state.satisfaction_rating = 0
    st.session_state.rating_total = 0
    st.session_state.num_of_queries = 0
    st.session_state.current_rating=0

def generate_output(text):
    response = st.session_state.chat.send_message(text, stream=True)
    response.resolve()
    return response

if 'text_input_key' not in st.session_state:
    st.session_state.text_input_key = 'text_input_1'
    st.session_state.all_text=""

inp = st.text_input("What do you think?",key=st.session_state.text_input_key)
send = st.button("Send")
if send and inp:
    if inp.lower() == "exit":
        st.write("Conversation ended")
    else:
        output = generate_output(inp)
        st.write(output.text)
        if "conversation" not in st.session_state:
            st.session_state.conversation = []
        st.session_state.conversation.append({"user": inp, "bot": output.text})
        st.session_state.num_of_queries=len(st.session_state.conversation)

def update_rating():
    rating = st.session_state.current_rating
    st.session_state.satisfaction_rating = st.session_state.satisfaction_rating+rating
    st.session_state.rating_total = st.session_state.rating_total+5

st.slider(
        "Rate the response (out of 5)", 
        min_value=0, 
        max_value=5, 
        value=3, 
        key="current_rating",
        on_change=update_rating
        )

st.session_state.text_input_key = f'text_input_{int(st.session_state.text_input_key.split("_")[-1]) + 1}'

if send and inp:
    if inp.lower() == "exit":
        st.write("Conversation ended")
    else:
        output = generate_output(inp)
        st.write(output.text)
        if "conversation" not in st.session_state:
            st.session_state.conversation = []
        st.session_state.conversation.append({"user": inp, "bot": output.text})
        st.session_state.num_of_queries=len(st.session_state.conversation)

        st.slider(
        "Rate the response (out of 5)", 
        min_value=0, 
        max_value=5, 
        value=3, 
        key="current_rating",
        on_change=update_rating
        )


    st.session_state.text_input_key = f'text_input_{int(st.session_state.text_input_key.split("_")[-1]) + 1}'

    if "conversation" in st.session_state:
        st.header("Conversation History")
    for exchange in st.session_state.conversation:
        st.write(f"**You**: {exchange['user']}")
        st.write(f"**Bot**: {exchange['bot']}")
        st.divider()
        st.session_state.all_text=st.session_state.all_text+exchange['user']+" "+exchange['bot']+" "

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import streamlit as st
import re

st.title("Dashboard")
st.write("This is a dashboard for analyzing text data. (Interact for 2-3 times with the Chatbot to activate the dashboard)")


if "inputs_tok" not in st.session_state:
    st.session_state.num_of_queries = 0
    st.session_state.inputs_tok=[]
    nltk.download('stopwords')
    nltk.download('punkt')


if st.session_state.num_of_queries!=0:
    st.session_state.num_of_queries=len(st.session_state.conversation)
    value_q=st.session_state.num_of_queries
    st.subheader(f"Number of queries: {value_q}")

    stop_english=stopwords.words('english')
    text=st.session_state.all_text
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    st.session_state.inputs_tok=word_tokenize(cleaned_text)
    out=[]
    out = [word.capitalize() for word in st.session_state.inputs_tok if word.lower() not in stop_english]
    word_count = Counter(out)
    sorted_word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
    
    df = pd.DataFrame(list(sorted_word_count.items()), columns=['Common Topic', 'Frequency'])
    df = df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)    
    st.subheader("Common Topic")
    st.bar_chart(df[:7], x="Common Topic", y="Frequency", horizontal=True)

    col1, col2 = st.columns(2)
    if st.session_state.rating_total==0:
        satisfaction_percentage=0
    else:
        satisfaction_percentage = (st.session_state.satisfaction_rating/st.session_state.rating_total)*100
        with col1:
            st.subheader(f"Customer Satisfaction")
        with col2:
             st.markdown(
                 f"""
                 <style>
                 .circle-wrap {{
                     margin: 0 auto;
                     width: 150px;
                     height: 150px;
                     background: #f2f2f2;
                     border-radius: 50%;
                     display: grid;
                     place-items: center;
                }}
                .circle {{
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    background: conic-gradient(
                        #4caf50 {satisfaction_percentage * 3.6}deg,
                        #ddd 0deg
                    );
                    display: grid;
                    place-items: center;
                    font-size: 20px;
                    font-weight: bold;
                    color: black;
                }}
                </style>
                <div class="circle-wrap">
                    <div class="circle">
                        {satisfaction_percentage}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True)






