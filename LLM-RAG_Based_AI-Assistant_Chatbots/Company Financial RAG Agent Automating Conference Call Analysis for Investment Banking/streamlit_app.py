from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from chatbot import process_query
from ingest import build_vectorstore
import os

st.set_page_config(page_title="Financial RAG Agent", page_icon="💬", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #F8FAFC; }
        .chat-container { border-radius: 12px; padding: 10px; }
        .header { font-size: 28px; font-weight: 600; color: #1E293B; margin-bottom: 20px; }
        .user-msg { background-color: #E0F2FE; border-radius: 8px; padding: 10px; }
        .assistant-msg { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">💬 Financial Insights Assistant</div>', unsafe_allow_html=True)

# Sidebar upload
st.sidebar.header("📄 Upload Company Reports")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("data", file.name), "wb") as f:
            f.write(file.getbuffer())
    build_vectorstore()
    st.sidebar.success("✅ Embeddings built successfully!")

if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.chat_input("Ask about company outlook, growth, or management commentary...")

if query:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(f"**{query}**")
    st.session_state["history"].append(("user", query))

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing..."):
            response = process_query(query)
            st.markdown(response)
    st.session_state["history"].append(("assistant", response))