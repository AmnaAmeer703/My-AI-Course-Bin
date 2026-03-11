import os
import streamlit as st
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
import torch
import torch.nn as nn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

Groq_Api_Key = os.getenv("Groq_API_Key")
Hugging_Face_Api = os.getenv("Hugging_Face_Api")
st.set_page_config(page_title="Customer Support - An LLM-powered Customer Support app")
# Load PDF and extract text
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# Initialize vector store
def create_vector_store(docs):
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = Chroma.from_documents(docs, embedding=embedding_model)
    return vector_store

def setup_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.3,"k":20}  
    )
    llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
                    temperature=0.0,
                    groq_api_key=Groq_Api_Key
                )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True  # Helps in debugging what documents are used
    )
    return qa_chain

def query_rag(qa_chain, query):
    retrieval_results = qa_chain.retriever.invoke(query)

    if not retrieval_results:
        return "I couldn't find enough relevant details in my knowledge base."

    # Combine retrieved document texts
    retrieved_text = "\n\n".join([doc.page_content for doc in retrieval_results])

    # Structured Prompt to force step-by-step answers
    structured_prompt = f"""
    You are an AI assistant that provides clear, structured responses **strictly using the provided knowledge base**.
    
    **User Question:** {query}

    **Relevant Information:**  
    {retrieved_text}

    **Instructions:**  
    - Answer **only** using the provided knowledge base.  
    - Provide a **step-by-step list** if applicable.  
    - Use **bullet points or numbered lists** where necessary.  
    - **Elaborate** on each step, making sure it's clear and informative.  
    - If no relevant information is found, state: 'I couldn't find enough details in my sources.'  

    **Answer in this format:**  
    1. **Step 1**: Explanation  
    2. **Step 2**: Explanation  
    3. **Step 3**: Explanation  

    **Bullet Points Example:**  
    * **Feature 1**: Explanation  
    * **Feature 2**: Explanation  

    **Final Answer:**
    """


    response = qa_chain(structured_prompt)

    return response

pdf_path = "FAQ's.pdf"
docs = load_documents(pdf_path)
vector_store = create_vector_store(docs)
qa_chain = setup_rag_pipeline(vector_store)

user_query = "what is the status of my current order?"
print("Response without Implementation of RAG")
print("-"*20)
llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
                    temperature=0.0,
                    groq_api_key=Groq_Api_Key
                )
print(llm.invoke(user_query))

response = query_rag(qa_chain, user_query)
print("Response After Implementation of RAG")
print("response:", response)
print("-"*20)
