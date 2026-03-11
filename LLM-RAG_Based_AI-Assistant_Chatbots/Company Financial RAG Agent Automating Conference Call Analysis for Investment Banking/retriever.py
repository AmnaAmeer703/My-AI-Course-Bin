import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
EMBEDDING_MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2"
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vectorstore")

def get_relevant_docs(query):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return db.similarity_search(query, k=4)