import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("🦴 Facharzt Orthopädie Trainer")

openai_api_key = st.text_input("OpenAI API Key", type="password")

@st.cache_resource
def load_docs():
    docs = []
    folder = "data"
    
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    split_docs = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    return vectorstore

if openai_api_key:
    vectorstore = load_docs()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    
    mode = st.selectbox(
        "Modus",
        ["Normale Frage", "Zusammenfassung", "Prüfungssimulation"]
    )
    
    question = st.text_input("Frage")
    
    if st.button("Senden"):
        
        if mode == "Zusammenfassung":
            question = f"Erstelle eine prüfungsrelevante Zusammenfassung: {question}"
        
        if mode == "Prüfungssimulation":
            question = f"Simuliere eine mündliche Facharztprüfung: {question}"
        
        answer = qa_chain.run(question)
        
        st.write(answer)
