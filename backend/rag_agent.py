import os
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



def create_vector_store(data):
    # Convert doctor data into plain text chunks
    docs = [f"Name: {doc['name']}\nLocation: {doc['location']}\nSpeciality: {doc['speciality']}\nAvailable: {doc['availability']}" for doc in data]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.create_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

def run_agent(user_query):
    file_path = "/home/bhuvanesh/Workspace/AI Agent/hosp-retrieval/data/doctors.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    vectorstore = create_vector_store(data)
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY)
    chain = RetrievalQA.from_chain_type(llm=model, retriever=vectorstore.as_retriever())
    response = chain.run(user_query)
    return response
