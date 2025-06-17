import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def create_vector_store(data):
    docs = [
        f"Name: {doc['name']}\nLocation: {doc['location']}\nSpeciality: {doc['speciality']}\n"
        f"Available: {doc['availability']}\nCertifications: {', '.join(doc['certifications'])}"
        for doc in data
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.create_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


def should_use_compliance_agent(user_query):
    """Dynamically decide whether to use the compliance agent based on Gemini classification."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY)
    prompt = (
        f"You are a classifier. Decide whether this query is about matching clinic requirements "
        f"with certifications, regulatory needs, or doctor role alignment.\n"
        f"Query: \"{user_query}\"\n"
        f"Reply only with 'yes' or 'no'."
    )
    result = llm.invoke(prompt).content.strip().lower()
    
    return "yes" in result


def run_compliance_agent(clinic_query: str, vectorstore):
    print("Running compliance agent...")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    prompt = (
        f"You are a Role & Compliance Matcher Agent. Given this clinic requirement:\n"
        f"\"{clinic_query}\"\n"
        f"From the doctor list, find the best matches considering role fit, availability, "
        f"location, and required certifications. Explain why they match."
    )

    return rag_chain.run(prompt)


def run_agent(user_query):
    file_path = "/home/bhuvanesh/Workspace/AI Agent/hosp-retrieval/data/doctors.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    vectorstore = create_vector_store(data)
    
    if should_use_compliance_agent(user_query):
        return run_compliance_agent(user_query, vectorstore)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY)
    chain = RetrievalQA.from_chain_type(llm=model, retriever=vectorstore.as_retriever())

    return chain.run(user_query)


def run_specialist_agent(clinic_jd: str):
    """Specialist Matching Agent – Pairs clinic roles with doctors based on location, specialization, schedule fit, and prior visiting experience."""

    file_path = "/home/bhuvanesh/Workspace/AI Agent/hosp-retrieval/data/doctors.json"
    with open(file_path, "r") as f:
        doctors = json.load(f)

    vectorstore = create_vector_store(doctors)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    prompt = (
        f"You are a Specialist Matching Agent.\n"
        f"Your job is to find the most suitable doctors for this clinic job description:\n"
        f"\"{clinic_jd}\"\n"
        f"Base your decision on: location proximity, medical specialization, availability schedule, and previous visiting experience.\n"
        f"Return a list of suitable doctors along with a brief explanation of why each one is a good fit."
    )

    return rag_chain.run(prompt)


def run_outreach_agent(user_query: str):
    """Outreach & Contract Generator Agent"""

    file_path = "/home/bhuvanesh/Workspace/AI Agent/hosp-retrieval/data/doctors.json"
    with open(file_path, "r") as f:
        doctors = json.load(f)

    vectorstore = create_vector_store(doctors)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY)
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    prompt = (
        f"You are an Outreach & Contract Generator Agent.\n"
        f"Given the following clinic requirement:\n"
        f"\"{user_query}\"\n\n"
        f"1. List out the matching medical professionals names for the given query\n"
        f"2. Recommend the most effective contact strategy (email, referral, portal).\n"
        f"3. Draft a professional outreach message.\n"
        f"4. Generate a preliminary contract or MoU template suitable for short-term roles.\n"
        f"Structure your output clearly under the headings: Strategy, Message, Contract."
    )

    return rag_chain.run(prompt)
