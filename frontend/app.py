import streamlit as st
import requests

st.title("🧠 AI Agent for Medical Staffing")

query = st.text_area("Ask about staffing availability, compliance, or matching:")

if st.button("Run Agent"):
    with st.spinner("Thinking..."):
        print(query)
        print('Sending request to agent...')
        response = requests.post("http://localhost:8000/query-agent", json={"query": query})
        print(response)
        st.success(response.json()["response"])
