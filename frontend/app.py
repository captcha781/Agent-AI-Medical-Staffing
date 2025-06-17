import streamlit as st
import requests

st.title("🧠 AI Agent for Medical Staffing")

st.header("General Agent Query")
query = st.text_area("Ask about staffing availability, compliance, or matching:")

if st.button("Run Agent"):
    with st.spinner("Thinking..."):
        response = requests.post("http://localhost:8000/query-agent", json={"query": query})
        st.success(response.json()["response"])


st.markdown("---")

st.header("Specialist Matching Agent (Clinic JD Input)")
clinic_jd = st.text_area("Paste the clinic's job description here:")

if st.button("Find Specialists"):
    with st.spinner("Matching doctors..."):
        response = requests.post("http://localhost:8000/specialist-agent", json={"query": clinic_jd})
        st.success(response.json()["response"])
        
st.markdown("---")
st.header("Outreach & Contract Generator Agent")

outreach_query = st.text_area("Describe your outreach need (e.g., need a short-term dermatologist in Chennai):")

if st.button("Generate Outreach & MoU"):
    with st.spinner("Generating strategy and contract..."):
        response = requests.post("http://localhost:8000/outreach-agent", json={"query": outreach_query})
        st.success(response.json()["response"])

