
# 🧠 AI-Powered Medical Staffing Assistant

A Gen-AI & Agentic system built with **FastAPI**, **Streamlit**, and **Gemini AI** to intelligently match freelance/part-time doctors with clinic needs, generate contracts, and recommend outreach strategies. Powered by **LangChain + Gemini + FAISS RAG pipeline**.

---

## 🚀 Features

### 🧑‍⚕️ 1. **General Agent**
- Handles open-ended queries about doctor availability, specialties, or locations.
- Dynamically detects if a **compliance agent** should be triggered.

### ✅ 2. **Role & Compliance Matcher Agent**
- Aligns clinic needs with regulatory/compliance constraints.
- Ensures certifications and legal requirements are met using RAG.

### 🩺 3. **Specialist Matching Agent**
- Matches clinic job descriptions (JDs) with doctors based on:
  - Location
  - Specialization
  - Availability
  - Past visiting experience

### 📩 4. **Outreach & Contract Generator Agent**
- Suggests the best outreach strategy (email/referral/job portal).
- Drafts a professional outreach message.
- Generates preliminary **contract/MoU templates** for short-term roles.

### 🔗 5. **Specialist → Outreach Chain**
- Chains Specialist Matching + Outreach Agent.
- Generates both doctor shortlist **and** contact strategy in one call.

---

## 🛠️ Tech Stack

| Layer            | Tools Used                              |
|------------------|------------------------------------------|
| Frontend         | 🖼️ Streamlit                             |
| Backend API      | ⚡ FastAPI                               |
| LLMs             | 🤖 Gemini 1.5 Flash via LangChain         |
| Embeddings       | 🔡 GoogleGenerativeAIEmbeddings           |
| Vector DB        | 🧠 FAISS (in-memory for demo)             |
| Prompt Routing   | 🧠 LLM classifier for compliance detection|
| Hosting Ready    | 🐳 Docker/localhost/Cloud ready           |

---

## 🧪 Demo Instructions

### 1️⃣ Start the backend
```bash
python backend/server.py
```

### 2️⃣ Launch the frontend
```bash
streamlit run frontend/app.py
```

---

## 📦 Project Structure

```bash
📁 backend/
│   ├── main.py                # FastAPI server with endpoints
│   ├── rag_agent.py           # All AI agent logic
│   ├── server.py              # Runs the uvicorn server instance of FastAPI
│
📁 data/
│   └── doctors.json           # (Unused now - replaced by mock API)
│
📁 frontend/
│   └── streamlit_app.py       # Interactive UI
│
.env                           # API key for Gemini
requirements.txt               # Python deps
README.md                      # You're here!
```

---

## 🔗 External Doctor Data Source

All agents fetch data dynamically from:

```plaintext
https://632012e69f82827dcf243f80.mockapi.io/api/doctors
```

> You can swap this with your own database or REST API later.

---

## 📡 API Endpoints

| Endpoint                  | Description                            |
|---------------------------|----------------------------------------|
| `POST /query-agent`       | General agent + compliance switcher    |
| `POST /specialist-agent`  | Clinic JD → doctor matching            |
| `POST /outreach-agent`    | Outreach strategy + contract drafting  |

---

## 🧠 Example Query (Specialist + Outreach)

```json
{
  "query": "Provide an actual clinical job description with all the details such as clinic name, location, etc..."
}
```

**Response:**
- 👩‍⚕️ Best doctor matches with reasons
- 📬 Outreach strategy (email or portal)
- 📄 Draft MoU

---

## ⚙️ Setup (Local)

```bash
git clone https://github.com/captcha781/Agent-AI-Medical-Staffing.git
cd Agent-AI-Medical-Staffing
pip install virtualenv
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
touch .env
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

---

## 🤖 Powered By

- [Gemini API (Google AI)](https://ai.google.dev/)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 🏁 Ideal Use Cases

- Clinics looking for **freelance doctors**
- Staffing agencies
- Compliance alignment for short-term hiring
- Contract/MoU automation
- Multi-agent RAG system demos

---

## 📬 Contact / Contributions

For issues, suggestions, or collabs:
📧 [bhuvanesh19112001@gmail.com]

---
