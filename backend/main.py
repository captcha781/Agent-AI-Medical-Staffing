from fastapi import FastAPI, Request
from rag_agent import run_agent
import os
import json


app = FastAPI()

@app.post("/query-agent")
async def query_agent(request: Request):
    body = await request.json()
    query = body.get("query")
    response = run_agent(query)
    print(f"User query: {query}")
    print(f"Agent response: {response}")
    if not response:
        return {"response": "No response from agent. Please try again."}
    return {"response": response}
