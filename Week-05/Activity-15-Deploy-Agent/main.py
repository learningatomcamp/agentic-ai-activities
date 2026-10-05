
from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_tool_agent


app = FastAPI(
    title="Tools & Function Calling Agent API",
    description="FastAPI backend for a tool-using AI agent",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "message": "Tools & Function Calling Agent API is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = run_tool_agent(request.message)

    return {
        "response": result
    }
