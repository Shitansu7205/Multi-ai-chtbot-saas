from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

# Create FastAPI application
app = FastAPI()

# Get API key
api_key = os.environ["GROQ_API_KEY"]

# Create Groq client
client = Groq(api_key=api_key)


# Request body
class ChatRequest(BaseModel):
    question: str


# Home route
@app.get("/")
def home():
    return {"message": "AI Chatbot with FastAPI is working!"}


# Chat route
@app.post("/chat")
def chat(request: ChatRequest):

    # Send question to Groq / LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": request.question
            }
        ]
    )

    # Extract actual AI answer
    answer = response.choices[0].message.content

    # Return response
    return {
        "question": request.question,
        "answer": answer
    }